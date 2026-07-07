"""
Data-Size Ablation for PCNet Detection
=======================================
Trains PCNet with N ∈ {50, 100, 250, 500, 750, 10000} training samples
(balanced 50/50 factual/hallucinated per size) and evaluates detection.

All other hyperparameters (depth=4, pc_in_channels=128, epochs, lr) are
fixed. The LLM is loaded ONCE per (llm, dataset) pair and reused across
all data-size conditions — only the projector and PCNet are reinitialised.

Output:
  data_size_ablation_logs/<seed>/<llm>/<dataset>/results.{json,md}
"""

import os
import gc
import json
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
from tqdm import tqdm

from guardrail import LLM_PC_Guardrail
from probabilistic_circuits import PCNet
from hf_dataloader import UnifiedDataLoader


# ------------------------------------------------------------------ config
TRAIN_SIZES = [50, 100, 250, 500, 750, 1000]

PC_IN_CHANNELS = 128
PC_DEPTH       = 4
PC_BRANCHING   = 3

DEFAULT_EPOCHS     = 50
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR         = 1e-3
DEFAULT_MARGIN     = 5.0
DEFAULT_ALPHA      = 0.8   # weight for factual NLL vs margin

N_TEST = 100   # fixed test set size (150 factual + 150 hallucinated)


# ------------------------------------------------------------------ utilities
def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fpr_at_tpr(fpr_arr, tpr_arr, tpr_level=0.95):
    """FPR at the threshold where TPR >= tpr_level."""
    idx = np.where(tpr_arr >= tpr_level)[0]
    return float(fpr_arr[idx[0]]) if len(idx) else 1.0


def evaluate_pcnet(guardrail, test_data, device, n_samples=20):
    """
    Compute PCNet NLL scores on the test set.
    Returns (scores, labels) where higher score = more likely hallucination.
    """
    guardrail.eval()
    scores, labels = [], []
    with torch.no_grad():
        for text, label in tqdm(test_data, desc="  Eval", leave=False):
            lp = guardrail(text, mpe_training=False, n_samples=n_samples)
            nll = -lp.mean().item()
            scores.append(nll)
            labels.append(int(label))
    return np.array(scores), np.array(labels)


def compute_detection_metrics(scores, labels):
    """Returns dict with AUROC, AUPRC, FPR@95, Best-F1."""
    if labels.sum() == 0 or (1 - labels).sum() == 0:
        return {"auroc": float("nan"), "auprc": float("nan"),
                "fpr95": float("nan"), "best_f1": 0.0}
    try:
        auroc = float(roc_auc_score(labels, scores))
        fpr_a, tpr_a, _ = roc_curve(labels, scores)
        fpr95 = _fpr_at_tpr(fpr_a, tpr_a)
        prec, rec, _ = precision_recall_curve(labels, scores)
        auprc = float(auc(rec, prec))
        f1_arr = 2 * prec * rec / (prec + rec + 1e-9)
        best_f1 = float(np.nanmax(f1_arr))
    except Exception as e:
        print(f"  [WARN] metric computation failed: {e}")
        auroc = auprc = fpr95 = float("nan")
        best_f1 = 0.0
    return {"auroc": auroc, "auprc": auprc, "fpr95": fpr95, "best_f1": best_f1}


def token_nll_scores(guardrail, test_data):
    """Baseline: mean token NLL via LLM logits. Computed once, reused."""
    scores, labels = [], []
    guardrail.llm.eval()
    tok = guardrail.tokenizer
    llm = guardrail.llm
    with torch.no_grad():
        for text, label in tqdm(test_data, desc="  Token-NLL baseline", leave=False):
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=512).to(llm.device)
            out = llm(**enc)
            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_ids    = enc["input_ids"][..., 1:].contiguous()
            nll = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_ids.view(-1),
            ).item()
            scores.append(nll)
            labels.append(int(label))
    return np.array(scores), np.array(labels)


# ------------------------------------------------------------------ fresh model init
def _fresh_projector(llm_hidden_size, pc_in_channels, device):
    return nn.Sequential(
        nn.Linear(llm_hidden_size, 512),
        nn.ReLU(),
        nn.Linear(512, pc_in_channels),
    ).to(device)


def _fresh_pcnet(pc_in_channels, max_depth, max_branching, seed, device):
    return PCNet(
        n_classes=1,
        max_depth=max_depth,
        max_branching=max_branching,
        seed=seed,
    ).to(device)


def init_pcnet_from_texts(guardrail, true_texts, device):
    """Initialize PCNet statistics from a small sample of factual embeddings."""
    sample = true_texts[:min(32, len(true_texts))]
    with torch.no_grad():
        embs = guardrail._get_llm_embeddings(sample)
    spatial = embs.unsqueeze(-1).unsqueeze(-1)
    guardrail.pc_prior.init_network(spatial)
    guardrail.pc_prior.to(device)


# ------------------------------------------------------------------ training
def train_one_config(
    guardrail, true_texts, fake_texts,
    epochs, batch_size, lr, margin, alpha, device,
    ckpt_path=None,
):
    """
    Contrastive training loop (same as train.py).  Saves the best-loss
    checkpoint to ckpt_path whenever a new epoch minimum is reached.
    Returns the best total loss.
    """
    optimizer = torch.optim.Adam(
        list(guardrail.pc_prior.parameters()) +
        list(guardrail.projector.parameters()),
        lr=lr, weight_decay=1e-5,
    )
    best_loss = float("inf")

    for epoch in range(epochs):
        random.shuffle(true_texts)
        random.shuffle(fake_texts)
        guardrail.train()
        ep_pos = ep_neg = ep_total = 0.0
        n_batches = 0

        max_batches = min(len(true_texts), len(fake_texts))
        for i in range(0, max_batches, batch_size):
            batch_true = true_texts[i : i + batch_size]
            batch_fake = fake_texts[i : i + batch_size]
            if not batch_true or not batch_fake:
                continue

            optimizer.zero_grad()
            lp_true = guardrail(batch_true, mpe_training=False)
            lp_fake = guardrail(batch_fake, mpe_training=False)

            loss_pos = -lp_true.mean()
            loss_neg = torch.relu(margin + lp_fake.mean() - lp_true.mean())
            loss     = alpha * loss_pos + (1.0 - alpha) * loss_neg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                guardrail.pc_prior.parameters(), max_norm=1.0
            )
            optimizer.step()

            ep_pos   += loss_pos.item()
            ep_neg   += loss_neg.item()
            ep_total += loss.item()
            n_batches += 1

        if n_batches:
            avg = ep_total / n_batches
            if avg < best_loss:
                best_loss = avg
                if ckpt_path:
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    torch.save(
                        {"pc_prior": guardrail.pc_prior,
                         "projector": guardrail.projector},
                        ckpt_path,
                    )

    guardrail.eval()
    return best_loss


# ------------------------------------------------------------------ reporting
def _markdown(rows, meta):
    lines = [
        f"# Data-Size Ablation — {meta['llm']} / {meta['dataset']}",
        "",
        f"- Seed: {meta['seed']}  ·  N test: {meta['n_test']}",
        f"- Fixed: depth={meta['pc_depth']}  ·  epochs={meta['epochs']}  "
        f"·  lr={meta['lr']}  ·  margin={meta['margin']}",
        f"- Generated: {meta['timestamp']}",
        "",
    ]
    headers = ["N train", "AUROC", "AUPRC", "FPR@95", "Best-F1", "Token-NLL AUROC (ref)"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    ref_auroc = meta.get("token_nll_auroc", float("nan"))
    for r in rows:
        m = r["metrics"]
        lines.append(
            f"| {r['n_train']:>7} "
            f"| {m['auroc']:.4f} | {m['auprc']:.4f} "
            f"| {m['fpr95']:.4f} | {m['best_f1']:.4f} "
            f"| {ref_auroc:.4f} |"
        )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ main loop
def run_ablation(guardrail, llm_name, ds_name, ds_subset,
                 test_data, train_sizes, config, args, out_base):
    device     = config.get("device", "cuda")
    epochs     = config.get("epochs", DEFAULT_EPOCHS)
    batch_size = config.get("batch_size", DEFAULT_BATCH_SIZE)
    lr         = config.get("learning_rate", DEFAULT_LR)
    margin     = config.get("margin", DEFAULT_MARGIN)
    alpha      = config.get("alpha", DEFAULT_ALPHA)
    seed       = args.seed
    llm_hidden = guardrail.llm.config.hidden_size

    # --- Token NLL baseline (computed once, independent of PCNet config) ---
    print("  Computing Token-NLL baseline...")
    tnll_scores, tnll_labels = token_nll_scores(guardrail, test_data)
    tnll_metrics = compute_detection_metrics(tnll_scores, tnll_labels)
    print(f"  Token-NLL  AUROC={tnll_metrics['auroc']:.4f}")

    rows = []
    for n_train in train_sizes:
        print(f"\n  [N={n_train}] Loading train data...")
        true_texts, fake_texts = UnifiedDataLoader.load_train_data(
            dataset_name=ds_name,
            subset=ds_subset,
            max_samples=n_train,
            seed=seed,
        )
        actual_n = len(true_texts) + len(fake_texts)
        if actual_n == 0:
            print(f"  [N={n_train}] No training data — skipping.")
            continue
        if actual_n < n_train:
            print(f"  [N={n_train}] Only {actual_n} samples available.")

        safe_llm = llm_name.replace("/", "_")
        safe_ds  = (ds_name + ("_" + ds_subset if ds_subset else "")).replace("/", "_")
        ckpt_dir = os.path.join(out_base, str(seed), safe_llm, safe_ds,
                                "checkpoints", f"n{n_train}")
        ckpt_path = os.path.join(ckpt_dir, "pcnet_best.pth")

        # Reinit projector + PCNet from scratch for each data-size condition
        _set_seed(seed)
        guardrail.projector = _fresh_projector(llm_hidden, PC_IN_CHANNELS, device)
        guardrail.pc_prior  = _fresh_pcnet(
            PC_IN_CHANNELS, PC_DEPTH, PC_BRANCHING, seed, device
        )
        init_pcnet_from_texts(guardrail, true_texts, device)

        print(f"  [N={n_train}] Training ({actual_n} samples, {epochs} epochs)...")
        best_loss = train_one_config(
            guardrail, true_texts, fake_texts,
            epochs, batch_size, lr, margin, alpha, device,
            ckpt_path=ckpt_path,
        )
        print(f"  [N={n_train}] Checkpoint → {ckpt_path}")

        print(f"  [N={n_train}] Evaluating...")
        scores, labels = evaluate_pcnet(guardrail, test_data, device)
        metrics = compute_detection_metrics(scores, labels)
        print(
            f"  [N={n_train}] AUROC={metrics['auroc']:.4f}  "
            f"AUPRC={metrics['auprc']:.4f}  FPR@95={metrics['fpr95']:.4f}  "
            f"BestF1={metrics['best_f1']:.4f}  (best_loss={best_loss:.4f})"
        )

        rows.append({
            "n_train":            actual_n,
            "n_train_requested":  n_train,
            "best_loss":          best_loss,
            "ckpt_path":          ckpt_path,
            "metrics":            metrics,
        })

    safe_llm = llm_name.replace("/", "_")
    safe_ds  = (ds_name + ("_" + ds_subset if ds_subset else "")).replace("/", "_")
    out_dir  = os.path.join(out_base, str(seed), safe_llm, safe_ds)
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "llm": llm_name, "dataset": ds_name, "subset": ds_subset,
        "seed": seed, "n_test": len(test_data),
        "pc_depth": PC_DEPTH, "pc_in_channels": PC_IN_CHANNELS,
        "epochs": epochs, "lr": lr, "margin": margin, "alpha": alpha,
        "token_nll_auroc": tnll_metrics["auroc"],
        "token_nll_metrics": tnll_metrics,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload = {"meta": meta, "rows": rows}

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(payload, f, indent=2,
                  default=lambda o: getattr(o, "tolist", lambda: str(o))())
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(_markdown(rows, meta))

    print(f"\n  → {out_dir}/results.{{json,md}}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="Eval/train JSON config.")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--n_test",     type=int, default=N_TEST)
    ap.add_argument("--train_sizes", type=str,
                    default=",".join(str(s) for s in TRAIN_SIZES),
                    help="Comma-separated list of training sizes.")
    ap.add_argument("--out_root",   type=str, default="data_size_ablation_logs")
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    _set_seed(args.seed)

    train_sizes = [int(x) for x in args.train_sizes.split(",")]
    device      = config.get("device", "cuda")
    llm_models  = config.get("llm_models", [])
    datasets    = config.get("datasets", [])

    for llm_name in llm_models:
        print(f"\n{'='*60}\nLLM: {llm_name}\n{'='*60}")
        guardrail = LLM_PC_Guardrail(
            llm_model_name=llm_name, pc_in_channels=PC_IN_CHANNELS, device=device
        )

        for ds_info in datasets:
            ds_name   = ds_info["name"]
            ds_subset = ds_info.get("subset")
            ds_split  = ds_info.get("split", "data")
            print(f"\n  Dataset: {ds_name} ({ds_subset})")

            test_data = UnifiedDataLoader.load_test_data(
                dataset_name=ds_name, subset=ds_subset, split=ds_split,
                max_samples=args.n_test, seed=args.seed,
            )
            if not test_data:
                print("  No test data — skipping.")
                continue
            print(f"  Test set: {len(test_data)} samples")

            run_ablation(
                guardrail, llm_name, ds_name, ds_subset,
                test_data, train_sizes, config, args, args.out_root,
            )

        del guardrail
        flush_memory()

    print("\nDone.")


if __name__ == "__main__":
    main()
