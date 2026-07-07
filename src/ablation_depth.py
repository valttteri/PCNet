"""
PCNet Depth Ablation
====================
Trains PCNet with max_depth ∈ {1, 2, 3, 4, 5} and evaluates detection
performance at each depth.  All other hyperparameters are fixed.

The LLM is loaded ONCE per (llm, dataset) pair and frozen; only the
projector and PCNet are reinitialised for each depth condition.

Depth interpretation (PCNet.init_network):
  depth=1  → leaves → ClassifierNode  (shallowest, no hidden layers)
  depth=2  → leaves → SumLayer → ClassifierNode
  depth=3  → leaves → Sum → Product → ClassifierNode
  depth=4  → leaves → Sum → Product → Sum → ClassifierNode  (default)
  depth=5  → leaves → Sum → Product → Sum → Product → ClassifierNode

Output:
  depth_ablation_logs/<seed>/<llm>/<dataset>/results.{json,md}
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
DEPTHS = [1, 2, 3, 4, 5]

PC_IN_CHANNELS = 128
PC_BRANCHING   = 3

DEFAULT_MAX_TRAIN  = 500   # number of training samples
DEFAULT_EPOCHS     = 50
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR         = 1e-3
DEFAULT_MARGIN     = 5.0
DEFAULT_ALPHA      = 0.8

N_TEST = 100   # fixed test set size


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
    idx = np.where(tpr_arr >= tpr_level)[0]
    return float(fpr_arr[idx[0]]) if len(idx) else 1.0


def evaluate_pcnet(guardrail, test_data, n_samples=20):
    """Returns (scores, labels); higher score = more likely hallucination."""
    guardrail.eval()
    scores, labels = [], []
    with torch.no_grad():
        for text, label in tqdm(test_data, desc="  Eval", leave=False):
            lp = guardrail(text, mpe_training=False, n_samples=n_samples)
            scores.append(-lp.mean().item())
            labels.append(int(label))
    return np.array(scores), np.array(labels)


def compute_detection_metrics(scores, labels):
    if labels.sum() == 0 or (1 - labels).sum() == 0:
        return {"auroc": float("nan"), "auprc": float("nan"),
                "fpr95": float("nan"), "best_f1": 0.0}
    try:
        auroc = float(roc_auc_score(labels, scores))
        fpr_a, tpr_a, _ = roc_curve(labels, scores)
        fpr95 = _fpr_at_tpr(fpr_a, tpr_a)
        prec, rec, _ = precision_recall_curve(labels, scores)
        auprc = float(auc(rec, prec))
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        best_f1 = float(np.nanmax(f1))
    except Exception as e:
        print(f"  [WARN] metric computation failed: {e}")
        auroc = auprc = fpr95 = float("nan")
        best_f1 = 0.0
    return {"auroc": auroc, "auprc": auprc, "fpr95": fpr95, "best_f1": best_f1}


def token_nll_scores(guardrail, test_data):
    """Baseline Token NLL — computed once per dataset, independent of depth."""
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


# ------------------------------------------------------------------ model helpers
def _fresh_projector(llm_hidden_size, pc_in_channels, device):
    return nn.Sequential(
        nn.Linear(llm_hidden_size, 512),
        nn.ReLU(),
        nn.Linear(512, pc_in_channels),
    ).to(device)


def _build_pcnet(pc_in_channels, max_depth, max_branching, seed, device):
    """Create a fresh PCNet at the specified depth."""
    return PCNet(
        n_classes=1,
        max_depth=max_depth,
        max_branching=max_branching,
        seed=seed,
    ).to(device)


def _param_count(guardrail):
    """Count trainable parameters in projector + pc_prior."""
    n_proj = sum(p.numel() for p in guardrail.projector.parameters())
    n_pc   = sum(p.numel() for p in guardrail.pc_prior.parameters())
    return n_proj, n_pc, n_proj + n_pc


def init_pcnet_from_texts(guardrail, true_texts, device):
    sample = true_texts[:min(32, len(true_texts))]
    with torch.no_grad():
        embs = guardrail._get_llm_embeddings(sample)
    guardrail.pc_prior.init_network(embs.unsqueeze(-1).unsqueeze(-1))
    guardrail.pc_prior.to(device)


# ------------------------------------------------------------------ training
def train_one_config(
    guardrail, true_texts, fake_texts,
    epochs, batch_size, lr, margin, alpha,
    ckpt_path=None,
):
    """
    Contrastive training loop. Saves the best-loss checkpoint to ckpt_path
    whenever a new epoch minimum is reached. Returns (best_loss, loss_history).
    """
    optimizer = torch.optim.Adam(
        list(guardrail.pc_prior.parameters()) +
        list(guardrail.projector.parameters()),
        lr=lr, weight_decay=1e-5,
    )
    best_loss = float("inf")
    loss_history = []

    for epoch in range(epochs):
        random.shuffle(true_texts)
        random.shuffle(fake_texts)
        guardrail.train()
        ep_total = 0.0
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

            ep_total  += loss.item()
            n_batches += 1

        if n_batches:
            avg = ep_total / n_batches
            loss_history.append(avg)
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
    return best_loss, loss_history


# ------------------------------------------------------------------ reporting
def _depth_label(d):
    mapping = {
        1: "leaves → Classifier  (depth=1)",
        2: "leaves → Sum → Classifier  (depth=2)",
        3: "leaves → Sum → Prod → Classifier  (depth=3)",
        4: "leaves → Sum → Prod → Sum → Classifier  (depth=4, default)",
        5: "leaves → Sum → Prod → Sum → Prod → Classifier  (depth=5)",
    }
    return mapping.get(d, f"depth={d}")


def _markdown(rows, meta):
    lines = [
        f"# PCNet Depth Ablation — {meta['llm']} / {meta['dataset']}",
        "",
        f"- Seed: {meta['seed']}  ·  N train: {meta['n_train']}  "
        f"·  N test: {meta['n_test']}",
        f"- Fixed: pc_in_channels={meta['pc_in_channels']}  ·  "
        f"epochs={meta['epochs']}  ·  lr={meta['lr']}",
        f"- Generated: {meta['timestamp']}",
        "",
        "Depth legend:",
    ]
    for d in DEPTHS:
        lines.append(f"  - `{d}`: {_depth_label(d)}")
    lines += [""]

    headers = ["Depth", "Arch", "PC params", "AUROC", "AUPRC",
               "FPR@95", "Best-F1", "Token-NLL AUROC (ref)"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    ref = meta.get("token_nll_auroc", float("nan"))
    for r in rows:
        m   = r["metrics"]
        arch = f"S→P→..." if r["depth"] > 2 else ("S→C" if r["depth"] == 2 else "leaves→C")
        lines.append(
            f"| {r['depth']} | {arch} | {r['n_pc_params']:,} "
            f"| {m['auroc']:.4f} | {m['auprc']:.4f} "
            f"| {m['fpr95']:.4f} | {m['best_f1']:.4f} "
            f"| {ref:.4f} |"
        )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ main loop
def run_ablation(guardrail, llm_name, ds_name, ds_subset,
                 true_texts, fake_texts, test_data,
                 depths, config, args, out_base):
    device     = config.get("device", "cuda")
    epochs     = config.get("epochs", DEFAULT_EPOCHS)
    batch_size = config.get("batch_size", DEFAULT_BATCH_SIZE)
    lr         = config.get("learning_rate", DEFAULT_LR)
    margin     = config.get("margin", DEFAULT_MARGIN)
    alpha      = config.get("alpha", DEFAULT_ALPHA)
    seed       = args.seed
    llm_hidden = guardrail.llm.config.hidden_size

    # Token NLL baseline — once per dataset
    print("  Computing Token-NLL baseline...")
    tnll_scores, tnll_labels = token_nll_scores(guardrail, test_data)
    tnll_metrics = compute_detection_metrics(tnll_scores, tnll_labels)
    print(f"  Token-NLL AUROC={tnll_metrics['auroc']:.4f}")

    rows = []
    for depth in depths:
        print(f"\n  [depth={depth}] {_depth_label(depth)}")

        safe_llm  = llm_name.replace("/", "_")
        safe_ds   = (ds_name + ("_" + ds_subset if ds_subset else "")).replace("/", "_")
        ckpt_path = os.path.join(
            out_base, str(seed), safe_llm, safe_ds,
            "checkpoints", f"depth{depth}", "pcnet_best.pth",
        )

        # Reinit projector + fresh PCNet at this depth
        _set_seed(seed)
        guardrail.projector = _fresh_projector(llm_hidden, PC_IN_CHANNELS, device)
        guardrail.pc_prior  = _build_pcnet(
            PC_IN_CHANNELS, depth, PC_BRANCHING, seed, device
        )
        init_pcnet_from_texts(guardrail, true_texts, device)

        n_proj, n_pc, n_total = _param_count(guardrail)
        print(f"    projector={n_proj:,}  pc_prior={n_pc:,}  total={n_total:,} params")

        print(f"    Training ({len(true_texts)+len(fake_texts)} samples, {epochs} epochs)...")
        best_loss, loss_hist = train_one_config(
            guardrail, true_texts, fake_texts,
            epochs, batch_size, lr, margin, alpha,
            ckpt_path=ckpt_path,
        )
        print(f"    Checkpoint → {ckpt_path}")

        print(f"    Evaluating...")
        scores, labels = evaluate_pcnet(guardrail, test_data)
        metrics = compute_detection_metrics(scores, labels)
        print(
            f"    AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
            f"FPR@95={metrics['fpr95']:.4f}  BestF1={metrics['best_f1']:.4f}  "
            f"(best_loss={best_loss:.4f})"
        )

        rows.append({
            "depth":          depth,
            "arch_label":     _depth_label(depth),
            "n_proj_params":  n_proj,
            "n_pc_params":    n_pc,
            "n_total_params": n_total,
            "best_loss":      best_loss,
            "loss_history":   loss_hist,
            "ckpt_path":      ckpt_path,
            "metrics":        metrics,
        })

    safe_llm = llm_name.replace("/", "_")
    safe_ds  = (ds_name + ("_" + ds_subset if ds_subset else "")).replace("/", "_")
    out_dir  = os.path.join(out_base, str(seed), safe_llm, safe_ds)
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "llm": llm_name, "dataset": ds_name, "subset": ds_subset,
        "seed": seed, "n_train": len(true_texts) + len(fake_texts),
        "n_test": len(test_data),
        "pc_in_channels": PC_IN_CHANNELS, "pc_branching": PC_BRANCHING,
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
    ap.add_argument("--n_train",    type=int, default=DEFAULT_MAX_TRAIN,
                    help="Training samples used at every depth (balanced).")
    ap.add_argument("--n_test",     type=int, default=N_TEST)
    ap.add_argument("--depths",     type=str,
                    default=",".join(str(d) for d in DEPTHS),
                    help="Comma-separated list of depths to evaluate.")
    ap.add_argument("--out_root",   type=str, default="depth_ablation_logs")
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    _set_seed(args.seed)

    depths     = [int(d) for d in args.depths.split(",")]
    device     = config.get("device", "cuda")
    llm_models = config.get("llm_models", [])
    datasets   = config.get("datasets", [])

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

            # Load train + test once per dataset
            true_texts, fake_texts = UnifiedDataLoader.load_train_data(
                dataset_name=ds_name, subset=ds_subset,
                max_samples=args.n_train, seed=args.seed,
            )
            if not true_texts and not fake_texts:
                print("  No training data — skipping.")
                continue
            print(f"  Train: {len(true_texts)} factual + {len(fake_texts)} hallucinated")

            test_data = UnifiedDataLoader.load_test_data(
                dataset_name=ds_name, subset=ds_subset, split=ds_split,
                max_samples=args.n_test, seed=args.seed,
            )
            if not test_data:
                print("  No test data — skipping.")
                continue
            print(f"  Test:  {len(test_data)} samples")

            run_ablation(
                guardrail, llm_name, ds_name, ds_subset,
                true_texts, fake_texts, test_data,
                depths, config, args, args.out_root,
            )

        del guardrail
        flush_memory()

    print("\nDone.")


if __name__ == "__main__":
    main()
