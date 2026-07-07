"""
Cosine Alignment Test for Marginal Imputation Correction (MIC).

Tests whether imputing anomalous dimensions (identified via per-dimension PC marginal
log-likelihoods) moves the projected embedding in the direction of the known-correct
factual embedding.

Interpretation:
  mean cosine > 0.1  -> PURSUE: PC correction signal is meaningful
  mean cosine ~ 0    -> REFINE: try larger bottleneck or full conditional inference
  mean cosine < 0    -> RETHINK: bottleneck too lossy; reframe as detection + reranking

Single-run usage (explicit config + checkpoint):
    python src/mic_alignment_test.py src/config/correction/qwen2.5_05b.json \
        --checkpoint_dir checkpoints/PCNet_Guardrail/Qwen_Qwen2.5-0.5B-Instruct/pminervini_HaluEval \
        --n_samples 50 --tau_percentile 20

Batch usage (auto-discovers all checkpoints × configs):
    python src/mic_alignment_test.py --checkpoint_dir checkpoints --n_samples 50 --tau_percentile 20
    python src/mic_alignment_test.py --force          # re-run even if results exist
"""

import glob
import os
import json
import argparse
import re

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from hf_dataloader import load_dataset

from guardrail import LLM_PC_Guardrail
from probabilistic_circuits import log_gaussian_jit, log_laplace_jit, log_student_jit

# ── Projection-dim helpers (mirrors ablation_projection.py) ───────────────────
_DIM_RE      = re.compile(r"PCNet_Guardrail_proj_(\d+)$")
_DEFAULT_DIM = 128

def _proj_dim_from_dirname(name: str) -> int:
    m = _DIM_RE.match(name)
    return int(m.group(1)) if m else _DEFAULT_DIM


# ============================================================
# Step 1: Per-Dimension Marginal Log-Likelihoods from PC Leaves
# ============================================================

def get_marginal_log_likelihoods(guardrail, z):
    """
    Compute per-dimension marginal log-likelihoods directly from PC leaf parameters.

    For each dimension i, evaluates log P(z_i) under the leaf's learned mixture
    of Gaussian / Laplace / Student-T. No backward pass required.

    Args:
        guardrail: LLM_PC_Guardrail with trained pc_prior
        z:         (B, D) projected embeddings (float32, on device)

    Returns:
        marginal_lls: (B, D) per-dimension log-likelihoods
    """
    pc = guardrail.pc_prior
    B, D = z.shape
    marginal_lls = torch.full((B, D), fill_value=-1e9, device=z.device)

    for leaf in pc.leaves:
        i = leaf.feature_idx
        if i >= D:
            continue

        vals = z[:, i]

        sigma = F.softplus(leaf.log_sigma) + 1e-6
        nu    = F.softplus(leaf.log_nu)    + 1e-3

        log_gauss = log_gaussian_jit(vals, leaf.mu, sigma)
        log_lapl  = log_laplace_jit(vals,  leaf.mu, sigma)
        log_stud  = log_student_jit(vals,  leaf.mu, sigma, nu)

        w        = torch.log_softmax(leaf.logits, dim=0)          # (3,)
        stack    = torch.stack([log_gauss, log_lapl, log_stud])   # (3, B)
        log_mix  = torch.logsumexp(w.unsqueeze(1) + stack, dim=0) # (B,)

        marginal_lls[:, i] = log_mix * torch.sigmoid(leaf.gate)

    return marginal_lls


def get_leaf_means(guardrail, D):
    """
    Return the learned mu for each leaf dimension.
    This is the PC's point estimate of the factual centre for each dimension.

    Returns:
        means: (D,) tensor
    """
    pc     = guardrail.pc_prior
    device = next(pc.parameters()).device
    means  = torch.zeros(D, device=device)

    for leaf in pc.leaves:
        if leaf.feature_idx < D:
            means[leaf.feature_idx] = leaf.mu.data

    return means


# ============================================================
# Step 2: Mean-Imputation (tractable MPE approximation)
# ============================================================

def mic_impute_mean(z, marginal_lls, leaf_means, tau_percentile=20):
    """
    Simple MIC: replace anomalous dimensions with the PC's learned mean (mu).

    This is the "mean imputation" approximation. Full conditional inference
    E[z_C | z_R] would be more principled but requires a properly structured
    smooth+decomposable PC with efficient conditional-query support.

    Args:
        z:               (B, D) projected embeddings
        marginal_lls:    (B, D) per-dim log-likelihoods
        leaf_means:      (D,)   PC learned means
        tau_percentile:  dimensions below this percentile are "corrupted"

    Returns:
        z_imputed:      (B, D) corrected embeddings
        corrupted_mask: (B, D) bool mask of imputed dimensions
    """
    tau            = torch.quantile(marginal_lls, tau_percentile / 100.0, dim=1, keepdim=True)
    corrupted_mask = marginal_lls < tau                           # (B, D)

    z_imputed = z.clone()
    z_imputed[corrupted_mask] = leaf_means.expand_as(z)[corrupted_mask]

    return z_imputed, corrupted_mask


# ============================================================
# Step 3: Cosine Alignment Metric
# ============================================================

def cosine_alignment(v_correction, v_target):
    """
    Cos-sim between the correction direction and the target direction.

        v_correction = z_imputed  - z_hallucinated   (what MIC does)
        v_target     = z_correct  - z_hallucinated   (oracle target)

    cos > 0  ->  MIC moves toward the factual embedding
    cos ~ 1  ->  MIC is near-optimal
    cos < 0  ->  MIC moves away (bottleneck too lossy)

    Returns: (B,) tensor of cosine similarities
    """
    return F.cosine_similarity(v_correction, v_target, dim=-1)


# ============================================================
# Main
# ============================================================

def run_alignment_test(config_path, checkpoint_dir,
                       n_samples=50, seed=42, tau_percentile=20,
                       pc_in_channels=None):
    """
    Run the MIC cosine alignment test for one (config, checkpoint_dir) pair.

    pc_in_channels: projection dimension to use. If None, read from config
                    (falls back to 128 if not present in config).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    with open(config_path) as f:
        config = json.load(f)

    device   = config.get("device", "cuda")
    llm_name = config.get("llm_models", ["Qwen/Qwen2.5-0.5B"])[0]

    if pc_in_channels is None:
        pc_in_channels = config.get("pc_in_channels", _DEFAULT_DIM)

    print(f"Loading guardrail  LLM={llm_name}  proj_dim={pc_in_channels}")
    guardrail = LLM_PC_Guardrail(
        llm_model_name=llm_name, pc_in_channels=pc_in_channels, device=device
    )

    pc_path = os.path.join(checkpoint_dir, "pcnet_best.pth")
    if not os.path.exists(pc_path):
        raise FileNotFoundError(f"No checkpoint found at {pc_path}")

    ckpt = torch.load(pc_path, map_location=device, weights_only=False)
    guardrail.pc_prior  = ckpt["pc_prior"].to(device)
    guardrail.projector = ckpt["projector"].to(device)
    guardrail.eval()

    print("Loading HaluEval QA paired data...")
    ds      = load_dataset("pminervini/HaluEval", "qa", split="data")
    samples = list(ds.select(range(min(n_samples, len(ds)))))

    D          = pc_in_channels
    leaf_means = get_leaf_means(guardrail, D)

    cosine_scores    = []
    n_corrupted_list = []

    print(f"\nRunning cosine alignment test on {len(samples)} samples "
          f"(tau_percentile={tau_percentile})...")

    for sample in tqdm(samples):
        question           = sample.get("question",            sample.get("input",        ""))
        hallucinated_answer = sample.get("hallucinated_answer", sample.get("wrong_answer", ""))
        correct_answer      = sample.get("right_answer",        sample.get("answer",       ""))

        if not question or not hallucinated_answer or not correct_answer:
            continue

        text_hallu   = f"{question}\nAnswer: {hallucinated_answer}"
        text_correct = f"{question}\nAnswer: {correct_answer}"

        with torch.no_grad():
            z_hallu   = guardrail._get_llm_embeddings([text_hallu])    # (1, 128)
            z_correct = guardrail._get_llm_embeddings([text_correct])  # (1, 128)

        # Per-dim marginal log-likelihoods (no backward pass)
        marginal_lls = get_marginal_log_likelihoods(guardrail, z_hallu)  # (1, 128)

        # Impute anomalous dimensions with leaf means
        z_imputed, corrupted_mask = mic_impute_mean(
            z_hallu, marginal_lls, leaf_means, tau_percentile=tau_percentile
        )

        # Alignment: does imputation move toward the correct embedding?
        v_correction = z_imputed - z_hallu    # what MIC does
        v_target     = z_correct  - z_hallu   # oracle target

        # Skip if correction is a zero vector (no dims flagged)
        if v_correction.norm() < 1e-8:
            continue

        cos = cosine_alignment(v_correction, v_target)  # (1,)
        cosine_scores.append(cos.item())
        n_corrupted_list.append(corrupted_mask.sum().item())

    cosine_scores = np.array(cosine_scores)

    mean_cos   = float(cosine_scores.mean())
    median_cos = float(np.median(cosine_scores))
    std_cos    = float(cosine_scores.std())
    frac_pos   = float((cosine_scores > 0.0).mean())
    frac_01    = float((cosine_scores > 0.1).mean())
    frac_03    = float((cosine_scores > 0.3).mean())
    avg_corr   = float(np.mean(n_corrupted_list))

    print("\n" + "=" * 60)
    print("  COSINE ALIGNMENT TEST RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated        : {len(cosine_scores)}")
    print(f"  Mean cosine alignment    : {mean_cos:.4f}  (>0 = correct direction)")
    print(f"  Median cosine alignment  : {median_cos:.4f}")
    print(f"  Std                      : {std_cos:.4f}")
    print(f"  Fraction > 0.0 (correct) : {frac_pos:.2%}")
    print(f"  Fraction > 0.1 (strong)  : {frac_01:.2%}")
    print(f"  Fraction > 0.3 (very strong): {frac_03:.2%}")
    print(f"  Avg corrupted dims/sample: {avg_corr:.1f} / {D}")
    print("=" * 60)

    if mean_cos > 0.1:
        print("\n✅ PURSUE — PC correction signal is meaningful.")
        print("   Next step: implement logit interpolation via z_imputed -> lm_head,")
        print("   or train a small correction head on top of z_imputed.")
    elif mean_cos > 0.0:
        print("\n⚠️  REFINE — Weak but positive alignment.")
        print("   Options: (a) increase bottleneck dim to 256/512,")
        print("            (b) switch to full PC conditional inference E[z_C | z_R],")
        print("            (c) use weighted mean instead of hard threshold.")
    else:
        print("\n❌ RETHINK — Correction direction does not align with factual direction.")
        print("   The 128D bottleneck may be too lossy. Consider:")
        print("   (a) Operating PC directly on higher-dim hidden states,")
        print("   (b) Reframing as detection + reranking instead of embedding correction.")

    # ── Save structured results for ablation plotting ──────────────────────
    proj_dim = config.get("pc_in_channels", D)
    results = {
        "proj_dim":          proj_dim,
        "llm":               llm_name,
        "checkpoint_dir":    checkpoint_dir,
        "n_samples":         len(cosine_scores),
        "tau_percentile":    tau_percentile,
        "mean_cosine":       round(mean_cos,   4),
        "median_cosine":     round(median_cos, 4),
        "std_cosine":        round(std_cos,    4),
        "frac_positive":     round(frac_pos,   4),
        "frac_strong_01":    round(frac_01,    4),
        "frac_very_strong_03": round(frac_03,  4),
        "avg_corrupted_dims":  round(avg_corr, 2),
        "proj_dim_total":    D,
    }
    out_path = os.path.join(checkpoint_dir, "mic_alignment_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Results saved → {out_path}")

    return cosine_scores


# ── Configuration discovery ───────────────────────────────────────────────────

def _normalize_llm_name(name: str) -> str:
    """
    Reduce an LLM name to a short lowercase key for fuzzy matching.
    Works with both HuggingFace slash paths and checkpoint dir underscore names.

    'meta-llama/Llama-3.2-1B-Instruct'  -> 'llama-3-2-1b'
    'deepseek-ai_deepseek-llm-7b-chat'  -> 'deepseek-llm-7b'
    'Qwen_Qwen3-4B-Instruct-2507'       -> 'qwen3-4b'
    """
    # strip org prefix: everything before the first "/" or "_"
    name = re.sub(r"^[^/_]+[/_]", "", name)
    name = name.lower()
    # normalise punctuation to hyphens
    name = re.sub(r"[^a-z0-9]+", "-", name)
    # strip common trailing variant suffixes
    name = re.sub(r"-(instruct|chat|base|2507|20250\d+).*$", "", name)
    return name.strip("-")


def _build_config_map(config_dir: str = "src/config/correction") -> dict[str, str]:
    """
    Returns {normalized_llm_name: config_path} for every JSON in config_dir.
    """
    mapping = {}
    for path in glob.glob(os.path.join(config_dir, "*.json")):
        try:
            with open(path) as f:
                cfg = json.load(f)
            llm = cfg["llm_models"][0]
            mapping[_normalize_llm_name(llm)] = path
        except Exception:
            pass
    return mapping


def _find_config(model_dir_name: str, config_map: dict[str, str]) -> str | None:
    """
    Given a checkpoint model directory name (e.g. 'meta-llama_Llama-3.2-1B-Instruct'),
    find the best-matching correction config path.
    """
    key = _normalize_llm_name(model_dir_name)
    # exact match first
    if key in config_map:
        return config_map[key]
    # prefix match: the checkpoint name may have extra suffixes
    for cfg_key, cfg_path in config_map.items():
        if key.startswith(cfg_key) or cfg_key.startswith(key):
            return cfg_path
    return None


def discover_configurations(
    ckpt_root: str = "checkpoints",
    config_dir: str = "src/config/correction",
) -> list[tuple[str, str, int]]:
    """
    Scan <ckpt_root>/PCNet_Guardrail*/<model>/<dataset>/pcnet_best.pth
    (excluding OLD/) and return a list of:
        (config_path, checkpoint_dir, proj_dim)
    Only includes entries where a matching correction config exists.
    """
    # Resolve to absolute paths so CWD doesn't matter
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.dirname(script_dir)   # src/ -> project root

    if not os.path.isabs(ckpt_root):
        ckpt_root = os.path.join(repo_root, ckpt_root)
    if not os.path.isabs(config_dir):
        config_dir = os.path.join(repo_root, config_dir)

    ckpt_root  = os.path.normpath(ckpt_root)
    config_dir = os.path.normpath(config_dir)

    print(f"[discover] ckpt_root  : {ckpt_root}")
    print(f"[discover] config_dir : {config_dir}")

    if not os.path.isdir(ckpt_root):
        print(f"[discover] ERROR: ckpt_root does not exist: {ckpt_root}")
        return []

    all_entries = os.listdir(ckpt_root)
    ckpt_dirs = sorted(
        d for d in all_entries
        if (d == "PCNet_Guardrail" or d.startswith("PCNet_Guardrail_proj_"))
        and d != "OLD"
    )
    print(f"[discover] all entries in ckpt_root : {all_entries}")
    print(f"[discover] PCNet_Guardrail dirs found: {ckpt_dirs}")

    config_map = _build_config_map(config_dir)
    print(f"[discover] configs loaded: {list(config_map.keys())}")

    results = []
    seen    = set()

    for dir_name in ckpt_dirs:
        proj_dim = _proj_dim_from_dirname(dir_name)
        pattern  = os.path.join(ckpt_root, dir_name, "**", "pcnet_best.pth")
        matches  = sorted(glob.glob(pattern, recursive=True))
        print(f"[discover] {dir_name}: glob matched {len(matches)} checkpoints")

        for pth in matches:
            # Index from the end: .../model_name/dataset_name/pcnet_best.pth
            # Works regardless of how many seed/intermediate dirs precede model.
            rel_parts    = os.path.relpath(pth, ckpt_root).split(os.sep)
            model_name     = rel_parts[-3]
            dataset_name   = rel_parts[-2]
            checkpoint_dir = os.path.dirname(pth)

            config_path = _find_config(model_name, config_map)
            if config_path is None:
                print(f"  [warn] no correction config found for '{model_name}' — skipping")
                continue

            key = (dir_name, model_name, dataset_name, checkpoint_dir)
            if key in seen:
                continue
            seen.add(key)
            results.append((config_path, checkpoint_dir, proj_dim))

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MIC Cosine Alignment Test.\n\n"
                    "With explicit config + checkpoint_dir: runs a single test.\n"
                    "Without positional args: auto-discovers all checkpoints × configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config",            nargs="?", default=None,
                        help="Path to correction config JSON (optional — omit for batch mode)")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Single-run: dir containing pcnet_best.pth. "
                             "Batch mode: root checkpoints directory (default: checkpoints)")
    parser.add_argument("--n_samples",      type=int, default=100)
    parser.add_argument("--tau_percentile", type=int, default=20,
                        help="Bottom-X%% of marginal LLs flagged as corrupted (default: 20)")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if mic_alignment_results.json already exists")
    args = parser.parse_args()

    if args.config and args.checkpoint_dir:
        # ── Single explicit run ──────────────────────────────────────────────
        run_alignment_test(
            args.config,
            args.checkpoint_dir,
            n_samples=args.n_samples,
            seed=args.seed,
            tau_percentile=args.tau_percentile,
        )

    else:
        # ── Batch mode: auto-discover all configurations ─────────────────────
        ckpt_root = args.checkpoint_dir or "checkpoints"
        configs = discover_configurations(ckpt_root=ckpt_root)
        if not configs:
            print("No configurations found. Check that checkpoints/PCNet_Guardrail* exist.")
            raise SystemExit(1)

        print(f"Found {len(configs)} (config, checkpoint, proj_dim) combinations.\n")

        skipped = done = failed = 0
        for i, (config_path, checkpoint_dir, proj_dim) in enumerate(configs, 1):
            result_path = os.path.join(checkpoint_dir, "mic_alignment_results.json")
            label = f"[{i}/{len(configs)}] proj={proj_dim}  {checkpoint_dir}"

            if not args.force and os.path.exists(result_path):
                print(f"  [skip] {label}")
                skipped += 1
                continue

            print(f"\n{'='*70}")
            print(f"  {label}")
            print(f"  config: {config_path}")
            print(f"{'='*70}")
            try:
                run_alignment_test(
                    config_path,
                    checkpoint_dir,
                    n_samples=args.n_samples,
                    seed=args.seed,
                    tau_percentile=args.tau_percentile,
                    pc_in_channels=proj_dim,
                )
                done += 1
            except Exception as e:
                print(f"  [error] {e}")
                failed += 1

        print(f"\n{'='*70}")
        print(f"Batch complete — done: {done}  skipped: {skipped}  failed: {failed}")
        print(f"{'='*70}")
