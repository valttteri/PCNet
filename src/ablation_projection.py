

import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tueplots import bundles

# ── LLM filter ─────────────────────────────────────────────────────────────────
# Filesystem model-name strings (underscores, as they appear in logs/).
# Set to None to include every model found in the logs.
PRIORITY_LLM = [
    "meta-llama_Llama-3.2-1B-Instruct",
    "mistralai_Mistral-7B-Instruct-v0.3",
]

# ── Dataset filter ──────────────────────────────────────────────────────────────
# Dataset directory names as they appear in logs/ (e.g. "coqa", "trivia_qa").
# Set to None to include every dataset found in the logs.
PRIORITY_DATASET = [
    "coqa",
    "truthful_qa",
]

# ── Seed filter ─────────────────────────────────────────────────────────────────
# Integer seed values embedded in the path (e.g. 42, 43, 44).
# Set to None to include every seed found in the logs.
PRIORITY_SEED = [42, 43, 44]

# Only this detection method is plotted / exported.
TARGET_METHOD     = "PCNet Latent NLL"
TARGET_METHOD_KEY = f"Proposed: {TARGET_METHOD}"

# ── Colors (mirrors plot_pc.py) ────────────────────────────────────────────────
C_PCNET = "#E67800"   # pcorange — our detection method
C_ALIGN = "#732D8C"   # methodpurple — MIC alignment

DATASET_MAP = {
    "coqa":                  "CoQA",
    "rajpurkar_squad_v2":    "SQuAD v2.0",
    "trivia_qa":             "TriviaQA",
    "truthful_qa":           "TruthfulQA",
    "pminervini_HaluEval":   "HaluEval",
    "nq_open":               "NQ-Open",
}
DATASET_COLORS = {
    "coqa":                "#1f77b4",
    "rajpurkar_squad_v2":  "#ff7f0e",
    "trivia_qa":           "#2ca02c",
    "truthful_qa":         "#d62728",
    "pminervini_HaluEval": "#9467bd",
    "nq_open":             "#8c564b",
}

# ── Metric metadata ─────────────────────────────────────────────────────────────
METRIC_META = {
    "AUROC":    dict(label="AUROC",           better="higher", ylim=(0.40, 1.02)),
    "AUPRC":    dict(label="AUPRC",           better="higher", ylim=(0.40, 1.02)),
    "Best_F1":  dict(label="Best F1",         better="higher", ylim=(0.00, 1.02)),
    "FPR95":    dict(label=r"FPR@95% TPR",   better="lower",  ylim=(0.00, 1.02)),
    "Precision":dict(label="Precision",       better="higher", ylim=(0.00, 1.02)),
    "Recall":   dict(label="Recall",          better="higher", ylim=(0.00, 1.02)),
}

FRAC_PALETTE = {
    "frac_positive":       ("#2ca02c", r"cos > 0 (correct dir)"),
    "frac_strong_01":      ("#1f77b4", r"cos > 0.1 (strong)"),
    "frac_very_strong_03": ("#d62728", r"cos > 0.3 (very strong)"),
}


# ── NeurIPS styling (mirrors plot_pc.py) ───────────────────────────────────────
def _apply_style():
    plt.rcParams.update(bundles.neurips2023(rel_width=1.0, usetex=False))
    plt.rcParams.update({
        "font.size":      8,
        "axes.titlesize": 9,
        "figure.dpi":     300,
        "savefig.dpi":    300,
    })


# ── Path dimension helper ───────────────────────────────────────────────────────
DIM_RE      = re.compile(r"PCNet_Guardrail_proj_(\d+)$")
DEFAULT_DIM = 128   # logs/PCNet_Guardrail uses pc_in_channels=128


def _proj_dim(dir_name: str) -> int:
    m = DIM_RE.match(dir_name)
    return int(m.group(1)) if m else DEFAULT_DIM


# ── Path parsers ────────────────────────────────────────────────────────────────
def _parse_metrics_path(parts: list) -> tuple[str, str, int | None] | None:
    """
    Extract (model_fs, dataset, seed) from split relative path for metrics.json.
    parts[0]='logs', parts[1]=dir_name, parts[-1]='metrics.json'

    Layout A (len=6): logs/dir/seed/model/dataset/metrics.json
    Layout B (len=6): logs/dir/model/dataset/seed/metrics.json
    Layout C (len=7): logs/dir/seed/model/dataset/seed/metrics.json
    """
    if len(parts) == 6:
        if parts[2].isdigit():
            return parts[3], parts[4], int(parts[2])                    # Layout A
        else:
            seed = int(parts[4]) if parts[4].isdigit() else None
            return parts[2], parts[3], seed                              # Layout B
    elif len(parts) == 7:
        if parts[2].isdigit():
            return parts[3], parts[4], int(parts[2])                    # Layout C
        elif parts[4].isdigit():
            return parts[2], parts[3], int(parts[4])
    return None


def _parse_alignment_path(parts: list) -> tuple[str, str, int | None] | None:
    """
    Extract (model_fs, dataset, seed) from split relative path for
    mic_alignment_results.json.
    parts[0]='checkpoints', parts[1]=dir_name, parts[-1]=filename

    len=5: ckpt/dir/model/dataset/file         (no seed)
    len=6: ckpt/dir/seed/model/dataset/file    (seed first)
         or ckpt/dir/model/dataset/seed/file   (seed last)
    """
    if len(parts) == 5:
        return parts[2], parts[3], None
    elif len(parts) == 6:
        if parts[2].isdigit():
            return parts[3], parts[4], int(parts[2])
        else:
            seed = int(parts[4]) if parts[4].isdigit() else None
            return parts[2], parts[3], seed
    return None


# ── Data loaders ────────────────────────────────────────────────────────────────
def load_projection_data() -> pd.DataFrame:
    """
    Scan logs/PCNet_Guardrail* directories for metrics.json files and return a
    long-form DataFrame: proj_dim | model_fs | model | dataset | metric | value

    Uses two glob patterns (3- and 4-level wildcards) to cover all path layouts.
    """
    rows     = []
    log_root = "logs"

    pcnet_dirs = sorted(
        d for d in os.listdir(log_root)
        if d == "PCNet_Guardrail" or d.startswith("PCNet_Guardrail_proj_")
    )
    if not pcnet_dirs:
        raise FileNotFoundError(
            "No PCNet_Guardrail* directories found under logs/. "
            "Run experiments first."
        )

    for dir_name in pcnet_dirs:
        proj_dim = _proj_dim(dir_name)

        # Two patterns handle Layout A/B (3 wildcards) and Layout C (4 wildcards)
        for pattern in [
            os.path.join(log_root, dir_name, "*", "*", "*",    "metrics.json"),
            os.path.join(log_root, dir_name, "*", "*", "*", "*", "metrics.json"),
        ]:
            for filepath in glob.glob(pattern):
                parts  = filepath.split(os.sep)
                parsed = _parse_metrics_path(parts)
                if parsed is None:
                    continue
                model_fs, dataset, seed = parsed

                if PRIORITY_LLM     is not None and model_fs not in PRIORITY_LLM:
                    continue
                if PRIORITY_DATASET is not None and dataset  not in PRIORITY_DATASET:
                    continue
                if PRIORITY_SEED    is not None and seed not in PRIORITY_SEED:
                    continue

                with open(filepath) as fh:
                    raw = json.load(fh)

                vals = raw.get(TARGET_METHOD_KEY)
                if vals is None:
                    continue

                for metric in ("AUROC", "AUPRC", "FPR95", "Best_F1",
                               "Accuracy", "Precision", "Recall"):
                    v = vals.get(metric)
                    if v is not None:
                        rows.append({
                            "proj_dim": proj_dim,
                            "model_fs": model_fs,
                            "dataset":  dataset,
                            "metric":   metric,
                            "value":    float(v),
                        })

    if not rows:
        raise ValueError(
            f"No '{TARGET_METHOD_KEY}' entries found in PCNet_Guardrail* logs "
            f"(PRIORITY_LLM={PRIORITY_LLM}, PRIORITY_DATASET={PRIORITY_DATASET}, "
            f"PRIORITY_SEED={PRIORITY_SEED})."
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["proj_dim", "model_fs", "dataset", "metric"])
    df["model"] = df["model_fs"].apply(
        lambda x: x.split("_", 1)[-1] if "_" in x else x
    )
    return df


def load_alignment_data() -> pd.DataFrame:
    """
    Scan checkpoints/PCNet_Guardrail* directories for mic_alignment_results.json
    and return a long-form DataFrame: proj_dim | model_fs | model | dataset | metric | value
    """
    ckpt_root = "checkpoints"
    rows      = []

    if not os.path.isdir(ckpt_root):
        return pd.DataFrame()

    ckpt_dirs = sorted(
        d for d in os.listdir(ckpt_root)
        if d == "PCNet_Guardrail" or d.startswith("PCNet_Guardrail_proj_")
    )

    for dir_name in ckpt_dirs:
        proj_dim = _proj_dim(dir_name)

        for pattern in [
            os.path.join(ckpt_root, dir_name, "*", "*",    "mic_alignment_results.json"),
            os.path.join(ckpt_root, dir_name, "*", "*", "*", "mic_alignment_results.json"),
        ]:
            for filepath in glob.glob(pattern):
                parts  = filepath.split(os.sep)
                parsed = _parse_alignment_path(parts)
                if parsed is None:
                    continue
                model_fs, dataset, seed = parsed

                if PRIORITY_LLM     is not None and model_fs not in PRIORITY_LLM:
                    continue
                if PRIORITY_DATASET is not None and dataset  not in PRIORITY_DATASET:
                    continue
                if PRIORITY_SEED    is not None and seed not in PRIORITY_SEED:
                    continue

                with open(filepath) as fh:
                    r = json.load(fh)

                for metric in ("mean_cosine", "median_cosine", "std_cosine",
                               "frac_positive", "frac_strong_01",
                               "frac_very_strong_03", "avg_corrupted_dims"):
                    v = r.get(metric)
                    if v is not None:
                        rows.append({
                            "proj_dim": proj_dim,
                            "model_fs": model_fs,
                            "model":    model_fs.split("_", 1)[-1] if "_" in model_fs else model_fs,
                            "dataset":  dataset,
                            "metric":   metric,
                            "value":    float(v),
                        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["proj_dim", "model_fs", "dataset", "metric"])
    return df


# ── I/O helpers ─────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def _save_csv(df: pd.DataFrame, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False)
    print(f"  saved → {path}")


# ── Low-level plot helpers ───────────────────────────────────────────────────────
def _line_with_shadow(ax, dims, mean, std, color, label, **kw):
    ax.plot(dims, mean, "o-", color=color, linewidth=1.5, markersize=5,
            label=label, **kw)
    ax.fill_between(dims, mean - std, mean + std, color=color, alpha=0.15)


def _set_dim_axis(ax, dims):
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims], fontsize=7)
    ax.set_xlabel("Projection Dimension")


# ── Individual metric plots ──────────────────────────────────────────────────────
def plot_single_metric(df: pd.DataFrame, metric: str,
                       out_dir: str, png: str, csv_name: str) -> None:
    """One plot: metric vs proj_dim, mean ± shaded std across all LLMs & datasets."""
    _apply_style()
    meta = METRIC_META[metric]
    sub  = df[df["metric"] == metric]
    dims = np.array(sorted(sub["proj_dim"].unique()))

    agg = (sub.groupby("proj_dim")["value"]
             .agg(mean="mean", std="std")
             .reindex(dims).reset_index())
    agg["std"] = agg["std"].fillna(0)

    fig, ax = plt.subplots(figsize=(8.0 / 2.54, 5.5 / 2.54), layout="constrained")
    _line_with_shadow(ax, dims, agg["mean"].values, agg["std"].values,
                      C_PCNET, TARGET_METHOD)
    _set_dim_axis(ax, dims)
    ax.set_ylabel(meta["label"])
    ax.set_ylim(*meta["ylim"])
    direction = r"$\uparrow$" if meta["better"] == "higher" else r"$\downarrow$"
    ax.set_title(f"{meta['label']} vs Proj. Dim ({direction} better)", fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)
    _save(fig, out_dir, png)

    agg["metric"] = metric
    _save_csv(agg, out_dir, csv_name)


# ── AUROC breakdown by dataset ───────────────────────────────────────────────────
def plot_auroc_by_dataset(df: pd.DataFrame, out_dir: str) -> None:
    """AUROC vs proj_dim: overall mean ± std (thick) + per-dataset thin dashed lines."""
    _apply_style()
    sub      = df[df["metric"] == "AUROC"]
    dims     = np.array(sorted(sub["proj_dim"].unique()))
    datasets = sorted(sub["dataset"].unique())

    fig, ax = plt.subplots(figsize=(10.0 / 2.54, 5.5 / 2.54), layout="constrained")

    agg_all = sub.groupby("proj_dim")["value"].agg(mean="mean", std="std").reindex(dims)
    _line_with_shadow(ax, dims,
                      agg_all["mean"].values, agg_all["std"].fillna(0).values,
                      C_PCNET, "Overall mean ± std")

    csv_rows = []
    for dataset in datasets:
        ddf  = sub[sub["dataset"] == dataset]
        dagg = ddf.groupby("proj_dim")["value"].mean().reindex(dims)
        color = DATASET_COLORS.get(dataset, "#555555")
        label = DATASET_MAP.get(dataset, dataset)
        ax.plot(dims, dagg.values, "--", linewidth=1.2, alpha=0.75,
                color=color, label=label, marker="s", markersize=3)
        for dim, val in zip(dims, dagg.values):
            csv_rows.append({"proj_dim": dim, "dataset": dataset, "auroc_mean": val})

    _set_dim_axis(ax, dims)
    ax.set_ylabel("AUROC")
    ax.set_ylim(*METRIC_META["AUROC"]["ylim"])
    ax.set_title(f"AUROC vs Projection Dimension\n({TARGET_METHOD}, per-dataset breakdown)",
                 fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.9, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.4)
    _save(fig, out_dir, "07_auroc_by_dataset_vs_proj_dim.png")

    agg_all_reset = agg_all.reset_index()
    agg_all_reset.columns = ["proj_dim", "auroc_mean_all", "auroc_std_all"]
    per_ds_df = pd.DataFrame(csv_rows)
    if not per_ds_df.empty:
        wide = per_ds_df.pivot(index="proj_dim", columns="dataset",
                               values="auroc_mean").reset_index()
        wide.columns.name = None
        csv_df = agg_all_reset.merge(wide, on="proj_dim", how="left")
    else:
        csv_df = agg_all_reset
    _save_csv(csv_df, out_dir, "07_auroc_by_dataset_vs_proj_dim.csv")


# ── MIC alignment plots ──────────────────────────────────────────────────────────
def plot_cosine_alignment(adf: pd.DataFrame, out_dir: str) -> None:
    _apply_style()
    dims     = np.array(sorted(adf["proj_dim"].unique()))
    mean_sub = adf[adf["metric"] == "mean_cosine"]
    std_sub  = adf[adf["metric"] == "std_cosine"]
    mean_agg = (mean_sub.groupby("proj_dim")["value"]
                        .agg(mean="mean", std="std").reindex(dims))
    std_agg  = std_sub.groupby("proj_dim")["value"].mean().reindex(dims)

    fig, ax = plt.subplots(figsize=(8.0 / 2.54, 5.5 / 2.54), layout="constrained")
    ax.fill_between(dims,
                    mean_agg["mean"].values - std_agg.values,
                    mean_agg["mean"].values + std_agg.values,
                    color=C_ALIGN, alpha=0.18, label="±1 std (within run)")
    ax.errorbar(dims, mean_agg["mean"].values,
                yerr=mean_agg["std"].fillna(0).values,
                fmt="o-", color=C_ALIGN, linewidth=1.5, markersize=5, capsize=3,
                label="mean cosine (cross-run ± std)")
    # ax.axhline(0.1, color="gray", linestyle="--", linewidth=1, alpha=0.7,
    #            label="PURSUE threshold (>0.1)")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    _set_dim_axis(ax, dims)
    ax.set_ylabel("Mean Cosine Alignment")
    ax.set_ylim(-0.4, 1.0)
    ax.set_title(r"MIC Cosine Alignment vs Proj. Dim ($\uparrow$ better)",
                 fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)
    _save(fig, out_dir, "08_cosine_alignment_vs_proj_dim.png")

    csv_df = mean_agg.reset_index()
    csv_df.columns = ["proj_dim", "mean_cosine_mean", "mean_cosine_cross_run_std"]
    csv_df["within_run_std"] = std_agg.values
    _save_csv(csv_df, out_dir, "08_cosine_alignment_vs_proj_dim.csv")


def plot_correction_quality_fractions(adf: pd.DataFrame, out_dir: str) -> None:
    _apply_style()
    dims     = np.array(sorted(adf["proj_dim"].unique()))
    fig, ax  = plt.subplots(figsize=(8.0 / 2.54, 5.5 / 2.54), layout="constrained")
    csv_rows = []

    for metric_key, (color, label) in FRAC_PALETTE.items():
        sub = adf[adf["metric"] == metric_key]
        if sub.empty:
            continue
        agg = (sub.groupby("proj_dim")["value"]
                  .agg(mean="mean", std="std").reindex(dims))
        _line_with_shadow(ax, dims,
                          agg["mean"].values, agg["std"].fillna(0).values,
                          color, label)
        for dim, m, s in zip(dims, agg["mean"].values, agg["std"].fillna(0).values):
            csv_rows.append({"proj_dim": dim, "metric": metric_key, "mean": m, "std": s})

    _set_dim_axis(ax, dims)
    ax.set_ylabel("Fraction of Samples")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(r"MIC Correction Quality vs Proj. Dim ($\uparrow$ better)",
                 fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)
    _save(fig, out_dir, "09_correction_quality_vs_proj_dim.png")
    _save_csv(pd.DataFrame(csv_rows), out_dir, "09_correction_quality_vs_proj_dim.csv")


# ── Summary 3×3 panel ────────────────────────────────────────────────────────────
def plot_summary(df: pd.DataFrame, adf: pd.DataFrame, out_dir: str) -> None:
    _apply_style()
    KEY_METRICS   = ["AUROC", "AUPRC", "Best_F1", "FPR95", "Precision", "Recall"]
    dims          = np.array(sorted(df["proj_dim"].unique()))
    has_alignment = not adf.empty

    nrows = 3 if has_alignment else 2
    fig, axes = plt.subplots(nrows, 3,
                             figsize=(18.0 / 2.54, (14.0 if has_alignment else 9.5) / 2.54),
                             layout="constrained")
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat[:6], KEY_METRICS):
        meta = METRIC_META[metric]
        sub  = df[df["metric"] == metric]
        agg  = (sub.groupby("proj_dim")["value"]
                   .agg(mean="mean", std="std").reindex(dims))
        _line_with_shadow(ax, dims,
                          agg["mean"].values, agg["std"].fillna(0).values,
                          C_PCNET, TARGET_METHOD)
        _set_dim_axis(ax, dims)
        ax.set_ylabel(meta["label"])
        ax.set_ylim(*meta["ylim"])
        direction = r"$\uparrow$" if meta["better"] == "higher" else r"$\downarrow$"
        ax.set_title(f"{meta['label']} {direction}", fontweight="bold")
        ax.legend(fontsize=6, framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.4)

    if has_alignment:
        adims    = np.array(sorted(adf["proj_dim"].unique()))
        mean_sub = adf[adf["metric"] == "mean_cosine"]
        std_sub  = adf[adf["metric"] == "std_cosine"]
        mean_agg = (mean_sub.groupby("proj_dim")["value"]
                            .agg(mean="mean", std="std").reindex(adims))
        std_agg  = std_sub.groupby("proj_dim")["value"].mean().reindex(adims)

        ax6 = axes_flat[6]
        ax6.fill_between(adims,
                         mean_agg["mean"].values - std_agg.values,
                         mean_agg["mean"].values + std_agg.values,
                         color=C_ALIGN, alpha=0.18)
        ax6.errorbar(adims, mean_agg["mean"].values,
                     yerr=mean_agg["std"].fillna(0).values,
                     fmt="o-", color=C_ALIGN, linewidth=1.5, markersize=5, capsize=3,
                     label="mean cosine")
        ax6.axhline(0.1, color="gray", linestyle="--", linewidth=1, alpha=0.7,
                    label="PURSUE (>0.1)")
        ax6.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        _set_dim_axis(ax6, adims)
        ax6.set_ylabel("Mean Cosine Alignment")
        ax6.set_ylim(-0.4, 1.0)
        ax6.set_title(r"MIC Cosine Align. $\uparrow$", fontweight="bold")
        ax6.legend(fontsize=6, framealpha=0.9)
        ax6.grid(True, linestyle=":", alpha=0.4)

        ax7 = axes_flat[7]
        for metric_key, (color, label) in FRAC_PALETTE.items():
            sub = adf[adf["metric"] == metric_key]
            if sub.empty:
                continue
            agg = (sub.groupby("proj_dim")["value"]
                      .agg(mean="mean", std="std").reindex(adims))
            _line_with_shadow(ax7, adims,
                              agg["mean"].values, agg["std"].fillna(0).values,
                              color, label)
        _set_dim_axis(ax7, adims)
        ax7.set_ylabel("Fraction of Samples")
        ax7.set_ylim(0.0, 1.05)
        ax7.set_title(r"MIC Correction Quality $\uparrow$", fontweight="bold")
        ax7.legend(fontsize=6, framealpha=0.9)
        ax7.grid(True, linestyle=":", alpha=0.4)

        ax8 = axes_flat[8]
        corr_sub = adf[adf["metric"] == "avg_corrupted_dims"]
        if not corr_sub.empty:
            cagg = (corr_sub.groupby("proj_dim")["value"]
                            .agg(mean="mean", std="std").reindex(adims))
            _line_with_shadow(ax8, adims,
                              cagg["mean"].values, cagg["std"].fillna(0).values,
                              C_ALIGN, "avg corrupted dims")
        _set_dim_axis(ax8, adims)
        ax8.set_ylabel("Avg Corrupted Dims / Sample")
        ax8.set_title("MIC Intervened Dims", fontweight="bold")
        ax8.legend(fontsize=6, framealpha=0.9)
        ax8.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle(
        f"PCNet Projection Dimension Ablation — {TARGET_METHOD}\n"
        "(mean ± std across all LLMs & datasets)",
        fontsize=9, fontweight="bold",
    )
    _save(fig, out_dir, "00_summary_all_metrics_vs_proj_dim.png")


# ── Compact NeurIPS 2-panel figure ───────────────────────────────────────────────
def plot_compact_one_row(df: pd.DataFrame, adf: pd.DataFrame, out_dir: str) -> None:
    """
    Two-panel figure for the paper:
      (a) AUROC vs proj_dim — overall mean ± std + per-dataset dashed lines
      (b) MIC mean cosine alignment vs proj_dim (if alignment data available)
    """
    _apply_style()
    has_alignment = not adf.empty

    fig = plt.figure(figsize=(16.0 / 2.54, 5.5 / 2.54), layout="constrained")
    n_panels = 2 if has_alignment else 1
    gs = fig.add_gridspec(1, n_panels)

    # ── (a) AUROC ──────────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    sub  = df[df["metric"] == "AUROC"]
    dims = np.array(sorted(sub["proj_dim"].unique()))

    agg_all = sub.groupby("proj_dim")["value"].agg(mean="mean", std="std").reindex(dims)
    _line_with_shadow(ax_a, dims,
                      agg_all["mean"].values, agg_all["std"].fillna(0).values,
                      C_PCNET, "Overall")

    for dataset in sorted(sub["dataset"].unique()):
        ddf  = sub[sub["dataset"] == dataset]
        dagg = ddf.groupby("proj_dim")["value"].mean().reindex(dims)
        color = DATASET_COLORS.get(dataset, "#888888")
        label = DATASET_MAP.get(dataset, dataset)
        ax_a.plot(dims, dagg.values, "--", linewidth=1.0, alpha=0.70,
                  color=color, label=label, marker="s", markersize=3)

    _set_dim_axis(ax_a, dims)
    ax_a.set_ylabel("AUROC")
    ax_a.set_ylim(*METRIC_META["AUROC"]["ylim"])
    ax_a.set_title(r"(a) AUROC vs Projection Dimension ($\uparrow$)", fontweight="bold")
    ax_a.legend(fontsize=6.5, framealpha=0.9, ncol=1, loc="lower right")
    ax_a.grid(True, linestyle=":", alpha=0.4)

    # ── (b) MIC Cosine Alignment ───────────────────────────────────────────────
    if has_alignment:
        ax_b     = fig.add_subplot(gs[0, 1])
        adims    = np.array(sorted(adf["proj_dim"].unique()))
        mean_sub = adf[adf["metric"] == "mean_cosine"]
        std_sub  = adf[adf["metric"] == "std_cosine"]
        mean_agg = (mean_sub.groupby("proj_dim")["value"]
                            .agg(mean="mean", std="std").reindex(adims))
        std_agg  = std_sub.groupby("proj_dim")["value"].mean().reindex(adims)

        ax_b.fill_between(adims,
                          mean_agg["mean"].values - std_agg.values,
                          mean_agg["mean"].values + std_agg.values,
                          color=C_ALIGN, alpha=0.18)
        ax_b.errorbar(adims, mean_agg["mean"].values,
                      yerr=mean_agg["std"].fillna(0).values,
                      fmt="o-", color=C_ALIGN, linewidth=1.5, markersize=5, capsize=3,
                      label="mean cosine")
        # ax_b.axhline(0.1, color="gray", linestyle="--", linewidth=1, alpha=0.7,
        #              label="PURSUE threshold (>0.1)")
        ax_b.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        _set_dim_axis(ax_b, adims)
        ax_b.set_ylabel("Mean Cosine Alignment")
        ax_b.set_ylim(-0.4, 1.0)
        ax_b.set_title(r"(b) MIC Cosine Alignment vs Proj. Dim ($\uparrow$)",
                       fontweight="bold")
        ax_b.legend(fontsize=6.5, framealpha=0.9, loc="lower right")
        ax_b.grid(True, linestyle=":", alpha=0.4)

    _save(fig, out_dir, "compact_ablation_projection.pdf")

    # Re-create for PNG (closing resets state)
    fig = plt.figure(figsize=(16.0 / 2.54, 5.5 / 2.54), layout="constrained")
    gs  = fig.add_gridspec(1, n_panels)

    ax_a = fig.add_subplot(gs[0, 0])
    agg_all = sub.groupby("proj_dim")["value"].agg(mean="mean", std="std").reindex(dims)
    _line_with_shadow(ax_a, dims,
                      agg_all["mean"].values, agg_all["std"].fillna(0).values,
                      C_PCNET, "Overall")
    for dataset in sorted(sub["dataset"].unique()):
        ddf  = sub[sub["dataset"] == dataset]
        dagg = ddf.groupby("proj_dim")["value"].mean().reindex(dims)
        color = DATASET_COLORS.get(dataset, "#888888")
        label = DATASET_MAP.get(dataset, dataset)
        ax_a.plot(dims, dagg.values, "--", linewidth=1.0, alpha=0.70,
                  color=color, label=label, marker="s", markersize=3)
    _set_dim_axis(ax_a, dims)
    ax_a.set_ylabel("AUROC")
    ax_a.set_ylim(*METRIC_META["AUROC"]["ylim"])
    ax_a.set_title(r"(a) AUROC vs Projection Dimension ($\uparrow$)", fontweight="bold")
    ax_a.legend(fontsize=6.5, framealpha=0.9, ncol=1, loc="lower right")
    ax_a.grid(True, linestyle=":", alpha=0.4)

    if has_alignment:
        ax_b     = fig.add_subplot(gs[0, 1])
        mean_agg = (mean_sub.groupby("proj_dim")["value"]
                            .agg(mean="mean", std="std").reindex(adims))
        std_agg  = std_sub.groupby("proj_dim")["value"].mean().reindex(adims)
        ax_b.fill_between(adims,
                          mean_agg["mean"].values - std_agg.values,
                          mean_agg["mean"].values + std_agg.values,
                          color=C_ALIGN, alpha=0.18)
        ax_b.errorbar(adims, mean_agg["mean"].values,
                      yerr=mean_agg["std"].fillna(0).values,
                      fmt="o-", color=C_ALIGN, linewidth=1.5, markersize=5, capsize=3,
                      label="mean cosine")
        # ax_b.axhline(0.1, color="gray", linestyle="--", linewidth=1, alpha=0.7,
        #              label="PURSUE threshold (>0.1)")
        ax_b.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        _set_dim_axis(ax_b, adims)
        ax_b.set_ylabel("Mean Cosine Alignment")
        ax_b.set_ylim(-0.4, 1.0)
        ax_b.set_title(r"(b) MIC Cosine Alignment vs Proj. Dim ($\uparrow$)",
                       fontweight="bold")
        ax_b.legend(fontsize=6.5, framealpha=0.9, loc="lower right")
        ax_b.grid(True, linestyle=":", alpha=0.4)

    _save(fig, out_dir, "compact_ablation_projection.png")


# ── Main ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading projection-dimension ablation data …")
    print(f"  method filter  : {TARGET_METHOD}")
    print(f"  LLM filter     : {PRIORITY_LLM  or 'all'}")
    print(f"  dataset filter : {PRIORITY_DATASET or 'all'}")

    df = load_projection_data()

    dims   = sorted(df["proj_dim"].unique())
    models = sorted(df["model"].unique())
    print(
        f"  → {len(dims)} dimension(s): {dims}\n"
        f"  → {len(models)} model(s): {models}\n"
        f"  → {df['dataset'].nunique()} dataset(s), "
        f"{len(df)} data points total"
    )

    out_dir = os.path.join("evaluation_reports", "ablation_projection")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nSaving plots & CSVs to '{out_dir}/' …")

    _save_csv(df, out_dir, "raw_detection_data.csv")

    INDIVIDUAL = [
        ("AUROC",    "01_auroc_vs_proj_dim.png",     "01_auroc_vs_proj_dim.csv"),
        ("AUPRC",    "02_auprc_vs_proj_dim.png",     "02_auprc_vs_proj_dim.csv"),
        ("Best_F1",  "03_f1_vs_proj_dim.png",        "03_f1_vs_proj_dim.csv"),
        ("FPR95",    "04_fpr95_vs_proj_dim.png",     "04_fpr95_vs_proj_dim.csv"),
        ("Precision","05_precision_vs_proj_dim.png", "05_precision_vs_proj_dim.csv"),
        ("Recall",   "06_recall_vs_proj_dim.png",    "06_recall_vs_proj_dim.csv"),
    ]
    for metric, png, csv_name in INDIVIDUAL:
        if metric in df["metric"].values:
            plot_single_metric(df, metric, out_dir, png, csv_name)
        else:
            print(f"  [skip] {metric} not in data")

    plot_auroc_by_dataset(df, out_dir)

    print("\nLoading MIC alignment data …")
    adf = load_alignment_data()
    if adf.empty:
        print("  [skip] No mic_alignment_results.json found — "
              "run mic_alignment_test.py first.")
    else:
        adims = sorted(adf["proj_dim"].unique())
        print(
            f"  → {len(adims)} dimension(s) with alignment data: {adims}\n"
            f"  → {adf['model'].nunique()} model(s), "
            f"{adf['dataset'].nunique()} dataset(s)"
        )
        _save_csv(adf, out_dir, "raw_alignment_data.csv")
        plot_cosine_alignment(adf, out_dir)
        plot_correction_quality_fractions(adf, out_dir)

    plot_summary(df, adf if not adf.empty else pd.DataFrame(), out_dir)
    plot_compact_one_row(df, adf if not adf.empty else pd.DataFrame(), out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
