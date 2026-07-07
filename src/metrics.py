

import argparse
import os
import glob
import json
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

DETECTION_ROOT = Path("paper_logs/detection")

# ── Helpers ────────────────────────────────────────────────────────────────────

def set_style():
    plt.style.use("seaborn-v0_8-paper")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def extract_metric(val_dict, key_variants):
    for k, v in val_dict.items():
        if k.lower() in [kv.lower() for kv in key_variants]:
            return v
    return np.nan


def discover_seeds(log_dir: Path):
    """Return sorted list of numeric immediate subdirectories of log_dir."""
    if not log_dir.exists():
        return []
    return sorted(
        [d for d in log_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda p: int(p.name),
    )


# ── Data loading ───────────────────────────────────────────────────────────────

def load_logs(base_dir: Path):
    """
    Load metrics from base_dir/<model>/<dataset>/metrics.json.
    Returns (merged, exp_info) dicts keyed by (model, dataset).
    """
    search_pattern = str(base_dir / "*" / "*" / "metrics.json")
    files = glob.glob(search_pattern)

    merged: dict = {}
    exp_info: dict = {}

    for file in files:
        path = Path(file)
        dataset = path.parent.name
        model   = path.parent.parent.name
        key = (model, dataset)

        with open(file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        merged.setdefault(key, {})
        exp_info.setdefault(key, raw.get("Experiment_Info", {}))

        for method_key, vals in raw.items():
            if method_key == "Experiment_Info":
                continue
            merged[key][method_key] = vals

    return merged, exp_info


def logs_to_df(merged: dict) -> pd.DataFrame:
    """Convert merged logs dict into a tidy DataFrame."""
    data = []
    for (model, dataset), methods in merged.items():
        for method_full, vals in methods.items():
            category   = method_full.split(": ")[0] if ": " in method_full else "Other"
            method_name = method_full.split(": ")[-1] if ": " in method_full else method_full
            data.append({
                "Model":         model,
                "Dataset":       dataset,
                "Method":        method_name,
                "Category":      category,
                "AUROC":         extract_metric(vals, ["auroc", "AUROC"]),
                "AUPRC":         extract_metric(vals, ["auprc", "AUPRC"]),
                "FPR95":         extract_metric(vals, ["fpr95tpr", "FPR@95TPR", "fpr95"]),
                "Best_F1":       extract_metric(vals, ["best_f1", "Best_F1"]),
                "Accuracy":      extract_metric(vals, ["accuracy", "Accuracy"]),
                "Precision":     extract_metric(vals, ["precision", "Precision"]),
                "Recall":        extract_metric(vals, ["recall", "Recall"]),
                "Num_Eval_Hall": extract_metric(vals, ["number_evaluated_as_hallucinations"]),
                "Num_Eval_Fact": extract_metric(vals, ["number_evaluated_as_factual"]),
            })
    df = pd.DataFrame(data)
    if not df.empty:
        df["Model"] = df["Model"].apply(lambda x: x.split("_", 1)[-1] if "_" in x else x)
    return df


# ── Per-seed report ────────────────────────────────────────────────────────────

METHOD_ORDER = [
    "Token NLL", "Token Entropy",
    "SEP (Semantic Entropy Probe)", "SEP (Semantic Entropy Probe)",
    "AutoFact NLI Contradict", "ChainPoll LLM Judge",
    "Latent MLP Probe", "HaloScope Recon Error",
    "PCNet Latent NLL",
]

RELEVANT_METHODS = [
    "Token NLL", "AutoFact NLI Contradict", "SEP (Semantic Entropy Probe)",
    "HaloScope Recon Error", "PCNet Latent NLL",
]

CAT_PALETTE = {
    "Baseline": "gray", "SOTA": "#1f77b4",
    "Proposed": "#2ca02c", "Other": "#d62728",
}


def generate_evaluation_report(base_dir: Path, out_dir: Path):
    """Generate all plots, CSVs, and markdown for a single seed / run."""
    merged, _ = load_logs(base_dir)
    if not merged:
        print(f"  No metrics.json files found in {base_dir} — skipping.")
        return None

    df = logs_to_df(merged)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "00_master_benchmark_data.csv", index=False)

    set_style()

    method_order = [m for m in METHOD_ORDER if m in df["Method"].unique()]

    # ── Plot 1: Overall AUROC ──────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df, y="Method", x="AUROC", hue="Category",
        order=method_order, palette=CAT_PALETTE, errorbar="sd", capsize=0.1,
    )
    plt.title("Overall Hallucination Detection Performance (Average AUROC)")
    plt.xlabel("AUROC Score (Higher is Better)")
    plt.ylabel("Detection Methodology")
    plt.xlim(0.3, 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "01_overall_auroc_comparison.png")
    df.groupby(["Category", "Method"])["AUROC"].agg(["mean", "std", "count"]).reset_index().to_csv(
        out_dir / "01_overall_auroc_comparison.csv", index=False
    )
    plt.close()

    # ── Plot 2: PCNet heatmap ─────────────────────────────────────────────────
    pcnet_df = df[df["Method"] == "PCNet Latent NLL"]
    heatmap_data = None
    if not pcnet_df.empty:
        heatmap_data = pcnet_df.pivot_table(index="Model", columns="Dataset", values="AUROC", aggfunc="mean")
        plt.figure(figsize=(8, 5))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu",
                    vmin=0.5, vmax=1.0, cbar_kws={"label": "AUROC"})
        plt.title("PCNet Latent NLL Performance (AUROC) by Model & Dataset")
        plt.tight_layout()
        plt.savefig(out_dir / "02_pcnet_heatmap.png")
        heatmap_data.to_csv(out_dir / "02_pcnet_heatmap.csv")
        plt.close()

    # ── Plot 3: Per-model breakdown ───────────────────────────────────────────
    for model in df["Model"].unique():
        model_df = df[(df["Model"] == model) & (df["Method"].isin(RELEVANT_METHODS))]
        if model_df.empty:
            continue
        plt.figure(figsize=(10, 5))
        sns.barplot(data=model_df, x="Dataset", y="AUROC", hue="Method", palette="Set2")
        plt.title(f"Relevant Approaches — {model}")
        plt.ylabel("AUROC")
        plt.xlabel("Dataset")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        safe = model.replace("/", "_")
        plt.savefig(out_dir / f"03_breakdown_model_{safe}.png")
        model_df.to_csv(out_dir / f"03_breakdown_model_{safe}.csv", index=False)
        plt.close()

    # ── Plot 3b: Per-dataset breakdown ────────────────────────────────────────
    for dataset in df["Dataset"].unique():
        ds_df = df[(df["Dataset"] == dataset) & (df["Method"].isin(RELEVANT_METHODS))]
        if ds_df.empty:
            continue
        plt.figure(figsize=(10, 5))
        sns.barplot(data=ds_df, x="Model", y="AUROC", hue="Method", palette="Set2")
        plt.title(f"Relevant Approaches — {dataset}")
        plt.ylabel("AUROC")
        plt.xlabel("LLM")
        plt.xticks(rotation=15, ha="right")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(out_dir / f"03b_breakdown_dataset_{dataset}.png")
        ds_df.to_csv(out_dir / f"03b_breakdown_dataset_{dataset}.csv", index=False)
        plt.close()

    # ── Plot 4: Overall Best F1 ───────────────────────────────────────────────
    if "Best_F1" in df.columns and not df["Best_F1"].isna().all():
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df, y="Method", x="Best_F1", hue="Category",
            order=method_order, palette=CAT_PALETTE, errorbar="sd", capsize=0.1,
        )
        plt.title("Overall Hallucination Detection Performance (Average Best F1)")
        plt.xlabel("F1 Score (Higher is Better)")
        plt.ylabel("Detection Methodology")
        plt.xlim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(out_dir / "04_overall_f1_comparison.png")
        df.groupby(["Category", "Method"])["Best_F1"].agg(["mean", "std", "count"]).reset_index().to_csv(
            out_dir / "04_overall_f1_comparison.csv", index=False
        )
        plt.close()

    # ── Plot 5: Precision vs Recall ────────────────────────────────────────────
    if "Precision" in df.columns and not df["Precision"].isna().all():
        pr_df = df.groupby(["Category", "Method"])[["Precision", "Recall"]].mean().reset_index()
        pr_melted = pr_df.melt(id_vars=["Category", "Method"], value_vars=["Precision", "Recall"],
                               var_name="Metric", value_name="Score")
        pr_order = [m for m in method_order if m in pr_melted["Method"].unique()]
        plt.figure(figsize=(12, 6))
        sns.barplot(data=pr_melted, x="Score", y="Method", hue="Metric", order=pr_order,
                    palette={"Precision": "#ff7f0e", "Recall": "#9467bd"})
        plt.title("Precision vs Recall by Detection Method (Averaged)")
        plt.xlabel("Score")
        plt.ylabel("Detection Methodology")
        plt.xlim(0.0, 1.0)
        plt.legend(title="Metric")
        plt.tight_layout()
        plt.savefig(out_dir / "05_precision_recall_comparison.png")
        pr_melted.to_csv(out_dir / "05_precision_recall_comparison.csv", index=False)
        plt.close()

    # ── Markdown tables ───────────────────────────────────────────────────────
    numeric_cols = [c for c in ["AUROC", "FPR95", "Best_F1", "Precision", "Recall", "Accuracy"]
                    if c in df.columns and not df[c].isna().all()]
    with open(out_dir / "results_tables.md", "w") as f:
        f.write("# Master Benchmark Results\n\n")

        f.write("## 1. Global Average Metrics\n\n")
        agg_df = df.groupby(["Category", "Method"])[numeric_cols].mean().round(4).reset_index()
        f.write(agg_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 2. Model Performance by Method (AUROC Averaged across datasets)\n\n")
        pivot = df.pivot_table(index=["Category", "Method"], columns="Model",
                               values="AUROC", aggfunc="mean").round(4)
        f.write(pivot.to_markdown())
        f.write("\n\n")

        if "Best_F1" in numeric_cols:
            f.write("## 2b. Model Performance by Method (Best F1 Averaged across datasets)\n\n")
            f1_pivot = df.pivot_table(index=["Category", "Method"], columns="Model",
                                      values="Best_F1", aggfunc="mean").round(4)
            f.write(f1_pivot.to_markdown())
            f.write("\n\n")

        f.write("## 3. PCNet Latent NLL Breakdown (AUROC)\n\n")
        if heatmap_data is not None:
            f.write(heatmap_data.round(4).to_markdown())
        f.write("\n\n")

        f.write("## 4. AUROC by Method × Dataset (avg across models, ↑ better)\n\n")
        auroc_pivot = _dataset_pivot_with_avg(df, "AUROC")
        auroc_pivot.index.name = "Dataset"
        f.write(auroc_pivot.to_markdown())
        f.write("\n\n")
        auroc_pivot.to_csv(out_dir / "04_auroc_by_dataset.csv")

        if "Best_F1" in numeric_cols:
            f.write("## 4b. Best F1 by Method × Dataset (avg across models, ↑ better)\n\n")
            f1_pivot = _dataset_pivot_with_avg(df, "Best_F1")
            f1_pivot.index.name = "Dataset"
            f.write(f1_pivot.to_markdown())
            f.write("\n\n")
            f1_pivot.to_csv(out_dir / "04b_f1_by_dataset.csv")

    print(f"  Reports saved → {out_dir}/")
    return df


# ── Aggregated report (mean ± std across seeds) ────────────────────────────────

def _ms(mean_val, std_val, digits=3):
    """Format mean ± std cell, or '—' if missing."""
    if pd.isna(mean_val):
        return "—"
    if pd.isna(std_val) or std_val == 0:
        return f"{mean_val:.{digits}f}"
    return f"{mean_val:.{digits}f} ± {std_val:.{digits}f}"


def _dataset_pivot_with_avg(df, metric):
    """
    Pivot: rows = Dataset, cols = Method, values = metric averaged across models.
    Appends a bold 'Average' row at the bottom showing the mean across datasets.
    """
    pivot = (
        df.pivot_table(index="Dataset", columns="Method", values=metric, aggfunc="mean")
        .round(4)
    )
    avg_row = pivot.mean().round(4)
    avg_row.name = "**Average**"
    return pd.concat([pivot, avg_row.to_frame().T])


def generate_aggregated_report(seed_dfs: list, seed_labels: list, out_dir: Path):
    """Generate mean ± std plots, CSVs, and markdown across all seeds."""
    n = len(seed_dfs)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Combined DataFrame (all seeds)
    combined = pd.concat(
        [df.assign(Seed=label) for df, label in zip(seed_dfs, seed_labels)],
        ignore_index=True,
    )
    combined.to_csv(out_dir / "00_master_benchmark_data_combined.csv", index=False)

    numeric_cols = [c for c in ["AUROC", "AUPRC", "FPR95", "Best_F1", "Accuracy", "Precision", "Recall"]
                    if c in combined.columns]

    # Per-(Model, Dataset, Method, Category) mean ± std across seeds
    group_keys = ["Model", "Dataset", "Method", "Category"]
    agg = combined.groupby(group_keys)[numeric_cols].agg(["mean", "std"]).round(4)
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()

    # Average per (Model, Method) across all datasets
    agg_model_method = (
        combined.groupby(["Model", "Method", "Category"])[numeric_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    agg_model_method.columns = [f"{col}_{stat}" for col, stat in agg_model_method.columns]
    agg_model_method = agg_model_method.reset_index()
    agg_model_method.insert(1, "Dataset", "All Datasets")

    # Average per Method across all models and datasets
    agg_method = (
        combined.groupby(["Method", "Category"])[numeric_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    agg_method.columns = [f"{col}_{stat}" for col, stat in agg_method.columns]
    agg_method = agg_method.reset_index()
    agg_method.insert(0, "Model", "All Models")
    agg_method.insert(1, "Dataset", "All Datasets")

    agg_full = pd.concat([agg, agg_model_method, agg_method], ignore_index=True)
    agg_full.to_csv(out_dir / "00_master_benchmark_data_aggregated.csv", index=False)

    set_style()
    method_order = [m for m in METHOD_ORDER if m in combined["Method"].unique()]

    # ── Plot 1: Overall AUROC (combined, errorbar = sd across seeds × models × datasets) ──
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=combined, y="Method", x="AUROC", hue="Category",
        order=method_order, palette=CAT_PALETTE, errorbar="sd", capsize=0.1,
    )
    plt.title(f"Overall AUROC — Mean ± Std ({n} seeds)")
    plt.xlabel("AUROC Score (Higher is Better)")
    plt.ylabel("Detection Methodology")
    plt.xlim(0.3, 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "01_overall_auroc_comparison.png")
    plt.close()

    # ── Plot 2: PCNet heatmap (mean across seeds) ─────────────────────────────
    pcnet = combined[combined["Method"] == "PCNet Latent NLL"]
    heatmap_data = None
    if not pcnet.empty:
        heatmap_data = pcnet.pivot_table(index="Model", columns="Dataset", values="AUROC", aggfunc="mean")
        plt.figure(figsize=(8, 5))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu",
                    vmin=0.5, vmax=1.0, cbar_kws={"label": "AUROC"})
        plt.title(f"PCNet Latent NLL AUROC — Mean across {n} seeds")
        plt.tight_layout()
        plt.savefig(out_dir / "02_pcnet_heatmap.png")
        heatmap_data.to_csv(out_dir / "02_pcnet_heatmap.csv")
        plt.close()

    # ── Plot 4: Best F1 ───────────────────────────────────────────────────────
    if "Best_F1" in combined.columns and not combined["Best_F1"].isna().all():
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=combined, y="Method", x="Best_F1", hue="Category",
            order=method_order, palette=CAT_PALETTE, errorbar="sd", capsize=0.1,
        )
        plt.title(f"Overall Best F1 — Mean ± Std ({n} seeds)")
        plt.xlabel("F1 Score (Higher is Better)")
        plt.ylabel("Detection Methodology")
        plt.xlim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(out_dir / "04_overall_f1_comparison.png")
        plt.close()

    # ── Markdown table with mean ± std ────────────────────────────────────────
    with open(out_dir / "results_tables.md", "w") as f:
        f.write(f"# Master Benchmark Results — Aggregated ({n} seeds: {', '.join(seed_labels)})\n\n")
        f.write("> All cells show **mean ± std** across seeds.\n\n")

        # Table 1: global average per method
        f.write("## 1. Global Average Metrics\n\n")
        rows = []
        for (cat, method), grp in combined.groupby(["Category", "Method"]):
            r = {"Category": cat, "Method": method}
            for col in numeric_cols:
                vals = grp[col].dropna()
                r[col] = _ms(vals.mean(), vals.std()) if len(vals) else "—"
            rows.append(r)
        f.write(pd.DataFrame(rows).to_markdown(index=False))
        f.write("\n\n")

        # Table 2: AUROC pivot (method × model) with mean ± std across seeds
        f.write("## 2. Model Performance by Method (AUROC, mean ± std across seeds)\n\n")
        models = sorted(combined["Model"].unique())
        pivot_rows = []
        for (cat, method), grp in combined.groupby(["Category", "Method"]):
            r = {"Category": cat, "Method": method}
            for model in models:
                vals = grp[grp["Model"] == model]["AUROC"].dropna()
                r[model] = _ms(vals.mean(), vals.std()) if len(vals) else "—"
            pivot_rows.append(r)
        f.write(pd.DataFrame(pivot_rows).to_markdown(index=False))
        f.write("\n\n")

        # Table 3: PCNet breakdown
        f.write("## 3. PCNet Latent NLL Breakdown (AUROC, mean across seeds)\n\n")
        if heatmap_data is not None:
            f.write(heatmap_data.round(4).to_markdown())
        f.write("\n\n")

        # Table 4: AUROC per dataset × method (mean ± std across seeds, avg across models)
        # For each (Method, Dataset) compute mean and std of AUROC across seeds × models.
        f.write("## 4. AUROC by Method × Dataset (mean ± std across seeds, avg across models, ↑ better)\n\n")
        methods_present = [m for m in METHOD_ORDER if m in combined["Method"].unique()]
        datasets_present = sorted(combined["Dataset"].unique())
        auroc_rows = []
        for dataset in datasets_present:
            r = {"Dataset": dataset}
            for method in methods_present:
                vals = combined[(combined["Dataset"] == dataset) & (combined["Method"] == method)]["AUROC"].dropna()
                r[method] = _ms(vals.mean(), vals.std()) if len(vals) else "—"
            auroc_rows.append(r)
        # Average row: mean across datasets for each method
        avg_r = {"Dataset": "**Average**"}
        for method in methods_present:
            vals = combined[combined["Method"] == method]["AUROC"].dropna()
            avg_r[method] = _ms(vals.mean(), vals.std()) if len(vals) else "—"
        auroc_rows.append(avg_r)
        f.write(pd.DataFrame(auroc_rows).to_markdown(index=False))
        f.write("\n\n")

        if "Best_F1" in numeric_cols:
            f.write("## 4b. Best F1 by Method × Dataset (mean ± std across seeds, avg across models, ↑ better)\n\n")
            f1_rows = []
            for dataset in datasets_present:
                r = {"Dataset": dataset}
                for method in methods_present:
                    vals = combined[(combined["Dataset"] == dataset) & (combined["Method"] == method)]["Best_F1"].dropna()
                    r[method] = _ms(vals.mean(), vals.std()) if len(vals) else "—"
                f1_rows.append(r)
            avg_r = {"Dataset": "**Average**"}
            for method in methods_present:
                vals = combined[combined["Method"] == method]["Best_F1"].dropna()
                avg_r[method] = _ms(vals.mean(), vals.std()) if len(vals) else "—"
            f1_rows.append(avg_r)
            f.write(pd.DataFrame(f1_rows).to_markdown(index=False))
            f.write("\n\n")

    print(f"Aggregated reports saved → {out_dir}/")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate detection evaluation reports.")
    parser.add_argument(
        "--log-dir", type=Path, default=None,
        help=(
            "Base detection-log directory containing numeric seed subdirectories. "
            f"Defaults to DETECTION_ROOT={DETECTION_ROOT}."
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("evaluation_reports"),
        help="Root output directory (default: evaluation_reports).",
    )
    args = parser.parse_args()

    log_dir = args.log_dir if args.log_dir is not None else DETECTION_ROOT
    out_root = args.out_dir

    seeds = discover_seeds(log_dir)
    if not seeds:
        print(f"No numeric seed subdirectories found in {log_dir}")
        return

    print(f"Found {len(seeds)} seed(s): {[s.name for s in seeds]}")

    seed_dfs, seed_labels = [], []
    for seed_path in seeds:
        seed_name = seed_path.name
        seed_out  = out_root / f"seed_{seed_name}"
        print(f"  Processing seed {seed_name}...")
        df = generate_evaluation_report(seed_path, seed_out)
        if df is not None:
            seed_dfs.append(df)
            seed_labels.append(seed_name)

    if not seed_dfs:
        print("No data loaded from any seed — nothing to aggregate.")
        return

    generate_aggregated_report(seed_dfs, seed_labels, out_root)
    print("Done.")


if __name__ == "__main__":
    main()
