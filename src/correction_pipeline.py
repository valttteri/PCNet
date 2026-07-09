import os
import json
import re
import random
import importlib
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
from tqdm import tqdm

import hf_dataloader
from guardrail import LLM_PC_Guardrail
from probabilistic_circuits import log_gaussian_jit, log_laplace_jit, log_student_jit

import gc
import warnings
warnings.filterwarnings("ignore")

# Optional text quality dependencies
try:
    from rouge_score import rouge_scorer as _rouge_module
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


# ============================================================
# PLOT STYLE
# ============================================================

def set_style():
    plt.style.use("seaborn-v0_8-paper")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "serif",
        "axes.titlesize": 14, "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    })


def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# MODULAR IMPORTS
# ============================================================
import correction_baselines
import correction_pcnet
from correction_pcnet import _compute_delta_h, _get_leaf_means


# ============================================================
# METRICS HELPERS
# ============================================================

def _confusion_counts(samples, threshold):
    TP = sum(1 for s in samples if s["gt_label"] == 1 and s["nll_pre"] >= threshold)
    FP = sum(1 for s in samples if s["gt_label"] == 0 and s["nll_pre"] >= threshold)
    TN = sum(1 for s in samples if s["gt_label"] == 0 and s["nll_pre"] <  threshold)
    FN = sum(1 for s in samples if s["gt_label"] == 1 and s["nll_pre"] <  threshold)
    return TP, FP, TN, FN


def _f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0, prec, rec


def _post_confusion(method_samples, threshold):
    tf = fp_rec = fp_cor = tf_fail = 0
    for s in method_samples:
        if not s.get("corrected"):
            continue
        if s["gt_label"] == 1:
            if s["nll_post"] < threshold: tf     += 1
            else:                         tf_fail += 1
        else:
            if s["nll_post"] < threshold: fp_rec += 1
            else:                         fp_cor += 1
    return tf, tf_fail, fp_rec, fp_cor


# ============================================================
# EXTENDED METRICS
# ============================================================

def _compute_auroc_auprc(samples):
    """AUROC and AUPRC from NLL scores vs ground truth labels."""
    y_true = np.array([s["gt_label"] for s in samples])
    scores = np.array([s["nll_pre"] for s in samples])
    if len(np.unique(y_true)) < 2 or np.all(scores == scores[0]):
        return 0.5, 0.5
    auroc = roc_auc_score(y_true, scores)
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, scores)
    auprc_val = auc(rec_arr, prec_arr)
    return auroc, auprc_val


def _compute_post_auroc(samples):
    """AUROC on post-correction NLL scores."""
    y_true = np.array([s["gt_label"] for s in samples])
    scores = np.array([s["nll_post"] for s in samples])
    if len(np.unique(y_true)) < 2 or np.all(scores == scores[0]):
        return 0.5
    return roc_auc_score(y_true, scores)


def _token_f1(prediction, reference):
    """SQuAD-style token overlap F1."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0, 0.0, 0.0
    prec = len(common) / len(pred_tokens) if pred_tokens else 0
    rec = len(common) / len(ref_tokens) if ref_tokens else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec


def _exact_match(prediction, reference):
    """Normalized exact match."""
    def _normalize(s):
        return re.sub(r'\s+', ' ', s.lower().strip())
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


REFUSAL_PATTERNS = [
    r"i don'?t know", r"i cannot", r"i can'?t", r"i'm not sure",
    r"i am not sure", r"no answer", r"not available", r"unknown",
    r"i have no information", r"i'm unable", r"i am unable",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def _is_informative(text):
    """Returns True if the text is NOT a refusal."""
    return not bool(REFUSAL_RE.search(text))


def _compute_text_quality_metrics(samples, compute_bertscore=False):
    """Compute Token-F1, EM, ROUGE-L, informativeness, and optionally BERTScore."""
    token_f1s, token_precs, token_recs = [], [], []
    ems, informative = [], []
    rouge_ls = []
    preds_for_bert, refs_for_bert = [], []

    scorer = None
    if HAS_ROUGE:
        scorer = _rouge_module.RougeScorer(['rougeL'], use_stemmer=True)

    for s in samples:
        pred = s.get("text_post", "")
        ref = s.get("ref_answer", "")
        informative.append(1.0 if _is_informative(pred) else 0.0)
        if not ref:
            continue
        f1, p, r = _token_f1(pred, ref)
        token_f1s.append(f1)
        token_precs.append(p)
        token_recs.append(r)
        ems.append(_exact_match(pred, ref))
        if scorer:
            rouge_ls.append(scorer.score(ref, pred)['rougeL'].fmeasure)
        preds_for_bert.append(pred)
        refs_for_bert.append(ref)

    result = {
        "avg_token_f1":       round(float(np.mean(token_f1s)), 4) if token_f1s else 0.0,
        "avg_token_precision": round(float(np.mean(token_precs)), 4) if token_precs else 0.0,
        "avg_token_recall":   round(float(np.mean(token_recs)), 4) if token_recs else 0.0,
        "avg_exact_match":    round(float(np.mean(ems)), 4) if ems else 0.0,
        "informativeness":    round(float(np.mean(informative)), 4) if informative else 0.0,
    }
    if rouge_ls:
        result["avg_rouge_l"] = round(float(np.mean(rouge_ls)), 4)

    if compute_bertscore and preds_for_bert:
        try:
            from bert_score import score as bert_score_fn
            P, R, F_bert = bert_score_fn(preds_for_bert, refs_for_bert, lang="en", verbose=False)
            result["avg_bertscore_p"] = round(float(P.mean()), 4)
            result["avg_bertscore_r"] = round(float(R.mean()), 4)
            result["avg_bertscore_f1"] = round(float(F_bert.mean()), 4)
        except ImportError:
            pass

    return result


def _compute_gating_metrics(samples, mode, threshold):
    """Compute gating-specific novelty metrics."""
    result = {}
    total = len(samples)
    if total == 0:
        return result

    if mode == "Gated":
        # preservation_rate: % of correct (gt=0) samples left untouched
        correct = [s for s in samples if s["gt_label"] == 0]
        if correct:
            untouched = sum(1 for s in correct if not s.get("corrected", False))
            result["preservation_rate"] = round(untouched / len(correct), 4)
        # gating_efficiency: fraction of samples actually corrected
        corrected_count = sum(1 for s in samples if s.get("corrected"))
        result["gating_efficiency"] = round(corrected_count / total, 4)
    elif mode == "Blind":
        # corruption_rate: % of correct (gt=0) samples whose NLL worsened
        correct = [s for s in samples if s["gt_label"] == 0]
        if correct:
            corrupted = sum(1 for s in correct if s["nll_post"] > s["nll_pre"])
            result["corruption_rate"] = round(corrupted / len(correct), 4)
        # unnecessary_corrections: non-anomalous that got worsened
        non_anomalous = [s for s in samples if s["nll_pre"] < threshold]
        result["unnecessary_corrections"] = sum(
            1 for s in non_anomalous if s["nll_post"] > s["nll_pre"]
        )
    return result


# ============================================================
# PLOTTING
# ============================================================

def _plot_per_method(method_name, samples, base_out_dir, threshold, pre_embs, post_embs):
    
    safe_method = method_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("/", "_").replace("[", "").replace("]", "")
    out_dir = os.path.join(base_out_dir, safe_method)
    os.makedirs(out_dir, exist_ok=True)

    set_style()
    corrected = [s for s in samples if s.get("corrected")]
    if not corrected:
        return out_dir

    tp_fixed   = sum(1 for s in corrected if s["gt_label"] == 1 and s["nll_post"] <  threshold)
    tp_failed  = sum(1 for s in corrected if s["gt_label"] == 1 and s["nll_post"] >= threshold)
    fp_rec     = sum(1 for s in corrected if s["gt_label"] == 0 and s["nll_post"] <  threshold)
    fp_cor     = sum(1 for s in corrected if s["gt_label"] == 0 and s["nll_post"] >= threshold)

    # 1. Fix-rate bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(["Fixed", "Failed"], [tp_fixed, tp_failed], color=["#2ca02c", "#d62728"])
    axes[0].set_title(f"Hallucination Fix Rate — {method_name}", fontweight="bold")
    axes[0].set_ylabel("True Positives Count")
    for i, v in enumerate([tp_fixed, tp_failed]):
        axes[0].text(i, v + 0.3, str(v), ha="center", fontsize=12)

    pre_nlls  = [s["nll_pre"]  for s in corrected]
    post_nlls = [s["nll_post"] for s in corrected]
    if len(pre_nlls) > 1:
        sns.kdeplot(pre_nlls,  fill=True, color="#d62728", label="Pre-correction",  ax=axes[1])
        sns.kdeplot(post_nlls, fill=True, color="#1f77b4", label="Post-correction", ax=axes[1])
    axes[1].axvline(threshold, color="black", linestyle="--", lw=2,
                    label=f"Threshold ({threshold:.2f})")
    axes[1].set_title("NLL Score Distribution Shift (Corrected Samples)", fontweight="bold")
    axes[1].set_xlabel("PC Negative Log-Likelihood")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fix_rate_density.png"))
    plt.close()

    # 2. NLL per-sample bar chart (all samples)
    all_pre  = [s["nll_pre"]  for s in samples]
    all_post = [s["nll_post"] for s in samples]
    x = np.arange(len(samples))
    w = 0.35
    plt.figure(figsize=(max(10, len(samples) * 0.3), 5))
    plt.bar(x - w/2, all_pre,  w, label="Pre-correction",  color="#d62728", alpha=0.8)
    plt.bar(x + w/2, all_post, w, label="Post-correction", color="#2ca02c", alpha=0.8)
    plt.axhline(threshold, color="black", linestyle="--", lw=1.5, label="Threshold")
    plt.title(f"NLL Per Sample — {method_name}")
    plt.ylabel("Negative Log-Likelihood")
    plt.xlabel("Sample Index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nll_per_sample.png"))
    plt.close()

    # 3. PCA trajectory (corrected samples, if embeddings provided)
    corrected_pre  = [e for idx, e in pre_embs  if samples[idx].get("corrected")]
    corrected_post = [e for idx, e in post_embs]
    if len(corrected_pre) > 1 and len(corrected_post) == len(corrected_pre):
        all_embs = np.vstack(corrected_pre + corrected_post)
        pca      = PCA(n_components=2).fit(all_embs)
        pre_2d   = pca.transform(np.vstack(corrected_pre))
        post_2d  = pca.transform(np.vstack(corrected_post))

        plt.figure(figsize=(8, 8))
        plt.scatter(pre_2d[:, 0],  pre_2d[:, 1],  c="#d62728", label="Pre-correction",  s=80, edgecolors="w")
        plt.scatter(post_2d[:, 0], post_2d[:, 1], c="#2ca02c", label="Post-correction", s=80, edgecolors="w")
        for i in range(len(pre_2d)):
            dx = post_2d[i, 0] - pre_2d[i, 0]
            dy = post_2d[i, 1] - pre_2d[i, 1]
            plt.arrow(pre_2d[i, 0], pre_2d[i, 1], dx, dy,
                      color="gray", alpha=0.5, width=0.005, head_width=0.04, length_includes_head=True)
        plt.title(f"Latent Trajectories (Corrected Samples) — {method_name}")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pca_trajectory.png"))
        plt.close()

    return out_dir


def _plot_cross_method_comparison(all_metrics, out_dir):
    
    os.makedirs(out_dir, exist_ok=True)
    set_style()

    methods     = [m["method"]          for m in all_metrics]
    corr_rates  = [m["correction_rate"] for m in all_metrics]
    pre_f1s     = [m["pre_f1"]          for m in all_metrics]
    post_f1s    = [m["post_f1"]         for m in all_metrics]
    ppls        = [m["avg_ppl"]         for m in all_metrics]
    nll_deltas  = [m["avg_nll_delta"]   for m in all_metrics]
    x = np.arange(len(methods))

    # ---- Correction rate ----
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    plt.figure(figsize=(max(14, len(methods) * 1.0), 7))
    bars = plt.bar(x, corr_rates, color=colors)
    plt.xticks(x, methods, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Correction Rate (Fixed / Attempted)")
    plt.title("Hallucination Correction Rate per Method")
    plt.ylim(0, 1)
    for bar, v in zip(bars, corr_rates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.2f}", ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correction_rates.png"))
    plt.close()

    # ---- F1 pre vs post ----
    w = 0.35
    plt.figure(figsize=(max(14, len(methods) * 1.0), 7))
    plt.bar(x - w/2, pre_f1s,  w, label="Pre-correction F1",  color="#d62728", alpha=0.85)
    plt.bar(x + w/2, post_f1s, w, label="Post-correction F1", color="#2ca02c", alpha=0.85)
    plt.xticks(x, methods, rotation=45, ha="right", fontsize=8)
    plt.ylabel("F1 Score")
    plt.title("F1 Score: Pre vs Post Correction per Method")
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "f1_comparison.png"))
    plt.close()

    # ---- Perplexity ----
    plt.figure(figsize=(max(14, len(methods) * 1.0), 7))
    bars = plt.bar(x, ppls, color=colors)
    plt.xticks(x, methods, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Average Perplexity (lower = more fluent)")
    plt.title("Generation Quality per Method (Perplexity)")
    for bar, v in zip(bars, ppls):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{v:.1f}", ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "perplexity_comparison.png"))
    plt.close()

    # ---- NLL delta ----
    plt.figure(figsize=(max(14, len(methods) * 1.0), 7))
    bar_colors = ["#2ca02c" if d > 0 else "#d62728" for d in nll_deltas]
    bars = plt.bar(x, nll_deltas, color=bar_colors)
    plt.xticks(x, methods, rotation=45, ha="right", fontsize=8)
    plt.axhline(0, color="black", lw=0.8)
    plt.ylabel("Avg NLL Delta (pre - post, higher = more corrected)")
    plt.title("NLL Reduction per Method (Corrected Samples)")
    for bar, v in zip(bars, nll_deltas):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.02 if v >= 0 else -0.08),
                 f"{v:.3f}", ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nll_delta_comparison.png"))
    plt.close()

    # ---- Heatmap ----
    metric_keys = ["correction_rate", "post_f1", "avg_nll_delta", "avg_ppl",
                   "avg_token_f1", "avg_exact_match", "informativeness"]
    if any("avg_rouge_l" in m for m in all_metrics):
        metric_keys.append("avg_rouge_l")
    if any("avg_bertscore_f1" in m for m in all_metrics):
        metric_keys.append("avg_bertscore_f1")

    hm_labels = {
        "correction_rate": "Corr. Rate", "post_f1": "Post-F1",
        "avg_nll_delta": "NLL Delta", "avg_ppl": "1/PPL",
        "avg_token_f1": "Token-F1", "avg_exact_match": "Exact Match",
        "informativeness": "Informative", "avg_rouge_l": "ROUGE-L",
        "avg_bertscore_f1": "BERTScore-F1",
    }
    hm_data = []
    for key in metric_keys:
        row = []
        for m in all_metrics:
            val = m.get(key, 0.0)
            if val is None:
                val = 0.0
            # Invert PPL so higher = better in the heatmap
            if key == "avg_ppl":
                val = 1.0 / (val + 1e-6)
            row.append(val)
        hm_data.append(row)

    hm_data = np.array(hm_data)
    row_max = hm_data.max(axis=1, keepdims=True) + 1e-9
    hm_norm = hm_data / row_max

    fig, ax = plt.subplots(figsize=(max(14, len(methods) * 0.8), max(5, len(metric_keys) * 0.8)))
    sns.heatmap(hm_norm, annot=np.round(hm_data, 3), fmt=".3f",
                xticklabels=methods,
                yticklabels=[hm_labels.get(k, k) for k in metric_keys],
                cmap="YlGn", ax=ax, cbar_kws={"label": "Normalised score"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.title("Method Comparison Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "method_heatmap.png"))
    plt.close()


def _plot_gating_advantage(all_metrics, base_methods, out_dir):
    """
    Grouped bar chart: for each base method, compare Gated vs Blind on key metrics.
    This is the core novelty visualization.
    """
    os.makedirs(out_dir, exist_ok=True)
    set_style()

    # Build lookup
    metrics_by_name = {m["method"]: m for m in all_metrics}
    methods_with_both = []
    for bm in base_methods:
        gk = f"{bm} [Gated]"
        bk = f"{bm} [Blind]"
        if gk in metrics_by_name and bk in metrics_by_name:
            methods_with_both.append(bm)

    if not methods_with_both:
        return

    compare_metrics = [
        ("correction_rate", "Correction Rate", True),
        ("post_f1",         "Post-F1 Score",   True),
        ("avg_nll_delta",   "NLL Delta",       True),
        ("avg_token_f1",    "Token-F1",        True),
    ]

    n_metrics = len(compare_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 7))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(len(methods_with_both))
    w = 0.35

    for ax, (key, label, higher_better) in zip(axes, compare_metrics):
        gated_vals = [metrics_by_name[f"{bm} [Gated]"].get(key, 0.0) for bm in methods_with_both]
        blind_vals = [metrics_by_name[f"{bm} [Blind]"].get(key, 0.0) for bm in methods_with_both]

        ax.bar(x - w/2, gated_vals, w, label="Gated (Ours)", color="#2ca02c", alpha=0.85)
        ax.bar(x + w/2, blind_vals, w, label="Blind (SOTA)",  color="#d62728", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(methods_with_both, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.suptitle("Gating Advantage: PCNet-Gated vs Blind Correction", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gating_advantage.png"))
    plt.close()

    # Preservation vs Corruption plot
    fig, ax = plt.subplots(figsize=(max(10, len(methods_with_both) * 1.2), 6))
    pres_vals = [metrics_by_name[f"{bm} [Gated]"].get("preservation_rate", 0.0) for bm in methods_with_both]
    corr_vals = [metrics_by_name[f"{bm} [Blind]"].get("corruption_rate", 0.0) for bm in methods_with_both]
    ax.bar(x - w/2, pres_vals, w, label="Preservation Rate (Gated)", color="#2ca02c", alpha=0.85)
    ax.bar(x + w/2, corr_vals, w, label="Corruption Rate (Blind)",  color="#d62728", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(methods_with_both, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Rate")
    ax.set_title("Preservation (Gated) vs Corruption (Blind) on Correct Samples")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "preservation_vs_corruption.png"))
    plt.close()


def _save_method_logs(method_name, samples, metrics_dict, out_dir):
    """Save instance_logs.json, metrics.json, and a markdown report."""
    os.makedirs(out_dir, exist_ok=True)

    # instance_logs.json — one entry per sample
    export = []
    for s in samples:
        export.append({
            "prompt":               s["prompt"],
            "original_text":        s.get("text_pre", ""),
            "corrected_text":       s.get("text_post", s.get("text_pre", "")),
            "reference_answer":     s.get("ref_answer", ""),
            "ground_truth_label":   s["gt_label"],
            "pre_nll":              s["nll_pre"],
            "post_nll":             s["nll_post"],
            "flagged":              s.get("flagged", False),
            "corrected":            s.get("corrected", False),
            "mode":                 s.get("mode", ""),
            "post_correction_status": s.get("status", "Not_Corrected"),
        })
    with open(os.path.join(out_dir, "instance_logs.json"), "w") as f:
        json.dump(export, f, indent=2)

    # metrics.json
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # report.md
    md = f"# Correction Report: {method_name}\n\n"
    md += f"## Summary\n"
    md += f"- **Model:** `{metrics_dict.get('llm', '')}`\n"
    md += f"- **Dataset:** `{metrics_dict.get('dataset', '')}`\n"
    md += f"- **Method:** `{method_name}`\n"
    md += f"- **Mode:** `{metrics_dict.get('mode', '')}`\n\n"

    md += f"## Detection (Pre-Correction)\n"
    md += f"| Metric | Value |\n|--------|-------|\n"
    md += f"| TP | {metrics_dict['TP']} |\n"
    md += f"| FP | {metrics_dict['FP']} |\n"
    md += f"| TN | {metrics_dict['TN']} |\n"
    md += f"| FN | {metrics_dict['FN']} |\n"
    md += f"| Pre-F1 | {metrics_dict['pre_f1']:.4f} |\n"
    md += f"| Pre-Precision | {metrics_dict['pre_precision']:.4f} |\n"
    md += f"| Pre-Recall | {metrics_dict['pre_recall']:.4f} |\n"
    md += f"| AUROC | {metrics_dict.get('auroc', 0):.4f} |\n"
    md += f"| AUPRC | {metrics_dict.get('auprc', 0):.4f} |\n\n"

    md += f"## Correction (Post-Correction)\n"
    md += f"| Metric | Value |\n|--------|-------|\n"
    md += f"| TP Fixed | {metrics_dict['TP_Fixed']} |\n"
    md += f"| TP Failed | {metrics_dict['TP_Failed']} |\n"
    md += f"| FP Recovered | {metrics_dict['FP_Recovered']} |\n"
    md += f"| FP Corrupted | {metrics_dict['FP_Corrupted']} |\n"
    md += f"| Correction Rate | {metrics_dict['correction_rate']:.4f} |\n"
    md += f"| Post-F1 | {metrics_dict['post_f1']:.4f} |\n"
    md += f"| Post-AUROC | {metrics_dict.get('post_auroc', 0):.4f} |\n"
    md += f"| Avg Pre NLL | {metrics_dict['avg_pre_nll']:.4f} |\n"
    md += f"| Avg Post NLL | {metrics_dict['avg_post_nll']:.4f} |\n"
    md += f"| Avg NLL Delta | {metrics_dict['avg_nll_delta']:.4f} |\n\n"

    md += f"## Generation Quality\n"
    md += f"| Metric | Value |\n|--------|-------|\n"
    md += f"| Avg Perplexity | {metrics_dict['avg_ppl']:.2f} |\n"
    md += f"| Median Perplexity | {metrics_dict.get('median_ppl', 0):.2f} |\n"
    md += f"| Informativeness | {metrics_dict.get('informativeness', 0):.4f} |\n\n"

    md += f"## Text Quality (vs Reference)\n"
    md += f"| Metric | Value |\n|--------|-------|\n"
    md += f"| Token-F1 | {metrics_dict.get('avg_token_f1', 0):.4f} |\n"
    md += f"| Token-Precision | {metrics_dict.get('avg_token_precision', 0):.4f} |\n"
    md += f"| Token-Recall | {metrics_dict.get('avg_token_recall', 0):.4f} |\n"
    md += f"| Exact Match | {metrics_dict.get('avg_exact_match', 0):.4f} |\n"
    if "avg_rouge_l" in metrics_dict:
        md += f"| ROUGE-L | {metrics_dict['avg_rouge_l']:.4f} |\n"
    if "avg_bertscore_f1" in metrics_dict:
        md += f"| BERTScore-F1 | {metrics_dict['avg_bertscore_f1']:.4f} |\n"
    md += "\n"

    # Gating-specific metrics
    mode = metrics_dict.get("mode", "")
    if mode == "Gated":
        md += f"## Gating Metrics\n"
        md += f"| Metric | Value |\n|--------|-------|\n"
        md += f"| Preservation Rate | {metrics_dict.get('preservation_rate', 0):.4f} |\n"
        md += f"| Gating Efficiency | {metrics_dict.get('gating_efficiency', 0):.4f} |\n"
    elif mode == "Blind":
        md += f"## Blind-Mode Metrics\n"
        md += f"| Metric | Value |\n|--------|-------|\n"
        md += f"| Corruption Rate | {metrics_dict.get('corruption_rate', 0):.4f} |\n"
        md += f"| Unnecessary Corrections | {metrics_dict.get('unnecessary_corrections', 0)} |\n"

    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(md)


# ============================================================
# CORRECTION DISPATCH
# ============================================================

def _apply_correction(method_name, guardrail, inputs_tok, args_dict, pad_id, norm_name, truthx_model):
    
    if method_name == "PCNet_Static (Energy)":
        return correction_pcnet.generate_pcnet_static(guardrail, inputs_tok, args_dict, mode="energy")

    elif method_name == "PCNet_Static (Z-Score)":
        return correction_pcnet.generate_pcnet_static(guardrail, inputs_tok, args_dict, mode="z_score")
    
    elif method_name == "PC_Constrained_Decoding":
        # Pass alpha=1.0; you can expose this in your config later
        # top_k_cands=3 is a good balance between speed and quality
        return correction_pcnet.generate_pc_adaptive_constrained_decoding(
            guardrail, inputs_tok, args_dict, temperature=1.0, top_k_cands=3
        )

    elif method_name == "ITI":
        head_specs = args_dict.get("iti_head_specs", None)
        iti_dir_h  = args_dict.get("iti_dir_h", None)
        # Per-head ITI uses alpha=15 (paper); fallback layer-level uses 0.15.
        alpha_iti = args_dict.get("iti_alpha", 15.0 if head_specs else 0.15)
        out, _ = correction_baselines.generate_iti(
            guardrail, inputs_tok, args_dict["max_new_tokens"], pad_id,
            norm_name=norm_name, iti_dir_h=iti_dir_h,
            alpha=alpha_iti, head_specs=head_specs,
            stop_ids=args_dict.get("stop_ids"),
        )
        return out

    elif method_name == "DoLa":
        alpha_dola = args_dict.get("dola_alpha", 0.1)
        return correction_baselines.generate_dola(
            guardrail, inputs_tok, args_dict["max_new_tokens"], pad_id,
            alpha=alpha_dola,
            bucket=args_dict.get("dola_bucket", "high"),
            stop_ids=args_dict.get("stop_ids"),
        )

    elif method_name == "TruthX":
        out, _ = correction_baselines.generate_truthx(
            guardrail, inputs_tok, args_dict["max_new_tokens"], pad_id,
            truthx_model, stop_ids=args_dict.get("stop_ids"),
        )
        return out

    elif method_name == "ICD":
        # Induce-then-Contrast Decoding (Zhang et al., NAACL Findings 2024).
        # F = (1+alpha)·log p_E − alpha·log p_A; α-mass plausibility on expert.
        return correction_baselines.generate_icd(
            guardrail, inputs_tok, args_dict["max_new_tokens"], pad_id,
            alpha=args_dict.get("icd_alpha", 1.0),
            plausibility_alpha=args_dict.get("icd_plausibility_alpha", 0.0),
            stop_ids=args_dict.get("stop_ids"),
        )

    elif method_name == "SADI":
        sadi_dir_h = args_dict.get("iti_dir_h", None)
        out, _ = correction_baselines.generate_sadi(
            guardrail, inputs_tok, args_dict["max_new_tokens"], pad_id,
            norm_name, sadi_dir_h,
            delta=args_dict.get("sadi_delta", 0.30),
            top_k_ratio=args_dict.get("sadi_top_k_ratio", 0.10),
            stop_ids=args_dict.get("stop_ids"),
        )
        return out

    elif method_name == "ACT":
        act_dirs = args_dict.get("act_dirs_h", None)
        out, _ = correction_baselines.generate_act(
            guardrail, inputs_tok, args_dict["max_new_tokens"], pad_id,
            norm_name, act_dirs,
            alpha=args_dict.get("act_alpha", 0.20),
            beta=args_dict.get("act_beta", 0.0),
            stop_ids=args_dict.get("stop_ids"),
        )
        return out

    elif method_name == "AdaSteer":
        # Two-direction linear λ; (w, b) fitted by logistic regression in Phase 2.
        out, _ = correction_baselines.generate_adasteer(
            guardrail, inputs_tok, args_dict["max_new_tokens"], pad_id,
            norm_name,
            adasteer_kwargs=args_dict.get("adasteer_kwargs"),
            alpha=args_dict.get("adasteer_alpha", 1.0),
            stop_ids=args_dict.get("stop_ids"),
        )
        return out

    elif method_name == "HalluCana":
        out, _ = correction_baselines.generate_hallucana(
            guardrail, inputs_tok, args_dict["max_new_tokens"], pad_id,
            nll_threshold=args_dict.get("optimal_threshold", 0.50),
            lookahead_k=args_dict.get("hallucana_lookahead", 8),
            top_alt=args_dict.get("hallucana_top_alt", 5),
            entropy_threshold=args_dict.get("hallucana_entropy_threshold", 2.0),
            score_alpha=args_dict.get("hallucana_score_alpha", 1.0),
            stop_ids=args_dict.get("stop_ids"),
        )
        return out

    # Fallback: vanilla
    return correction_baselines.generate_vanilla(
        guardrail, inputs_tok, args_dict["max_new_tokens"], pad_id,
        stop_ids=args_dict.get("stop_ids"),
    )

# ============================================================
# PIPELINE
# ============================================================

def _set_global_determinism(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    try:
        import transformers
        transformers.set_seed(seed)
    except Exception:
        pass


class ExperimentalPipeline:
    def __init__(self, config_file: str, seed: int = 42):
        self.config_path   = config_file
        self.seed          = seed
        self.global_indices = {}
        self.iti_dir_h     = None
        self.load_config()
        self.summary_rows  = []   # one row per (llm, dataset, method) for final CSV

    def load_config(self):
        with open(self.config_path) as f:
            self.config = json.load(f)
        self.device            = self.config.get("device", "cuda")
        self.algorithm         = self.config.get("algorithm", "PCNet_Guardrail")
        self.llm_models        = self.config.get("llm_models", [])
        self.datasets          = self.config.get("datasets", [])
        self.max_samples       = self.config.get("max_samples", 100)
        self.max_new_tokens    = self.config.get("max_new_tokens", 150)
        self.optimal_threshold = self.config.get("optimal_threshold", 0.50)
        self.compute_bertscore = self.config.get("compute_bertscore", False)
        _set_global_determinism(self.seed)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _pad_id(self, guardrail):
        pid = guardrail.tokenizer.eos_token_id
        if isinstance(pid, list): pid = pid[0]
        return pid if pid is not None else (guardrail.tokenizer.pad_token_id or 0)

    def _get_stop_ids(self, guardrail):
        """
        Collect every valid stop / end-of-turn token id across Llama-3, Qwen,
        Mistral, DeepSeek, etc. Returns a deduped Python list of ints.
        """
        tok = guardrail.tokenizer
        stop_ids = set()

        # 1. Standard EOS (can be int or list[int] in HF tokenizers)
        eos = tok.eos_token_id
        if eos is not None:
            if isinstance(eos, (list, tuple)):
                stop_ids.update(int(x) for x in eos if x is not None)
            else:
                stop_ids.add(int(eos))

        # 2. PAD, if distinct
        if tok.pad_token_id is not None:
            stop_ids.add(int(tok.pad_token_id))

        # 3. Architecture-specific chat / turn terminators.
        #    get_vocab() is expensive; call once.
        vocab = tok.get_vocab()
        for s in ("<|eot_id|>", "<|im_end|>", "<|end_of_text|>", "</s>"):
            if s in vocab:
                stop_ids.add(int(vocab[s]))

        return list(stop_ids) if stop_ids else [0]

    def _final_norm_name(self, llm):
        for candidate in ["model.norm", "model.final_layernorm",
                          "transformer.ln_f", "gpt_neox.final_layer_norm"]:
            obj = llm
            try:
                for part in candidate.split("."): obj = getattr(obj, part)
                return candidate
            except AttributeError:
                continue
        return ""

    def calculate_perplexity(self, llm, tokenizer, text):
        if not text or not text.strip():
            return float("inf")
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        if inputs["input_ids"].numel() < 2:
            return float("inf")
        with torch.no_grad():
            out = llm(**inputs, labels=inputs["input_ids"])
        return torch.exp(out.loss).item() if out.loss else 10.0

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------

    def execute(self):
        print(f"Correction Pipeline (Dual-Mode) | config: {self.config_path}")
        print(f"Threshold: {self.optimal_threshold} | Samples: {self.max_samples}\n")

        base_methods = [
            # # "PCNet_Static (Energy)",
            # # "PCNet_Static (Z-Score)",
            # # "PCNet_TAC (Clamping)",
            # # "PC_Ridge_Imputation",       # Idea B: min-norm ridge inverse (Wiener target)
            # # "PC_Conditional_Imputation", # Idea A: MAP conditional via full PC joint
            # # "PC-DoLa",
            # # "MIC (tau=20)",
            # # "MIC (tau=10)",
            "ITI",
            "DoLa",
            # # "TruthX",
            # # "PC-Guided_Contrastive",
            # # "Soft_Wiener_Imputation",
            "PC_Constrained_Decoding",
            # # "PC_Adaptive_Logit_Interpolation",
            # "PC_Vocabulary_Banishment",
            # # "PC_Score_Vocab_Banishment",
            # # "PC_h-DoLa",
            # # "PC_Anti-Score_Contrast",
            # "PC_Langevin_Tweedie",
            # # "PC_Onset_Correction",
            "ICD",
            "SADI",
            # # "ACT",
            "AdaSteer",
            "HalluCana",
            # # "PC_Langevin_Banishment",
            # "PC_DPA",
        ]

        # Build full method list: Vanilla + each base method x {Gated, Blind}
        ablation_entries = [("Vanilla_Baseline", None)]
        for m in base_methods:
            ablation_entries.append((m, "Gated"))
            ablation_entries.append((m, "Blind"))

        for llm_name in self.llm_models:
            print(f"\n{'='*60}\nLLM: {llm_name}\n{'='*60}")
            guardrail = LLM_PC_Guardrail(
                llm_model_name=llm_name, pc_in_channels=128, device=self.device
            )
            llm       = guardrail.llm
            tokenizer = guardrail.tokenizer

            for ds_info in self.datasets:
                ds_name   = ds_info["name"]
                ds_subset = ds_info.get("subset", None)
                ds_split  = ds_info.get("split", "data")
                ds_display = f"{ds_name} ({ds_subset})" if ds_subset else ds_name

                importlib.reload(hf_dataloader)
                dataset = hf_dataloader.UnifiedDataLoader.load_test_data(
                    ds_name, subset=ds_subset, split=ds_split,
                    max_samples=self.max_samples, seed=self.seed,
                    return_refs=True,
                )
                if not dataset:
                    print(f"  Empty dataset for {ds_display}, skipping.")
                    continue

                safe_llm = llm_name.replace("/", "_")
                safe_ds  = ds_name.replace("/", "_")
                if ds_subset:
                    safe_ds = f"{safe_ds}_{ds_subset}"
                base_log_dir = os.path.join("2_last_correction_pipeline_logs", str(self.seed), safe_llm, safe_ds)
                os.makedirs(base_log_dir, exist_ok=True)

                # Load trained PCNet 
                pc_path = os.path.join(
                    "checkpoints", self.algorithm, str(self.seed),
                    llm_name.replace("/", "_"), ds_name.replace("/", "_"),
                    "pcnet_best.pth"
                )
                if os.path.exists(pc_path):
                    ckpt = torch.load(pc_path, map_location=self.device, weights_only=False)
                    guardrail.pc_prior  = ckpt["pc_prior"].to(self.device)
                    guardrail.projector = ckpt["projector"].to(self.device)
                    print(f"  Loaded PCNet from {pc_path}")
                else:
                    print(f"  No PCNet checkpoint at {pc_path}. Using untrained model.")

                guardrail.eval()
                D          = 128
                leaf_means = _get_leaf_means(guardrail, D)

                # ------------------------------------------------
                # PHASE 1: Baseline generation
                # ------------------------------------------------
                print(f"\n  PHASE 1: Baseline generation ({ds_display})")
                phase1 = []
                pre_embs_all = []   # (idx, np.array) for PCA

                for idx, sample_tuple in enumerate(tqdm(dataset, desc="  Phase 1")):
                    text, gt_label, ref_answer = sample_tuple
                    prompt = text.split("\nAnswer:")[0] + "\nAnswer:"
                    text_pre = text.split("\nAnswer:")[1].strip() if "\nAnswer:" in text else text.strip()
                    inputs_tok = tokenizer(prompt, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        z_pre = guardrail._get_llm_embeddings([prompt + " " + text_pre])
                        
                    # CRITICAL FIX: Use Tractable Dropout Inference (TDI) to match experiment.py
                    log_prob = guardrail(prompt + " " + text_pre, mpe_training=False, n_samples=20)
                    nll = -log_prob.mean().item()

                    ppl = self.calculate_perplexity(llm, tokenizer, text_pre)
                    pre_embs_all.append((idx, z_pre.detach().cpu().numpy()))

                    phase1.append({
                        "idx":          idx,
                        "prompt":       prompt,
                        "inputs_tok":   inputs_tok,
                        "text_pre":     text_pre,
                        "gt_label":     gt_label,
                        "ref_answer":   ref_answer,
                        "is_anomalous": nll >= self.optimal_threshold,
                        "nll_pre":      nll,
                        "ppl_pre":      ppl,
                        "z_pre":        z_pre.detach(),
                    })

                # --- Calibrate threshold on observed NLL distribution ---
                all_nlls   = np.array([r["nll_pre"] for r in phase1])
                gt_labels  = np.array([r["gt_label"] for r in phase1])
                if gt_labels.sum() > 0 and (gt_labels == 0).sum() > 0:
                    prec_arr, rec_arr, thr_arr = precision_recall_curve(gt_labels, all_nlls)
                    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
                    calibrated_threshold = float(thr_arr[np.argmax(f1_arr)])
                    print(f"  Calibrated threshold: {calibrated_threshold:.4f} "
                          f"(was {self.optimal_threshold:.4f})")
                    self.optimal_threshold = calibrated_threshold
                    for r in phase1:
                        r["is_anomalous"] = r["nll_pre"] >= self.optimal_threshold
                else:
                    print(f"  WARNING: Cannot calibrate — using config threshold {self.optimal_threshold}")

                anomalous = [r for r in phase1 if r["is_anomalous"]]

                # Pre-correction confusion matrix (shared across all methods)
                TP, FP, TN, FN = _confusion_counts(phase1, self.optimal_threshold)
                pre_f1, pre_prec, pre_rec = _f1(TP, FP, FN)
                print(f"  Pre-correction — TP:{TP} FP:{FP} TN:{TN} FN:{FN} | F1:{pre_f1:.3f}")

                # Pre-correction AUROC/AUPRC (shared)
                pre_auroc, pre_auprc = _compute_auroc_auprc(phase1)
                print(f"  Pre-correction — AUROC:{pre_auroc:.3f} AUPRC:{pre_auprc:.3f}")

                # ------------------------------------------------
                # PHASE 2: Global gradient + ITI direction
                # ------------------------------------------------
                print(f"\n  PHASE 2: Diagnosing circuits ({len(anomalous)} flagged samples)")
                self.global_indices = {"energy": [], "z_score": []}
                self.iti_dir_h        = None
                self.iti_head_specs   = None
                self.adasteer_kwargs  = None

                if anomalous:
                    global_grads = torch.zeros(llm.config.hidden_size, device=self.device)
                    for r in tqdm(anomalous, desc="  Gradients"):
                        out_states = llm(**r["inputs_tok"], output_hidden_states=True)
                        h_last = out_states.hidden_states[-1][:, -1:, :].detach().requires_grad_(True)
                        with torch.enable_grad():
                            proj = guardrail.projector(h_last.squeeze(1).float())
                            nll_val = -guardrail.pc_prior(proj.unsqueeze(-1).unsqueeze(-1)).mean()
                        grad_h = torch.autograd.grad(nll_val, h_last)[0]
                        global_grads += grad_h.abs().squeeze(0).squeeze(0)

                    # Energy threshold
                    sorted_g, sorted_idx = torch.sort(global_grads, descending=True)
                    cum_e    = torch.cumsum(sorted_g, dim=0) / (sorted_g.sum() + 1e-9)
                    e_thresh = self.config.get("gradient_energy_threshold", 0.10)
                    cutoff   = torch.where(cum_e >= e_thresh)[0]
                    self.global_indices["energy"] = sorted_idx[:cutoff[0].item() + 1].tolist() if len(cutoff) else []

                    # Z-score threshold
                    z_thresh = self.config.get("z_score_threshold", 3.0)
                    thr_val  = global_grads.mean() + z_thresh * global_grads.std()
                    self.global_indices["z_score"] = torch.where(global_grads > thr_val)[0].tolist()

                    print(f"  Energy dims: {len(self.global_indices['energy'])}  "
                          f"Z-score dims: {len(self.global_indices['z_score'])}")

                    # ITI direction (legacy single-direction fallback)
                    zf = [r["z_pre"] for r in phase1 if r["gt_label"] == 0]
                    zh = [r["z_pre"] for r in phase1 if r["gt_label"] == 1]
                    if zf and zh:
                        iti_dir_z  = F.normalize(
                            torch.cat(zf).mean(0, keepdim=True) -
                            torch.cat(zh).mean(0, keepdim=True), dim=-1
                        )
                        self.iti_dir_h = _compute_delta_h(guardrail, iti_dir_z)
                        print(f"  ITI legacy direction: {len(zf)} factual / {len(zh)} hallu samples.")

                    # Paper-faithful ITI: per-attention-head probe + top-K=48.
                    samples_iter = [
                        (r["inputs_tok"], int(r["gt_label"])) for r in phase1
                    ]
                    try:
                        self.iti_head_specs = correction_baselines.calibrate_per_head_iti(
                            llm, samples_iter, n_top_heads=48,
                        )
                        n_sel = sum(len(v) for v in self.iti_head_specs.values())
                        print(f"  ITI per-head: selected {n_sel} heads "
                              f"across {len(self.iti_head_specs)} layers.")
                    except Exception as exc:
                        print(f"  [WARN] ITI per-head calibration failed "
                              f"({type(exc).__name__}: {exc}). "
                              f"Falling back to layer-level ITI.")
                        self.iti_head_specs = None

                    # AdaSteer: two-direction linear λ + logistic-regression fit.
                    try:
                        norm_name_calib = correction_pcnet._final_norm_name(llm)
                        self.adasteer_kwargs = correction_baselines.calibrate_adasteer(
                            llm, samples_iter, layer_name=norm_name_calib,
                        )
                        if self.adasteer_kwargs is not None:
                            print(f"  AdaSteer: fitted "
                                  f"w_r={self.adasteer_kwargs['w_r']:.3f} "
                                  f"b_r={self.adasteer_kwargs['b_r']:.3f} "
                                  f"w_c={self.adasteer_kwargs['w_c']:.3f} "
                                  f"b_c={self.adasteer_kwargs['b_c']:.3f}")
                    except Exception as exc:
                        print(f"  [WARN] AdaSteer calibration failed "
                              f"({type(exc).__name__}: {exc}).")
                        self.adasteer_kwargs = None

                # ------------------------------------------------
                # PHASE 3: Dual-mode corrections
                # ------------------------------------------------
                print(f"\n  PHASE 3: Applying {len(base_methods)} methods x 2 modes + Vanilla...")

                # Initialize storage for all method+mode combinations
                display_names = []
                for base, mode in ablation_entries:
                    dn = base if mode is None else f"{base} [{mode}]"
                    display_names.append(dn)

                method_samples = {dn: [] for dn in display_names}
                method_post_embs = {dn: [] for dn in display_names if dn != "Vanilla_Baseline"}

                # Skip methods whose metrics.json already exists in the log dir.
                def _safe_method_name(display):
                    return display.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("/", "_").replace("[", "").replace("]", "")

                done_methods = set()
                for _base, _mode in ablation_entries:
                    _display = _base if _mode is None else f"{_base} [{_mode}]"
                    _mdir = os.path.join(base_log_dir, _safe_method_name(_display))
                    if os.path.exists(os.path.join(_mdir, "metrics.json")):
                        done_methods.add(_display)
                        print(f"  [SKIP] {_display} — metrics.json found, will load from cache.")

                # Multi-LLM stop criterion: collect every EOS / end-of-turn
                # token id for this tokenizer (Llama-3 <|eot_id|>, Qwen
                # <|im_end|>, etc.) and hand it to every correction method.
                pad_id   = correction_pcnet._pad_id(guardrail)
                stop_ids = self._get_stop_ids(guardrail)

                args_dict = {
                    "global_indices": self.global_indices,
                    "max_new_tokens": self.max_new_tokens,
                    "mic_alpha": self.config.get("mic_alpha", 0.15),
                    "leaf_means": leaf_means,
                    "iti_alpha": self.config.get("iti_alpha", 15.0),
                    "iti_dir_h": self.iti_dir_h,
                    "iti_head_specs": self.iti_head_specs,
                    "dola_alpha": self.config.get("dola_alpha", 0.1),
                    "icd_alpha": self.config.get("icd_alpha", 1.0),
                    "icd_plausibility_alpha": self.config.get("icd_plausibility_alpha", 0.0),
                    "adasteer_kwargs": self.adasteer_kwargs,
                    "adasteer_alpha": self.config.get("adasteer_alpha", 1.0),
                    "stop_ids": stop_ids,   # list[int], used by custom loops
                }
                norm_name = correction_pcnet._final_norm_name(guardrail.llm)
                truthx_model = None  # TruthX fallback

                for r in tqdm(phase1, desc="  Corrections"):
                    inputs_tok = r["inputs_tok"]
                    prompt     = r["prompt"]
                    idx        = r["idx"]
                    flagged    = r["is_anomalous"]
                    ref_answer = r["ref_answer"]

                    def _decode(out):
                        txt = tokenizer.decode(
                            out[0][inputs_tok["input_ids"].shape[1]:],
                            skip_special_tokens=True
                        )
                        # Safety net: if a stop token was missed and the LLM
                        # rambled into a second QA turn, cut at the first
                        # newline. Models that legitimately emit multi-line
                        # answers for this benchmark do not exist.
                        if "\n" in txt:
                            txt = txt.split("\n", 1)[0]
                        return txt.strip()

                    def _nll_and_emb(txt):
                        text_full = prompt + " " + txt
                        with torch.no_grad():
                            emb = guardrail._get_llm_embeddings([text_full])
                        log_prob = guardrail(text_full, mpe_training=False, n_samples=20)
                        nll = -log_prob.mean().item()
                        return nll, emb.detach().cpu().numpy()

                    def _status(gt, nll_post, was_corrected):
                        if not was_corrected:
                            return "Not_Corrected"
                        if gt == 1:
                            return "TP_Fixed" if nll_post < self.optimal_threshold else "TP_Failed"
                        else:
                            return "FP_Recovered" if nll_post < self.optimal_threshold else "FP_Corrupted"

                    def _make_sample(txt_post, nll_post, was_corrected, mode_str, ppl):
                        return {
                            "idx":        idx,
                            "prompt":     prompt,
                            "text_pre":   r["text_pre"],
                            "text_post":  txt_post,
                            "gt_label":   r["gt_label"],
                            "ref_answer": ref_answer,
                            "nll_pre":    r["nll_pre"],
                            "nll_post":   nll_post,
                            "flagged":    flagged,
                            "corrected":  was_corrected,
                            "mode":       mode_str,
                            "status":     _status(r["gt_label"], nll_post, was_corrected),
                            "ppl":        ppl,
                        }

                    # --- Vanilla Baseline (no correction, shared reference) ---
                    # ppl_pre already computed in Phase 1 — no extra LLM call.
                    if "Vanilla_Baseline" not in done_methods:
                        method_samples["Vanilla_Baseline"].append(
                            _make_sample(r["text_pre"], r["nll_pre"], False, None, r["ppl_pre"])
                        )

                    # --- Per base method: Gated + Blind ---
                    for base_method in base_methods:
                        gated_key = f"{base_method} [Gated]"
                        blind_key = f"{base_method} [Blind]"

                        gated_done = gated_key in done_methods
                        blind_done = blind_key in done_methods

                        if gated_done and blind_done:
                            continue

                        if flagged:
                            # Sample is flagged: run correction ONCE, share for both modes.
                            # PPL of the corrected text is also computed once and reused.
                            out = _apply_correction(
                                base_method, guardrail, inputs_tok, args_dict,
                                pad_id, norm_name, truthx_model
                            )
                            txt = _decode(out)
                            nll_p, emb_p = _nll_and_emb(txt)
                            ppl_cor = self.calculate_perplexity(llm, tokenizer, txt)

                            if not gated_done:
                                method_samples[gated_key].append(
                                    _make_sample(txt, nll_p, True, "Gated", ppl_cor)
                                )
                                method_post_embs[gated_key].append((idx, emb_p))
                            if not blind_done:
                                method_samples[blind_key].append(
                                    _make_sample(txt, nll_p, True, "Blind", ppl_cor)
                                )
                                method_post_embs[blind_key].append((idx, emb_p))
                        else:
                            # Gated: not flagged → copy vanilla; reuse ppl_pre.
                            if not gated_done:
                                method_samples[gated_key].append(
                                    _make_sample(r["text_pre"], r["nll_pre"], False, "Gated", r["ppl_pre"])
                                )

                            # Blind: not flagged → still run correction; compute PPL once.
                            if not blind_done:
                                out = _apply_correction(
                                    base_method, guardrail, inputs_tok, args_dict,
                                    pad_id, norm_name, truthx_model
                                )
                                txt = _decode(out)
                                nll_p, emb_p = _nll_and_emb(txt)
                                method_samples[blind_key].append(
                                    _make_sample(txt, nll_p, True, "Blind",
                                                 self.calculate_perplexity(llm, tokenizer, txt))
                                )
                                method_post_embs[blind_key].append((idx, emb_p))

                # ------------------------------------------------
                # PHASE 4: Metrics + plots + logs per method
                # ------------------------------------------------
                print(f"\n  PHASE 4: Computing metrics and generating plots...")
                all_method_metrics = []

                for base, mode in ablation_entries:
                    display = base if mode is None else f"{base} [{mode}]"

                    if display in done_methods:
                        method_dir = os.path.join(base_log_dir, _safe_method_name(display))
                        with open(os.path.join(method_dir, "metrics.json")) as _f:
                            metrics = json.load(_f)
                        all_method_metrics.append(metrics)
                        self.summary_rows.append(metrics)
                        print(f"  {display:40s} | [loaded from cache]")
                        continue

                    samples = method_samples[display]

                    tf, tf_fail, fp_rec, fp_cor = _post_confusion(samples, self.optimal_threshold)
                    attempted  = tf + tf_fail + fp_rec + fp_cor
                    fixed      = tf + fp_rec
                    corr_rate  = fixed / attempted if attempted > 0 else 0.0

                    # Post-F1
                    post_f1, post_prec, post_rec = _f1(tf_fail, fp_cor, FN)

                    # NLL stats (on corrected samples only)
                    corrected_s = [s for s in samples if s["corrected"]]
                    pre_nlls    = [s["nll_pre"]  for s in corrected_s] or [0]
                    post_nlls   = [s["nll_post"] for s in corrected_s] or [0]
                    all_ppls    = [s["ppl"] for s in samples]

                    # Post-AUROC
                    post_auroc = _compute_post_auroc(samples)

                    # Text quality metrics
                    text_quality = _compute_text_quality_metrics(samples, self.compute_bertscore)

                    # Gating-specific metrics
                    gating = _compute_gating_metrics(samples, mode, self.optimal_threshold) if mode else {}

                    metrics = {
                        "llm":             llm_name,
                        "dataset":         ds_display,
                        "method":          display,
                        "mode":            mode or "None",
                        # Pre-correction detection (shared)
                        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
                        "pre_f1":          round(pre_f1,  4),
                        "pre_precision":   round(pre_prec, 4),
                        "pre_recall":      round(pre_rec,  4),
                        "auroc":           round(pre_auroc, 4),
                        "auprc":           round(pre_auprc, 4),
                        # Post-correction effectiveness
                        "TP_Fixed":        tf,
                        "TP_Failed":       tf_fail,
                        "FP_Recovered":    fp_rec,
                        "FP_Corrupted":    fp_cor,
                        "attempted":       attempted,
                        "correction_rate": round(corr_rate, 4),
                        "post_f1":         round(post_f1,  4),
                        "post_precision":  round(post_prec, 4),
                        "post_recall":     round(post_rec,  4),
                        "post_auroc":      round(post_auroc, 4),
                        # NLL
                        "avg_pre_nll":     round(float(np.mean(pre_nlls)),  4),
                        "avg_post_nll":    round(float(np.mean(post_nlls)), 4),
                        "avg_nll_delta":   round(float(np.mean(pre_nlls)) - float(np.mean(post_nlls)), 4),
                        # Generation quality
                        "avg_ppl":         round(float(np.mean(all_ppls)), 4),
                        "median_ppl":      round(float(np.median(all_ppls)), 4),
                        "std_ppl":         round(float(np.std(all_ppls)), 4),
                        # Text quality
                        **text_quality,
                        # Gating
                        **gating,
                    }
                    all_method_metrics.append(metrics)

                    # Save logs
                    safe_m = display.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("/", "_").replace("[", "").replace("]", "")
                    method_dir = os.path.join(base_log_dir, safe_m)
                    _save_method_logs(display, samples, metrics, method_dir)

                    # Per-method plots
                    _plot_per_method(
                        display, samples, base_log_dir, self.optimal_threshold,
                        pre_embs_all,
                        method_post_embs.get(display, [])
                    )

                    print(f"  {display:40s} | CorRate: {corr_rate:.3f} | Post-F1: {post_f1:.3f} "
                          f"| NLL d: {metrics['avg_nll_delta']:+.3f} | PPL: {metrics['avg_ppl']:.1f}"
                          f"| TokF1: {metrics.get('avg_token_f1', 0):.3f}")

                    # Accumulate for global CSV
                    self.summary_rows.append(metrics)

                # ------------------------------------------------
                # PHASE 5: Cross-method comparison plots
                # ------------------------------------------------
                print(f"\n  PHASE 5: Cross-method comparison plots...")
                comp_dir = os.path.join(base_log_dir, "comparison")
                _plot_cross_method_comparison(all_method_metrics, comp_dir)
                _plot_gating_advantage(all_method_metrics, base_methods, comp_dir)

                # Save per-dataset comparison CSV
                pd.DataFrame(all_method_metrics).to_csv(
                    os.path.join(comp_dir, "method_summary.csv"), index=False
                )
                print(f"  Comparison saved to {comp_dir}/")

                flush_memory()

            print(f"\n  Cleaning up {llm_name}...")
            del guardrail
            flush_memory()

        self._export_global_report()

    # ------------------------------------------------------------------
    # Global summary report
    # ------------------------------------------------------------------

    def _export_global_report(self):
        if not self.summary_rows:
            return
        df = pd.DataFrame(self.summary_rows)
        os.makedirs("2_last_correction_pipeline_logs", exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"2_last_correction_pipeline_logs/global_summary_{ts}.csv"
        df.to_csv(out, index=False)

        print(f"\n{'='*60}")
        print("GLOBAL SUMMARY (averaged across all LLMs and datasets)")
        print("="*60)

        agg_cols = ["correction_rate", "pre_f1", "post_f1", "avg_nll_delta",
                    "avg_ppl", "avg_token_f1", "avg_exact_match", "informativeness"]
        # Only include columns that exist
        agg_cols = [c for c in agg_cols if c in df.columns]

        pivot = (
            df.groupby("method")[agg_cols]
            .mean()
            .sort_values("correction_rate", ascending=False)
            .reset_index()
        )
        print(pivot.to_markdown(index=False))
        print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dual-Mode Correction Pipeline: PCNet-Gated vs Blind for all methods"
    )
    parser.add_argument("config", help="Path to evaluation JSON config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    ExperimentalPipeline(args.config, args.seed).execute()
