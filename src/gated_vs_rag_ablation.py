
import os
import re
import json
import math
import time
import argparse
import importlib
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

import hf_dataloader
import correction_baselines
import correction_pcnet
from correction_pcnet import _compute_delta_h, _get_leaf_means
from guardrail import LLM_PC_Guardrail
from correction_pipeline import (
    _apply_correction,
    _set_global_determinism,
    _token_f1,
    _exact_match,
    flush_memory,
)


# -------------------------------------------------------------------- BM25
_TOK_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text):
    return _TOK_RE.findall(text.lower())


class BM25:
    """Pure-Python BM25 (Okapi). Adequate for ~10^5 short docs."""
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b  = b
        self.docs = docs_tokens
        self.N    = len(docs_tokens)
        self.doc_lens = np.array([len(d) for d in docs_tokens], dtype=np.float64)
        self.avgdl    = float(self.doc_lens.mean()) if self.N else 0.0
        self.df = Counter()
        self.tf = []
        for d in docs_tokens:
            counts = Counter(d)
            self.tf.append(counts)
            self.df.update(counts.keys())
        self.idf = {
            t: math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            for t, df in self.df.items()
        }

    def score_topk(self, query_tokens, k=3):
        scores = np.zeros(self.N, dtype=np.float64)
        denom_factor = self.k1 * (1.0 - self.b + self.b * self.doc_lens / max(self.avgdl, 1e-6))
        for q in query_tokens:
            idf = self.idf.get(q)
            if idf is None:
                continue
            for did, count in self._postings(q):
                num = count * (self.k1 + 1.0)
                den = count + denom_factor[did]
                scores[did] += idf * num / den
        if not np.any(scores):
            return []
        top_idx = np.argpartition(-scores, min(k, self.N - 1))[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    def _postings(self, term):
        if not hasattr(self, "_post_cache"):
            self._post_cache = {}
        cached = self._post_cache.get(term)
        if cached is not None:
            return cached
        out = [(did, tfs[term]) for did, tfs in enumerate(self.tf) if term in tfs]
        self._post_cache[term] = out
        return out


# -------------------------------------------------------------------- helpers
def _pad_id(guardrail):
    pid = guardrail.tokenizer.eos_token_id
    if isinstance(pid, list):
        pid = pid[0]
    return pid if pid is not None else (guardrail.tokenizer.pad_token_id or 0)


def _stop_ids(guardrail):
    tok = guardrail.tokenizer
    s = set()
    eos = tok.eos_token_id
    if eos is not None:
        if isinstance(eos, (list, tuple)):
            s.update(int(x) for x in eos if x is not None)
        else:
            s.add(int(eos))
    if tok.pad_token_id is not None:
        s.add(int(tok.pad_token_id))
    vocab = tok.get_vocab()
    for tag in ("<|eot_id|>", "<|im_end|>", "<|end_of_text|>", "</s>"):
        if tag in vocab:
            s.add(int(vocab[tag]))
    return list(s) if s else [0]


def _final_norm_name(llm):
    for cand in ("model.norm", "model.final_layernorm",
                 "transformer.ln_f", "gpt_neox.final_layer_norm"):
        obj = llm
        try:
            for part in cand.split("."):
                obj = getattr(obj, part)
            return cand
        except AttributeError:
            continue
    return ""


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _decode_new_tokens(tokenizer, out_ids, prompt_len):
    txt = tokenizer.decode(out_ids[0][prompt_len:], skip_special_tokens=True)
    if "\n" in txt:
        txt = txt.split("\n", 1)[0]
    return txt.strip()


def _extract_question(prompt):
    m = re.search(r"Question:\s*(.*?)\s*\nAnswer:", prompt, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return prompt.strip()


def _build_rag_prompt(retrieved_docs, question):
    ctx = "\n".join(f"- {d}" for d in retrieved_docs)
    return (
        "Use the following context to answer the question. "
        "If the context is not helpful, answer from your own knowledge.\n"
        f"Context:\n{ctx}\n"
        f"Question: {question}\nAnswer:"
    )


# -------------------------------------------------------------------- corpus
def _build_corpus(seed, max_corpus):
    factuals, _ = hf_dataloader.UnifiedDataLoader.load_train_data(
        "trivia_qa", subset="rc.nocontext", split="train",
        max_samples=max_corpus, seed=seed,
    )
    docs_text = []
    for entry in factuals:
        m = re.match(r"Question:\s*(.*?)\s*\nAnswer:\s*(.*)", entry, flags=re.DOTALL)
        if m:
            q, a = m.group(1).strip(), m.group(2).strip()
            docs_text.append(f"{q} {a}")
        else:
            docs_text.append(entry)
    docs_tokens = [_tokenize(d) for d in docs_text]
    return docs_text, docs_tokens


# -------------------------------------------------------------------- TruthfulQA loaders
def _load_truthfulqa_mc(n, seed):
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    ds = ds.shuffle(seed=seed)
    rows = []
    for row in ds:
        q = row.get("question", "")
        mc1 = row.get("mc1_targets", {})
        mc2 = row.get("mc2_targets", {})
        if not (mc1.get("choices") and mc1.get("labels")):
            continue
        correct1 = next(
            (i for i, lbl in enumerate(mc1["labels"]) if str(lbl).strip() in ("1", "1.0")),
            None,
        )
        if correct1 is None:
            continue
        rows.append({
            "question":    q,
            "mc1_choices": mc1["choices"],
            "mc1_correct": correct1,
            "mc2_choices": mc2.get("choices", []),
            "mc2_labels":  mc2.get("labels", []),
        })
        if len(rows) >= n:
            break
    return rows


def _load_truthfulqa_gen(n, seed):
    importlib.reload(hf_dataloader)
    data = hf_dataloader.UnifiedDataLoader.load_test_data(
        "truthful_qa", subset="generation", split="validation",
        max_samples=n, seed=seed, return_refs=True,
    )
    return data


# -------------------------------------------------------------------- MC scoring
def _choice_logprob(llm, tokenizer, question_prompt, choice, device):
    
    full_text = question_prompt + " " + choice
    q_ids   = tokenizer(question_prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full_text,      return_tensors="pt").input_ids.to(device)
    n_choice = full_ids.shape[1] - q_ids.shape[1]
    if n_choice <= 0:
        return -1e9
    with torch.no_grad():
        logits = llm(input_ids=full_ids).logits          # (1, L, V)
    log_probs = F.log_softmax(logits[0, :-1, :], dim=-1) # (L-1, V)
    targets   = full_ids[0, 1:]                          # (L-1,)
    choice_start = q_ids.shape[1] - 1                    # 0-indexed position
    total_lp = log_probs[choice_start:, :].gather(
        1, targets[choice_start:].unsqueeze(1)
    ).sum().item()
    return total_lp / max(n_choice, 1)


def _pcnet_choice_score(guardrail, question_prompt, choice, device, alpha=1.0):
    """log P(choice|q) + alpha * PC-prior score for the combined text."""
    lp = _choice_logprob(guardrail.llm, guardrail.tokenizer,
                         question_prompt, choice, device)
    full_text = question_prompt + " " + choice
    with torch.no_grad():
        log_prob = guardrail(full_text, mpe_training=False, n_samples=20)
    pc_score = log_prob.mean().item()
    return lp + alpha * pc_score


def _mc_nll(guardrail, question_prompt, choice):
    """Negative PC-prior log-prob for the (Q, choice) pair — used for gating."""
    full_text = question_prompt + " " + choice
    with torch.no_grad():
        lp = guardrail(full_text, mpe_training=False, n_samples=20)
    return -lp.mean().item()


def _mc2_score(log_probs, labels):
    """
    MC2: normalized probability mass on correct choices.
    score = sum(softmax(log_probs)[i] for i where label==1)
    """
    arr = np.array(log_probs, dtype=np.float64)
    arr -= arr.max()
    exp_arr = np.exp(arr)
    softmax = exp_arr / (exp_arr.sum() + 1e-9)
    correct_mask = np.array([1 if int(l) == 1 else 0 for l in labels], dtype=np.float64)
    return float((softmax * correct_mask).sum())


def _mc3_score(log_probs, labels):
    """
    MC3: fraction of (correct, incorrect) pairs where P(correct) > P(incorrect).
    """
    correct_lps   = [lp for lp, l in zip(log_probs, labels) if int(l) == 1]
    incorrect_lps = [lp for lp, l in zip(log_probs, labels) if int(l) == 0]
    if not correct_lps or not incorrect_lps:
        return 0.0
    n_correct = sum(
        1 for c in correct_lps for inc in incorrect_lps if c > inc
    )
    return n_correct / (len(correct_lps) * len(incorrect_lps))


def _score_mc_all(score_fn, mc1_choices, mc2_choices, mc2_labels):
    """
    Apply score_fn(choice) to both MC1 and MC2 choice sets.
    Returns (mc1_pred_idx, mc2_score, mc3_score).
    """
    mc1_lps = [score_fn(c) for c in mc1_choices]
    mc1_pred = int(np.argmax(mc1_lps))
    if mc2_choices:
        mc2_lps = [score_fn(c) for c in mc2_choices]
        mc2 = _mc2_score(mc2_lps, mc2_labels)
        mc3 = _mc3_score(mc2_lps, mc2_labels)
    else:
        mc2, mc3 = 0.0, 0.0
    return mc1_pred, mc2, mc3


# -------------------------------------------------------------------- MC arms
def _run_mc_vanilla(row, guardrail, device):
    """
    Score all MC1+MC2 choices, return (pred_idx, dt, nll, mc2, mc3).
    """
    q = row["question"]
    prompt = f"Question: {q}\nAnswer:"
    _sync(); t0 = time.perf_counter()
    score_fn = lambda c: _choice_logprob(guardrail.llm, guardrail.tokenizer, prompt, c, device)
    pred_idx, mc2, mc3 = _score_mc_all(
        score_fn, row["mc1_choices"], row["mc2_choices"], row["mc2_labels"]
    )
    _sync(); dt = time.perf_counter() - t0
    nll = _mc_nll(guardrail, prompt, row["mc1_choices"][pred_idx])
    return pred_idx, dt, nll, mc2, mc3


def _run_mc_pcnet_gated(row, guardrail, device, alpha, is_flagged):
    """
    If flagged: rerank MC1+MC2 choices by (log P + alpha * PC-score).
    Returns (pred_idx, mc2, mc3, dt) or None if not flagged.
    """
    if not is_flagged:
        return None
    q = row["question"]
    prompt = f"Question: {q}\nAnswer:"
    _sync(); t0 = time.perf_counter()
    score_fn = lambda c: _pcnet_choice_score(guardrail, prompt, c, device, alpha=alpha)
    pred_idx, mc2, mc3 = _score_mc_all(
        score_fn, row["mc1_choices"], row["mc2_choices"], row["mc2_labels"]
    )
    _sync(); dt = time.perf_counter() - t0
    return pred_idx, mc2, mc3, dt


def _run_mc_rag(row, guardrail, device, bm25, docs_text, top_k):
    """
    Always retrieve; score MC1+MC2 choices with RAG-augmented prompt.
    Returns (pred_idx, mc2, mc3, dt, n_retrieved).
    """
    q = row["question"]
    hits = bm25.score_topk(_tokenize(q), k=top_k)
    retrieved = [docs_text[i] for i, _ in hits] if hits else []
    rag_prompt = _build_rag_prompt(retrieved, q) if retrieved else f"Question: {q}\nAnswer:"
    _sync(); t0 = time.perf_counter()
    score_fn = lambda c: _choice_logprob(guardrail.llm, guardrail.tokenizer, rag_prompt, c, device)
    pred_idx, mc2, mc3 = _score_mc_all(
        score_fn, row["mc1_choices"], row["mc2_choices"], row["mc2_labels"]
    )
    _sync(); dt = time.perf_counter() - t0
    return pred_idx, mc2, mc3, dt, len(retrieved)


# -------------------------------------------------------------------- generation phase1
def _phase1_gen(guardrail, dataset, threshold, stop_ids_list, max_new_tokens, device):
    
    llm  = guardrail.llm
    tok  = guardrail.tokenizer
    pad_id = _pad_id(guardrail)
    out  = []
    for idx, (text, gt, ref) in enumerate(tqdm(dataset, desc="  Phase 1")):
        prompt   = text.split("\nAnswer:")[0] + "\nAnswer:"
        text_pre = text.split("\nAnswer:")[1].strip() if "\nAnswer:" in text else ""
        inputs_tok = tok(prompt, return_tensors="pt").to(llm.device)

        # NLL on dataset-stored answer (for threshold calibration only)
        with torch.no_grad():
            z_pre = guardrail._get_llm_embeddings([prompt + " " + text_pre])
        log_prob_pre = guardrail(prompt + " " + text_pre, mpe_training=False, n_samples=20)
        nll_pre = -log_prob_pre.mean().item()

        # Generate vanilla answer with proper stop tokens
        vanilla_out = correction_baselines.generate_vanilla(
            guardrail, inputs_tok, max_new_tokens, pad_id, stop_ids=stop_ids_list
        )
        txt_vanilla = _decode_new_tokens(tok, vanilla_out, inputs_tok["input_ids"].shape[1])

        # NLL on vanilla generation → runtime gating signal
        if txt_vanilla:
            log_prob_van = guardrail(
                prompt + " " + txt_vanilla, mpe_training=False, n_samples=20
            )
            nll_vanilla = -log_prob_van.mean().item()
        else:
            nll_vanilla = nll_pre  # fallback if generation empty

        out.append({
            "idx":         idx,
            "prompt":      prompt,
            "inputs_tok":  inputs_tok,
            "text_pre":    text_pre,
            "gt_label":    gt,
            "ref":         ref or "",
            "nll_pre":     nll_pre,       # used for calibration
            "nll_vanilla": nll_vanilla,   # used for gating
            "z_pre":       z_pre.detach(),
            "txt_vanilla": txt_vanilla,
            "vanilla_out": vanilla_out,   # raw token ids
            "is_anomalous": nll_vanilla >= threshold,
            "question":    _extract_question(prompt),
        })
    return out


def _calibrate_threshold(phase1_records, nll_key, default):
    nlls = np.array([r[nll_key] for r in phase1_records])
    gts  = np.array([r["gt_label"]  for r in phase1_records])
    if gts.sum() == 0 or (gts == 0).sum() == 0:
        return default
    p, r, t = precision_recall_curve(gts, nlls)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-8)
    return float(t[np.argmax(f1)])


def _phase2(guardrail, phase1, anomalous, config):
    llm = guardrail.llm
    global_indices = {"energy": [], "z_score": []}
    if anomalous:
        grads = torch.zeros(llm.config.hidden_size, device=llm.device)
        for r in tqdm(anomalous, desc="  Gradients"):
            o = llm(**r["inputs_tok"], output_hidden_states=True)
            h = o.hidden_states[-1][:, -1:, :].detach().requires_grad_(True)
            with torch.enable_grad():
                proj = guardrail.projector(h.squeeze(1).float())
                v = -guardrail.pc_prior(proj.unsqueeze(-1).unsqueeze(-1)).mean()
            g = torch.autograd.grad(v, h)[0]
            grads += g.abs().squeeze(0).squeeze(0)
        sg, si = torch.sort(grads, descending=True)
        ce = torch.cumsum(sg, dim=0) / (sg.sum() + 1e-9)
        cu = torch.where(ce >= config.get("gradient_energy_threshold", 0.10))[0]
        global_indices["energy"] = si[:cu[0].item() + 1].tolist() if len(cu) else []
        zt = config.get("z_score_threshold", 3.0)
        thr = grads.mean() + zt * grads.std()
        global_indices["z_score"] = torch.where(grads > thr)[0].tolist()

    iti_dir_h = None
    zf = [r["z_pre"] for r in phase1 if r["gt_label"] == 0]
    zh = [r["z_pre"] for r in phase1 if r["gt_label"] == 1]
    if zf and zh:
        iti_dir_z = F.normalize(
            torch.cat(zf).mean(0, keepdim=True) -
            torch.cat(zh).mean(0, keepdim=True), dim=-1
        )
        iti_dir_h = _compute_delta_h(guardrail, iti_dir_z)

    samples_iter = [(r["inputs_tok"], int(r["gt_label"])) for r in phase1]
    iti_head_specs = None
    try:
        iti_head_specs = correction_baselines.calibrate_per_head_iti(
            llm, samples_iter, n_top_heads=48,
        )
    except Exception as exc:
        print(f"  [WARN] ITI per-head calibration failed: {type(exc).__name__}: {exc}")

    adasteer_kwargs = None
    try:
        adasteer_kwargs = correction_baselines.calibrate_adasteer(
            llm, samples_iter, layer_name=_final_norm_name(llm),
        )
    except Exception as exc:
        print(f"  [WARN] AdaSteer calibration failed: {type(exc).__name__}: {exc}")

    return {
        "global_indices":  global_indices,
        "iti_dir_h":       iti_dir_h,
        "iti_head_specs":  iti_head_specs,
        "adasteer_kwargs": adasteer_kwargs,
    }


# -------------------------------------------------------------------- generation arms
def _run_correction(method, r, guardrail, args_dict, pad_id, norm_name):
    _sync(); t0 = time.perf_counter()
    out = _apply_correction(method, guardrail, r["inputs_tok"], args_dict,
                            pad_id, norm_name, None)
    _sync(); dt = time.perf_counter() - t0
    txt = _decode_new_tokens(guardrail.tokenizer, out, r["inputs_tok"]["input_ids"].shape[1])
    return txt, dt


def _run_rag_gen(r, guardrail, max_new_tokens, pad_id, stop_ids_list, bm25, docs_text, top_k):
    tok = guardrail.tokenizer
    llm = guardrail.llm
    _sync(); t_ret_0 = time.perf_counter()
    hits = bm25.score_topk(_tokenize(r["question"]), k=top_k)
    _sync(); t_ret = time.perf_counter() - t_ret_0
    retrieved = [docs_text[i] for i, _ in hits] if hits else []
    new_prompt = _build_rag_prompt(retrieved, r["question"])
    rag_inputs = tok(new_prompt, return_tensors="pt").to(llm.device)
    _sync(); t_gen_0 = time.perf_counter()
    # Use generate_vanilla so stop_ids are applied correctly
    out = correction_baselines.generate_vanilla(
        guardrail, rag_inputs, max_new_tokens, pad_id, stop_ids=stop_ids_list
    )
    _sync(); t_gen = time.perf_counter() - t_gen_0
    txt = _decode_new_tokens(tok, out, rag_inputs["input_ids"].shape[1])
    return txt, (t_ret + t_gen), t_ret


# -------------------------------------------------------------------- summary helpers
def _summarise_gen_arm(name, results, n_total):
    if not results:
        return {"arm": name, "n": 0, "token_f1": 0.0, "exact_match": 0.0,
                "lat_mean_s": 0.0, "lat_median_s": 0.0, "lat_p95_s": 0.0,
                "lat_total_s": 0.0, "retrieve_total_s": 0.0,
                "intervened": 0, "intervene_rate": 0.0}
    f1s  = [r["f1"]  for r in results]
    ems  = [r["em"]  for r in results]
    lats = [r["latency_s"]        for r in results]
    rets = [r.get("retrieve_s", 0.0) for r in results]
    n_int = sum(1 for r in results if r["intervened"])
    return {
        "arm":              name,
        "n":                len(results),
        "token_f1":         float(np.mean(f1s)),
        "exact_match":      float(np.mean(ems)),
        "lat_mean_s":       float(np.mean(lats)),
        "lat_median_s":     float(np.median(lats)),
        "lat_p95_s":        float(np.percentile(lats, 95)),
        "lat_total_s":      float(np.sum(lats)),
        "retrieve_total_s": float(np.sum(rets)),
        "intervened":       n_int,
        "intervene_rate":   n_int / n_total if n_total else 0.0,
    }


def _summarise_mc_arm(name, results, n_total):
    if not results:
        return {"arm": name, "n": 0, "mc1_acc": 0.0, "mc2_score": 0.0,
                "mc3_score": 0.0, "lat_mean_s": 0.0, "lat_p95_s": 0.0,
                "lat_total_s": 0.0, "intervened": 0, "intervene_rate": 0.0}
    accs  = [r["mc1_correct"] for r in results]
    mc2s  = [r["mc2_score"]   for r in results]
    mc3s  = [r["mc3_score"]   for r in results]
    lats  = [r["latency_s"]   for r in results]
    n_int = sum(1 for r in results if r.get("intervened", False))
    return {
        "arm":            name,
        "n":              len(results),
        "mc1_acc":        float(np.mean(accs)),
        "mc2_score":      float(np.mean(mc2s)),
        "mc3_score":      float(np.mean(mc3s)),
        "lat_mean_s":     float(np.mean(lats)),
        "lat_p95_s":      float(np.percentile(lats, 95)),
        "lat_total_s":    float(np.sum(lats)),
        "intervened":     n_int,
        "intervene_rate": n_int / n_total if n_total else 0.0,
    }


def _markdown_gen(rows, meta):
    lines = [
        f"# Gated vs RAG (Generation) — {meta['llm']} / {meta['dataset']}",
        "",
        f"- Seed: {meta['seed']}  ·  N: {meta['n_test']}  ·  Corpus: {meta['corpus_n']}",
        f"- Correction: **{meta['correction_method']}**  ·  BM25 top-k: {meta['top_k']}",
        f"- NLL threshold (calibrated): {meta['threshold']:.4f}  "
        f"·  Anomalous: {meta['anom_frac']:.3f}",
        f"- Generated: {meta['timestamp']}",
        "",
    ]
    headers = ["Arm", "Token-F1", "EM", "Lat mean (s)", "P95 (s)",
               "Ret tot (s)", "Intervened", "vs Vanilla"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    base_lat = next((r["lat_mean_s"] for r in rows if r["arm"] == "Vanilla"), None)
    for r in rows:
        ratio = (r["lat_mean_s"] / base_lat) if base_lat else float("nan")
        lines.append(
            f"| {r['arm']} | {r.get('token_f1',0):.3f} | {r.get('exact_match',0):.3f} "
            f"| {r['lat_mean_s']:.3f} | {r['lat_p95_s']:.3f} "
            f"| {r['retrieve_total_s']:.2f} | {r['intervened']}/{r['n']} | {ratio:.2f}x |"
        )
    return "\n".join(lines) + "\n"


def _markdown_mc(rows, meta):
    lines = [
        f"# Gated vs RAG (MC) — {meta['llm']} / TruthfulQA",
        "",
        f"- Seed: {meta['seed']}  ·  N: {meta['n_mc']}",
        f"- NLL threshold: {meta['threshold']:.4f}  "
        f"·  Flagged: {meta['anom_frac']:.3f}",
        f"- Generated: {meta['timestamp']}",
        "",
    ]
    headers = ["Arm", "MC1-Acc", "MC2-Score", "MC3-Score",
               "Lat mean (s)", "P95 (s)", "Lat tot (s)", "Intervened", "vs Vanilla"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    base_lat = next((r["lat_mean_s"] for r in rows if r["arm"] == "Vanilla"), None)
    for r in rows:
        ratio = (r["lat_mean_s"] / base_lat) if base_lat else float("nan")
        lines.append(
            f"| {r['arm']} | {r.get('mc1_acc',0):.3f} "
            f"| {r.get('mc2_score',0):.3f} | {r.get('mc3_score',0):.3f} "
            f"| {r['lat_mean_s']:.3f} | {r['lat_p95_s']:.3f} "
            f"| {r['lat_total_s']:.2f} | {r['intervened']}/{r['n']} | {ratio:.2f}x |"
        )
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------- main
def _load_pcnet(guardrail, algorithm, seed, llm_name, ds_name, device):
    safe_llm = llm_name.replace("/", "_")
    safe_ds  = ds_name.replace("/", "_")
    for s in [str(seed), "42"]:
        pc_path = os.path.join("checkpoints", algorithm, s, safe_llm, safe_ds, "pcnet_best.pth")
        if os.path.exists(pc_path):
            ckpt = torch.load(pc_path, map_location=device, weights_only=False)
            guardrail.pc_prior  = ckpt["pc_prior"].to(device)
            guardrail.projector = ckpt["projector"].to(device)
            print(f"  Loaded PCNet from {pc_path}")
            return
    print(f"  No PCNet checkpoint found for {llm_name}/{ds_name} — using untrained model.")


def _save(out_dir, payload, md_text):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(payload, f, indent=2,
                  default=lambda o: getattr(o, "tolist", lambda: str(o))())
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(md_text)
    print(f"  → {out_dir}/results.{{json,md}}")


def run_generation_track(
    guardrail, dataset_items, llm_name, ds_name,
    correction_method, config, args, bm25, docs_text,
):
    """Run the 4-arm generation ablation for one dataset."""
    device        = config.get("device", "cuda")
    algorithm     = config.get("algorithm", "PCNet_Guardrail")
    max_new_tokens = config.get("max_new_tokens", 150)
    optimal_threshold = config.get("optimal_threshold", 0.50)
    top_k         = args.top_k
    seed          = args.seed

    _load_pcnet(guardrail, algorithm, seed, llm_name, ds_name, device)
    guardrail.eval()
    leaf_means = _get_leaf_means(guardrail, 128)

    pad_id       = _pad_id(guardrail)
    stop_ids_list = _stop_ids(guardrail)
    norm_name    = correction_pcnet._final_norm_name(guardrail.llm)

    # Phase 1: generate vanilla + compute both NLLs
    phase1 = _phase1_gen(
        guardrail, dataset_items, optimal_threshold,
        stop_ids_list, max_new_tokens, device,
    )

    # Calibrate threshold on text_pre NLL vs gt_label
    threshold = _calibrate_threshold(phase1, "nll_pre", optimal_threshold)
    # Apply calibrated threshold to vanilla-generation NLL for gating
    for r in phase1:
        r["is_anomalous"] = r["nll_vanilla"] >= threshold
    anomalous = [r for r in phase1 if r["is_anomalous"]]
    anom_frac = len(anomalous) / max(len(phase1), 1)
    print(f"  Threshold {threshold:.4f}  · anomalous {len(anomalous)}/{len(phase1)}")

    extras = _phase2(guardrail, phase1, anomalous, config)
    args_dict = {
        "global_indices":         extras["global_indices"],
        "max_new_tokens":         max_new_tokens,
        "mic_alpha":              config.get("mic_alpha", 0.15),
        "leaf_means":             leaf_means,
        "iti_alpha":              config.get("iti_alpha", 15.0),
        "iti_dir_h":              extras["iti_dir_h"],
        "iti_head_specs":         extras["iti_head_specs"],
        "dola_alpha":             config.get("dola_alpha", 0.1),
        "icd_alpha":              config.get("icd_alpha", 1.0),
        "icd_plausibility_alpha": config.get("icd_plausibility_alpha", 0.0),
        "adasteer_kwargs":        extras["adasteer_kwargs"],
        "adasteer_alpha":         config.get("adasteer_alpha", 1.0),
        "stop_ids":               stop_ids_list,
        "optimal_threshold":      threshold,
    }

    arm_vanilla, arm_gated_corr, arm_blind_rag, arm_gated_rag = [], [], [], []

    for r in tqdm(phase1, desc=f"  Gen arms ({ds_name})"):
        ref = r["ref"]
        # Vanilla: already generated in phase1
        txt_v  = r["txt_vanilla"]
        lat_v  = 0.0  # latency already spent in phase1; report as 0 here
        f1_v,_,_ = _token_f1(txt_v, ref)
        arm_vanilla.append({
            "idx": r["idx"], "answer": txt_v, "ref": ref, "f1": f1_v,
            "em": _exact_match(txt_v, ref), "latency_s": lat_v,
            "retrieve_s": 0.0, "intervened": False, "flagged": r["is_anomalous"],
        })

        # Gated correction
        if r["is_anomalous"]:
            txt_gc, lat_gc = _run_correction(
                correction_method, r, guardrail, args_dict, pad_id, norm_name
            )
            intervened_gc = True
        else:
            txt_gc, lat_gc = txt_v, 0.0
            intervened_gc = False
        f1_gc,_,_ = _token_f1(txt_gc, ref)
        arm_gated_corr.append({
            "idx": r["idx"], "answer": txt_gc, "ref": ref, "f1": f1_gc,
            "em": _exact_match(txt_gc, ref), "latency_s": lat_gc,
            "retrieve_s": 0.0, "intervened": intervened_gc, "flagged": r["is_anomalous"],
        })

        # Blind RAG
        txt_br, lat_br, t_ret_br = _run_rag_gen(
            r, guardrail, max_new_tokens, pad_id, stop_ids_list, bm25, docs_text, top_k
        )
        f1_br,_,_ = _token_f1(txt_br, ref)
        arm_blind_rag.append({
            "idx": r["idx"], "answer": txt_br, "ref": ref, "f1": f1_br,
            "em": _exact_match(txt_br, ref), "latency_s": lat_br,
            "retrieve_s": t_ret_br, "intervened": True, "flagged": r["is_anomalous"],
        })

        # Gated RAG (reuses blind-RAG result for flagged samples)
        if r["is_anomalous"]:
            txt_gr, lat_gr, t_ret_gr = txt_br, lat_br, t_ret_br
            intervened_gr = True
        else:
            txt_gr, lat_gr, t_ret_gr = txt_v, 0.0, 0.0
            intervened_gr = False
        f1_gr,_,_ = _token_f1(txt_gr, ref)
        arm_gated_rag.append({
            "idx": r["idx"], "answer": txt_gr, "ref": ref, "f1": f1_gr,
            "em": _exact_match(txt_gr, ref), "latency_s": lat_gr,
            "retrieve_s": t_ret_gr, "intervened": intervened_gr,
            "flagged": r["is_anomalous"],
        })

    n_total = len(phase1)
    rows = [
        _summarise_gen_arm("Vanilla",                          arm_vanilla,    n_total),
        _summarise_gen_arm(f"Gated_Correction ({correction_method})",
                           arm_gated_corr, n_total),
        _summarise_gen_arm("Blind_RAG",                       arm_blind_rag,  n_total),
        _summarise_gen_arm("Gated_RAG",                       arm_gated_rag,  n_total),
    ]

    safe_ds = ds_name.replace("/", "_").replace(".", "_")
    out_dir = os.path.join(args.out_root, str(seed),
                           llm_name.replace("/", "_"), safe_ds)
    meta = {
        "llm": llm_name, "dataset": ds_name, "seed": seed,
        "n_test": n_total, "corpus_n": len(docs_text),
        "correction_method": correction_method, "top_k": top_k,
        "max_new_tokens": max_new_tokens, "threshold": threshold,
        "anom_frac": anom_frac, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload = {
        "meta": meta, "summary": rows,
        "vanilla": arm_vanilla, "gated_correction": arm_gated_corr,
        "blind_rag": arm_blind_rag, "gated_rag": arm_gated_rag,
    }
    _save(out_dir, payload, _markdown_gen(rows, meta))


def run_mc_track(
    guardrail, mc_rows, llm_name, config, args, bm25, docs_text,
):
    """Run the 4-arm MC ablation for TruthfulQA."""
    device    = config.get("device", "cuda")
    algorithm = config.get("algorithm", "PCNet_Guardrail")
    threshold_cfg = config.get("optimal_threshold", 0.50)
    pc_alpha  = config.get("mc_pcnet_alpha", 1.0)
    top_k     = args.top_k
    seed      = args.seed

    _load_pcnet(guardrail, algorithm, seed, llm_name, "truthful_qa", device)
    guardrail.eval()

    arm_vanilla, arm_gated_corr, arm_blind_rag, arm_gated_rag = [], [], [], []
    all_nlls = []

    print(f"  MC Vanilla scoring ({len(mc_rows)} rows)...")
    for row in tqdm(mc_rows, desc="  MC Vanilla"):
        pred_idx, dt, nll, mc2, mc3 = _run_mc_vanilla(row, guardrail, device)
        all_nlls.append(nll)
        arm_vanilla.append({
            "question": row["question"],
            "pred_idx": pred_idx, "correct_idx": row["mc1_correct"],
            "mc1_correct": int(pred_idx == row["mc1_correct"]),
            "mc2_score": mc2, "mc3_score": mc3,
            "latency_s": dt, "intervened": False, "nll": nll,
        })

    # Calibrate threshold on MC NLLs — no gt_label here, so use median
    nll_arr = np.array(all_nlls)
    threshold = float(np.median(nll_arr))
    print(f"  MC NLL median threshold: {threshold:.4f}")
    flagged_idx = set(i for i, nll in enumerate(all_nlls) if nll >= threshold)
    anom_frac = len(flagged_idx) / max(len(mc_rows), 1)
    print(f"  Flagged: {len(flagged_idx)}/{len(mc_rows)}")

    print("  MC Gated_Correction + RAG arms...")
    for i, row in enumerate(tqdm(mc_rows, desc="  MC Arms")):
        is_flagged = i in flagged_idx
        van         = arm_vanilla[i]
        q           = row["question"]

        # Gated_Correction
        gc_result = _run_mc_pcnet_gated(row, guardrail, device, pc_alpha, is_flagged)
        if gc_result is None:
            gc_pred, gc_mc2, gc_mc3, dt_gc = van["pred_idx"], van["mc2_score"], van["mc3_score"], 0.0
        else:
            gc_pred, gc_mc2, gc_mc3, dt_gc = gc_result
        arm_gated_corr.append({
            "question": q, "pred_idx": gc_pred, "correct_idx": row["mc1_correct"],
            "mc1_correct": int(gc_pred == row["mc1_correct"]),
            "mc2_score": gc_mc2, "mc3_score": gc_mc3,
            "latency_s": dt_gc, "intervened": is_flagged,
        })

        # Blind RAG
        br_pred, br_mc2, br_mc3, dt_br, _ = _run_mc_rag(
            row, guardrail, device, bm25, docs_text, top_k
        )
        arm_blind_rag.append({
            "question": q, "pred_idx": br_pred, "correct_idx": row["mc1_correct"],
            "mc1_correct": int(br_pred == row["mc1_correct"]),
            "mc2_score": br_mc2, "mc3_score": br_mc3,
            "latency_s": dt_br, "intervened": True,
        })

        # Gated RAG
        if is_flagged:
            gr_pred, gr_mc2, gr_mc3, dt_gr = br_pred, br_mc2, br_mc3, dt_br
        else:
            gr_pred, gr_mc2, gr_mc3, dt_gr = van["pred_idx"], van["mc2_score"], van["mc3_score"], 0.0
        arm_gated_rag.append({
            "question": q, "pred_idx": gr_pred, "correct_idx": row["mc1_correct"],
            "mc1_correct": int(gr_pred == row["mc1_correct"]),
            "mc2_score": gr_mc2, "mc3_score": gr_mc3,
            "latency_s": dt_gr, "intervened": is_flagged,
        })

    n_total = len(mc_rows)
    rows = [
        _summarise_mc_arm("Vanilla",              arm_vanilla,    n_total),
        _summarise_mc_arm("Gated_PCNet",          arm_gated_corr, n_total),
        _summarise_mc_arm("Blind_RAG",            arm_blind_rag,  n_total),
        _summarise_mc_arm("Gated_RAG",            arm_gated_rag,  n_total),
    ]

    out_dir = os.path.join(args.out_root, str(seed),
                           llm_name.replace("/", "_"), "truthfulqa_mc")
    meta = {
        "llm": llm_name, "seed": seed, "n_mc": n_total,
        "threshold": threshold, "anom_frac": anom_frac,
        "pc_alpha": pc_alpha, "top_k": top_k,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload = {
        "meta": meta, "summary": rows,
        "vanilla": arm_vanilla, "gated_correction": arm_gated_corr,
        "blind_rag": arm_blind_rag, "gated_rag": arm_gated_rag,
    }
    _save(out_dir, payload, _markdown_mc(rows, meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="Eval JSON config (same format as correction_pipeline.py).")
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--n_test",         type=int,   default=200,
                    help="Samples per generation track.")
    ap.add_argument("--n_mc",           type=int,   default=200,
                    help="Rows for TruthfulQA MC track.")
    ap.add_argument("--corpus_size",    type=int,   default=10_000)
    ap.add_argument("--top_k",          type=int,   default=3)
    ap.add_argument("--correction_method", type=str, default="PC_Constrained_Decoding",
                    choices=["ITI", "DoLa", "AdaSteer", "SADI", "ICD",
                             "PC_Constrained_Decoding", "PC_Vocabulary_Banishment",
                             "PC_Langevin_Tweedie"])
    ap.add_argument("--tracks",         type=str, default="mc,truthfulqa_gen,triviaqa",
                    help="Comma-separated list of tracks to run: mc, truthfulqa_gen, triviaqa")
    ap.add_argument("--out_root",       type=str,   default="rag_ablation_logs")
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    _set_global_determinism(args.seed)

    tracks = [t.strip() for t in args.tracks.split(",")]

    print(f"Building BM25 corpus from TriviaQA train ({args.corpus_size} items)...")
    importlib.reload(hf_dataloader)
    docs_text, docs_tokens = _build_corpus(args.seed, args.corpus_size)
    bm25 = BM25(docs_tokens)
    print(f"  Corpus: {len(docs_text)} docs · avgdl={bm25.avgdl:.1f}")

    llm_models = config.get("llm_models", [])
    device     = config.get("device", "cuda")
    algorithm  = config.get("algorithm", "PCNet_Guardrail")

    for llm_name in llm_models:
        print(f"\n{'='*60}\nLLM: {llm_name}\n{'='*60}")

        guardrail = LLM_PC_Guardrail(
            llm_model_name=llm_name, pc_in_channels=128, device=device,
        )

        # ---- TruthfulQA MC (fast, no generation) ----
        if "mc" in tracks:
            print("\n--- TruthfulQA MC track ---")
            mc_rows = _load_truthfulqa_mc(args.n_mc, args.seed)
            print(f"  Loaded {len(mc_rows)} MC rows")
            run_mc_track(guardrail, mc_rows, llm_name, config, args, bm25, docs_text)

        # ---- TruthfulQA generation ----
        if "truthfulqa_gen" in tracks:
            print("\n--- TruthfulQA generation track ---")
            tqa_gen = _load_truthfulqa_gen(args.n_test, args.seed)
            if tqa_gen:
                tqa_gen = tqa_gen[:args.n_test]
                run_generation_track(
                    guardrail, tqa_gen, llm_name, "truthful_qa",
                    args.correction_method, config, args, bm25, docs_text,
                )
            else:
                print("  Empty TruthfulQA-gen set, skipping.")

        # ---- TriviaQA generation ----
        if "triviaqa" in tracks:
            print("\n--- TriviaQA generation track ---")
            importlib.reload(hf_dataloader)
            triviaqa = hf_dataloader.UnifiedDataLoader.load_test_data(
                "trivia_qa", subset="rc.nocontext", split="data",
                max_samples=args.n_test, seed=args.seed, return_refs=True,
            )
            if triviaqa:
                triviaqa = triviaqa[:args.n_test]
                run_generation_track(
                    guardrail, triviaqa, llm_name, "trivia_qa",
                    args.correction_method, config, args, bm25, docs_text,
                )
            else:
                print("  Empty TriviaQA set, skipping.")

        del guardrail
        flush_memory()


if __name__ == "__main__":
    main()
