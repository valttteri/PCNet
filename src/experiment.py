import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from baselines import (
    LatentMLPProbe,
    SemanticConsistencyBaseline,
    HaloScopeLatentBaseline,
    AutoFactNLIBaseline,
    ChainPollBaseline,
    SemanticEntropyProbe,
)

class HallucinationExperiment:
    """
    Benchmarking framework to compare the PCNet Latent Guardrail against 
    standard LLM token-level hallucination detection baselines.
    """
    def __init__(self, guardrail_model,mlp_probe=None, device="cuda"):
        self.device = device
        self.guardrail = guardrail_model
        self.guardrail.eval()
        
        # We reuse the LLM and Tokenizer already loaded in your Guardrail
        self.llm = self.guardrail.llm
        self.tokenizer = self.guardrail.tokenizer
        self.semantic_baseline = SemanticConsistencyBaseline(self.llm, self.tokenizer, device)
        
        # HaloScope k ~ 128 is closer to the paper's sweep on 4096-D Llama hidden states
        self.haloscope_baseline = HaloScopeLatentBaseline(n_components=128, device=device)
        self.autofact_baseline = AutoFactNLIBaseline(device=device)
        self.chainpoll_baseline = ChainPollBaseline(self.llm, self.tokenizer, device)
        self.sep_baseline = SemanticEntropyProbe(device=device)

        self.mlp_probe = mlp_probe
        if self.mlp_probe:
            self.mlp_probe.eval()

    def fit_haloscope(self, train_embeddings):
        self.haloscope_baseline.fit(train_embeddings)

    def fit_sep(self, train_embeddings, train_se_targets):
        self.sep_baseline.fit(train_embeddings, train_se_targets)

    def compute_semantic_entropy_for_fit(self, texts, num_samples=5, max_new_tokens=40, temperature=1.0):
        from tqdm import tqdm
        targets = []
        for t in tqdm(texts, desc="  [SEP] Computing SE targets"):
            try:
                se = self.semantic_baseline.compute_semantic_entropy(
                    t, num_samples=num_samples, max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                print(f"  [SEP] SE compute failed ({e}) — using NaN.")
                se = float("nan")
            targets.append(se)
        return np.array(targets, dtype=np.float64)

    def _parse_context_and_answer(self, text):
        context = ""
        answer = ""
        
        if "Context: " in text:
            context_part = text.split("Context: ")[1]
            if "\nTask: " in context_part:
                context = context_part.split("\nTask: ")[0]
            elif "\nQuestion: " in context_part:
                context = context_part.split("\nQuestion: ")[0]
            elif "\nAnswer: " in context_part:
                context = context_part.split("\nAnswer: ")[0]
                
        if "\nSummary: " in text:
            answer = text.split("\nSummary: ")[1]
        elif "\nAnswer: " in text:
            answer = text.split("\nAnswer: ")[1]
            
        if not context and "Question: " in text:
            context_part = text.split("Question: ")[1]
            if "\nAnswer: " in context_part:
                context = context_part.split("\nAnswer: ")[0]
                
        return context.strip(), answer.strip()

    @torch.no_grad()
    def baseline_token_likelihood_and_entropy(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        # Forward pass through the Causal LLM head
        outputs = self.llm(**inputs)
        logits = outputs.logits  # Shape: (Batch, Seq_Len, Vocab_Size)
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # 1. Average Token Log-Likelihood (Confidence)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        # CrossEntropyLoss expects (N, C) so we reshape
        token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        avg_nll = token_nll.mean().item()  # Lower NLL = Higher Confidence
        
        # 2. Predictive Entropy (Uncertainty)
        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1) # Sum across vocabulary
        avg_entropy = entropy.mean().item()
        
        # For anomaly detection, higher scores should indicate "Hallucination"
        return avg_nll, avg_entropy

    @torch.no_grad()
    @torch.no_grad()
    def proposed_pcnet_guardrail(self, text, n_samples=100):
        self.guardrail.eval() 
        
        # Trigger the MC Dropout loop by passing n_samples!
        log_prob = self.guardrail(text, mpe_training=False, n_samples=n_samples)
        
        # Safely extract Epistemic Uncertainty
        if hasattr(self.guardrail.pc_prior, '_last_var'):
            uncertainty = self.guardrail.pc_prior._last_var.mean().item()
        else:
            uncertainty = 0.0 
        
        # Score = Negative Log-Likelihood 
        nll_latent = -log_prob.mean().item()
        
        return nll_latent, uncertainty
    
    def run_benchmark(self, dataset, dataset_name="Unknown", llm_name="Unknown"):
        print("🚀 Starting Comprehensive Hallucination Benchmark...")
        
        # 1. Initialize all arrays correctly
        results = {
            "labels": [],
            "baseline_nll": [],
            "baseline_entropy": [],         # <-- Restored
            "sota_semantic_entropy": [], 
            "sota_mlp_latent": [],       
            "proposed_latent_nll": [],
            "proposed_epistemic_unc": [],    # <-- Added for TDI Uncertainty
            "sota_haloscope_latent": [],
            "sota_autofact_nli": [],
            "sota_chainpoll_judgement": [],
            "sota_sep": [],
        }

        for text, label in tqdm(dataset, desc="Evaluating"):
            # 1. Standard Token NLL & Token Entropy
            b_nll, b_entropy = self.baseline_token_likelihood_and_entropy(text) # <-- Capture both
            
            # 2. SOTA Black-Box: Semantic Entropy
            sem_entropy = self.semantic_baseline.compute_semantic_entropy(text, num_samples=10)
            
            # Extract plain text components
            context, answer = self._parse_context_and_answer(text)
            
            with torch.no_grad():
                # 3. SOTA White-Box: MLP Probe & HaloScope
                # Get 128-dim projected embeddings for the MLP baseline
                proj_emb = self.guardrail._get_llm_embeddings([text])  
                # Get RAW (e.g. 2048-dim) embeddings exclusively for HaloScope to prevent data leaks
                raw_emb = self.guardrail.get_raw_base_embeddings(text)
                
                mlp_score = 0.0
                if self.mlp_probe:
                    mlp_score = torch.sigmoid(self.mlp_probe(proj_emb)).item()
                
                # Feed the RAW embedding to HaloScope
                haloscope_score = self.haloscope_baseline.score(raw_emb)

                # SOTA: SEP — single forward-pass surrogate for Semantic Entropy
                sep_score = self.sep_baseline.score(raw_emb) if self.sep_baseline.is_fitted else float("nan")

                # SOTA 4 & 5: AutoFact NLI & ChainPoll LLM Judge
                autofact_score = self.autofact_baseline.score(context, answer)
                chainpoll_score = self.chainpoll_baseline.score(context, answer)

            # 4. Proposed: PCNet Guardrail
            p_nll, p_unc = self.proposed_pcnet_guardrail(text, n_samples=20)
            
            # Store ALL results
            results["labels"].append(label)
            results["baseline_nll"].append(b_nll)
            results["baseline_entropy"].append(b_entropy)
            results["sota_semantic_entropy"].append(sem_entropy)
            results["sota_mlp_latent"].append(mlp_score)
            results["proposed_latent_nll"].append(p_nll)
            results["proposed_epistemic_unc"].append(p_unc)
            results["sota_haloscope_latent"].append(haloscope_score)
            results["sota_autofact_nli"].append(autofact_score)
            results["sota_chainpoll_judgement"].append(chainpoll_score)
            results["sota_sep"].append(sep_score)

        return self._compute_metrics(results, dataset_name, llm_name)

    from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score
    import numpy as np

    def _compute_metrics(self, results, dataset_name, llm_name):
        """Calculates comprehensive classification and uncertainty metrics."""
        y_true = np.array(results["labels"])
        
        metrics_summary = {}
        
        score_dict = {
            "Baseline: Token NLL": np.array(results["baseline_nll"]),
            "Baseline: Token Entropy": np.array(results["baseline_entropy"]),
            "SOTA: Semantic Entropy": np.array(results["sota_semantic_entropy"]),
            "SOTA: Latent MLP Probe": np.array(results["sota_mlp_latent"]),
            "SOTA: HaloScope Recon Error": np.array(results["sota_haloscope_latent"]),
            "SOTA: AutoFact NLI Contradict": np.array(results["sota_autofact_nli"]),
            "SOTA: ChainPoll LLM Judge": np.array(results["sota_chainpoll_judgement"]),
            "SOTA: SEP (Semantic Entropy Probe)": np.array(results["sota_sep"]),
            "Proposed: PCNet Latent NLL": np.array(results["proposed_latent_nll"]),
            "Proposed: PCNet Epistemic Unc": np.array(results["proposed_epistemic_unc"])
        }
        
        metrics_summary["Experiment_Info"] = {
            "dataset": dataset_name,
            "llm": llm_name,
            "total_samples": len(y_true),
            "hallucinations_done_by_llm": int(np.sum(y_true == 1)),
            "factual_done_by_llm": int(np.sum(y_true == 0))
        }

        print("\n" + "="*60)
        print("📊 Comprehensive Benchmark Results")
        print("="*60)
        
        for name, scores in score_dict.items():
            # Exclude NaN samples (e.g. AutoFact on no-context datasets like TruthfulQA
            # deliberately returns NaN rather than biasing to 0.5 / min-score).
            scores = np.asarray(scores, dtype=np.float64)
            nan_mask = np.isnan(scores)
            if nan_mask.all() or scores.size == 0:
                # Baseline is fully inapplicable on this dataset
                auroc, auprc, fpr95, best_f1, best_acc = float("nan"), float("nan"), float("nan"), 0.0, 0.0
                best_prec, best_rec = 0.0, 0.0
                y_pred = np.zeros_like(y_true)
                metrics_summary[name] = {
                    "AUROC": auroc, "AUPRC": auprc, "FPR95": fpr95,
                    "Best_F1": best_f1, "Accuracy": best_acc,
                    "Precision": best_prec, "Recall": best_rec,
                    "number_evaluated_as_hallucinations": 0,
                    "number_evaluated_as_factual": 0,
                    "note": "all-NaN — baseline inapplicable on this dataset",
                }
                print(f"{name}: inapplicable on this dataset (all-NaN)\n")
                continue
            if nan_mask.any():
                scores_eval = scores[~nan_mask]
                y_true_eval = y_true[~nan_mask]
            else:
                scores_eval = scores
                y_true_eval = y_true

            # Safety check: if an array is completely flat (e.g., failed to compute)
            if np.all(scores_eval == scores_eval[0]):
                auroc, auprc, fpr95, best_f1, best_acc = 0.5, 0.5, 1.0, 0.0, 0.5
                best_prec, best_rec = 0.0, 0.0
                y_pred = np.zeros_like(y_true_eval)
            else:
                # 1. Standard Area Metrics
                auroc = roc_auc_score(y_true_eval, scores_eval)
                precision_arr, recall_arr, pr_thresholds = precision_recall_curve(y_true_eval, scores_eval)
                auprc = auc(recall_arr, precision_arr)

                # 2. FPR @ 95% TPR
                fpr, tpr, roc_thresholds = roc_curve(y_true_eval, scores_eval)
                idx95 = np.where(tpr >= 0.95)[0][0]
                fpr95 = fpr[idx95]

                # 3. Optimal Threshold Metrics (Max F1)
                f1_scores = np.divide(
                    2 * (precision_arr * recall_arr),
                    (precision_arr + recall_arr),
                    out=np.zeros_like(precision_arr),
                    where=(precision_arr + recall_arr) != 0
                )

                best_idx = np.argmax(f1_scores)
                best_f1 = f1_scores[best_idx]

                if best_idx < len(pr_thresholds):
                    best_thresh = pr_thresholds[best_idx]
                    y_pred = (scores_eval >= best_thresh).astype(int)
                else:
                    y_pred = np.ones_like(y_true_eval)

                best_acc = accuracy_score(y_true_eval, y_pred)
                best_prec = precision_score(y_true_eval, y_pred, zero_division=0)
                best_rec = recall_score(y_true_eval, y_pred, zero_division=0)
                
            metrics_summary[name] = {
                "AUROC": auroc, "AUPRC": auprc, "FPR95": fpr95,
                "Best_F1": best_f1, "Accuracy": best_acc,
                "Precision": best_prec, "Recall": best_rec,
                "number_evaluated_as_hallucinations": int(np.sum(y_pred == 1)),
                "number_evaluated_as_factual": int(np.sum(y_pred == 0)),
                "n_excluded_nan": int(nan_mask.sum()),
            }
            
            print(f"{name}:")
            print(f"  -> AUROC:     {auroc:.4f}")
            print(f"  -> AUPRC:     {auprc:.4f}")
            print(f"  -> FPR@95TPR: {fpr95:.4f}  (Lower is better)")
            print(f"  -> Optimal Point Metrics (Max F1 Threshold):")
            print(f"     * F1:        {best_f1:.4f}")
            print(f"     * Accuracy:  {best_acc:.4f}")
            print(f"     * Precision: {best_prec:.4f}")
            print(f"     * Recall:    {best_rec:.4f}\n")
            
        return metrics_summary

class BaselineDetectionExperiment:
    def __init__(self, llm, tokenizer, device="cuda"):
        self.device = device
        self.llm = llm
        self.tokenizer = tokenizer
        
        self.llm.eval()

        self.semantic_baseline = SemanticConsistencyBaseline(self.llm, self.tokenizer, device)
        self.haloscope_baseline = HaloScopeLatentBaseline(n_components=128, device=device)
        self.autofact_baseline = AutoFactNLIBaseline(device=device)
        self.chainpoll_baseline = ChainPollBaseline(self.llm, self.tokenizer, device)
        self.sep_baseline = SemanticEntropyProbe(device=device)

    def fit_haloscope(self, train_embeddings):
        self.haloscope_baseline.fit(train_embeddings)

    def fit_sep(self, train_embeddings, train_se_targets):
        self.sep_baseline.fit(train_embeddings, train_se_targets)

    def compute_semantic_entropy_for_fit(self, texts, num_samples=5, max_new_tokens=40, temperature=1.0):
        from tqdm import tqdm
        targets = []
        for t in tqdm(texts, desc="  [SEP] Computing SE targets"):
            try:
                se = self.semantic_baseline.compute_semantic_entropy(
                    t, num_samples=num_samples, max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                print(f"  [SEP] SE compute failed ({e}) — using NaN.")
                se = float("nan")
            targets.append(se)
        return np.array(targets, dtype=np.float64)

    def _parse_context_and_answer(self, text):
        context = ""
        answer = ""
        if "Context: " in text:
            context_part = text.split("Context: ")[1]
            if "\nTask: " in context_part:
                context = context_part.split("\nTask: ")[0]
            elif "\nQuestion: " in context_part:
                context = context_part.split("\nQuestion: ")[0]
            elif "\nAnswer: " in context_part:
                context = context_part.split("\nAnswer: ")[0]
        if "\nSummary: " in text:
            answer = text.split("\nSummary: ")[1]
        elif "\nAnswer: " in text:
            answer = text.split("\nAnswer: ")[1]
        if not context and "Question: " in text:
            context_part = text.split("Question: ")[1]
            if "\nAnswer: " in context_part:
                context = context_part.split("\nAnswer: ")[0]
        return context.strip(), answer.strip()

    @torch.no_grad()
    def baseline_token_likelihood_and_entropy(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        outputs = self.llm(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        import torch.nn.functional as F
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        avg_nll = token_nll.mean().item()
        
        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        avg_entropy = entropy.mean().item()
        return avg_nll, avg_entropy

    def run_benchmark(self, dataset, dataset_name="Unknown", llm_name="Unknown"):
        from tqdm import tqdm
        print("🚀 Starting Baseline Benchmark...")
        
        results = {
            "labels": [],
            "baseline_nll": [],
            "baseline_entropy": [],
            "sota_semantic_entropy": [],
            "sota_haloscope_latent": [],
            "sota_autofact_nli": [],
            "sota_chainpoll_judgement": [],
            "sota_sep": [],
        }

        for text, label in tqdm(dataset, desc="Evaluating"):
            b_nll, b_entropy = self.baseline_token_likelihood_and_entropy(text)
            sem_entropy = self.semantic_baseline.compute_semantic_entropy(text, num_samples=10)
            context, answer = self._parse_context_and_answer(text)

            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                outputs = self.llm(**inputs, output_hidden_states=True)
                raw_emb = outputs.hidden_states[-1][:, -1, :].detach()

                haloscope_score = self.haloscope_baseline.score(raw_emb)
                sep_score = self.sep_baseline.score(raw_emb) if self.sep_baseline.is_fitted else float("nan")
                autofact_score = self.autofact_baseline.score(context, answer)
                chainpoll_score = self.chainpoll_baseline.score(context, answer)

            results["labels"].append(label)
            results["baseline_nll"].append(b_nll)
            results["baseline_entropy"].append(b_entropy)
            results["sota_semantic_entropy"].append(sem_entropy)
            results["sota_haloscope_latent"].append(haloscope_score)
            results["sota_autofact_nli"].append(autofact_score)
            results["sota_chainpoll_judgement"].append(chainpoll_score)
            results["sota_sep"].append(sep_score)

        return self._compute_metrics(results, dataset_name, llm_name)

    def _compute_metrics(self, results, dataset_name, llm_name):
        from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score
        import numpy as np
        
        y_true = np.array(results["labels"])
        metrics_summary = {}
        
        score_dict = {
            "Baseline: Token NLL": np.array(results["baseline_nll"]),
            "Baseline: Token Entropy": np.array(results["baseline_entropy"]),
            "SOTA: Semantic Entropy": np.array(results["sota_semantic_entropy"]),
            "SOTA: HaloScope Recon Error": np.array(results["sota_haloscope_latent"]),
            "SOTA: AutoFact NLI Contradict": np.array(results["sota_autofact_nli"]),
            "SOTA: ChainPoll LLM Judge": np.array(results["sota_chainpoll_judgement"]),
            "SOTA: SEP (Semantic Entropy Probe)": np.array(results["sota_sep"]),
        }
        
        metrics_summary["Experiment_Info"] = {
            "dataset": dataset_name,
            "llm": llm_name,
            "total_samples": len(y_true),
            "hallucinations_done_by_llm": int(np.sum(y_true == 1)),
            "factual_done_by_llm": int(np.sum(y_true == 0))
        }

        print("\n" + "="*60)
        print("📊 Baseline Benchmark Results")
        print("="*60)
        
        for name, scores in score_dict.items():
            scores = np.asarray(scores, dtype=np.float64)
            nan_mask = np.isnan(scores)
            if nan_mask.all() or scores.size == 0:
                y_pred = np.zeros_like(y_true)
                metrics_summary[name] = {
                    "AUROC": float("nan"), "AUPRC": float("nan"), "FPR95": float("nan"),
                    "Best_F1": 0.0, "Accuracy": 0.0,
                    "Precision": 0.0, "Recall": 0.0,
                    "number_evaluated_as_hallucinations": 0,
                    "number_evaluated_as_factual": 0,
                    "note": "all-NaN — baseline inapplicable on this dataset",
                }
                print(f"{name}: inapplicable on this dataset (all-NaN)\n")
                continue
                
            if nan_mask.any():
                scores_eval = scores[~nan_mask]
                y_true_eval = y_true[~nan_mask]
            else:
                scores_eval = scores
                y_true_eval = y_true

            if np.all(scores_eval == scores_eval[0]):
                auroc, auprc, fpr95, best_f1, best_acc = 0.5, 0.5, 1.0, 0.0, 0.5
                best_prec, best_rec = 0.0, 0.0
                y_pred = np.zeros_like(y_true_eval)
            else:
                auroc = roc_auc_score(y_true_eval, scores_eval)
                precision_arr, recall_arr, pr_thresholds = precision_recall_curve(y_true_eval, scores_eval)
                auprc = auc(recall_arr, precision_arr)
                fpr, tpr, roc_thresholds = roc_curve(y_true_eval, scores_eval)
                try:
                    idx95 = np.where(tpr >= 0.95)[0][0]
                    fpr95 = fpr[idx95]
                except Exception:
                    fpr95 = float("nan")

                f1_scores = np.divide(
                    2 * (precision_arr * recall_arr),
                    (precision_arr + recall_arr),
                    out=np.zeros_like(precision_arr),
                    where=(precision_arr + recall_arr) != 0
                )
                best_idx = np.argmax(f1_scores)
                best_f1 = f1_scores[best_idx]
                if best_idx < len(pr_thresholds):
                    best_thresh = pr_thresholds[best_idx]
                    y_pred = (scores_eval >= best_thresh).astype(int)
                else:
                    y_pred = np.ones_like(y_true_eval)

                best_acc = accuracy_score(y_true_eval, y_pred)
                best_prec = precision_score(y_true_eval, y_pred, zero_division=0)
                best_rec = recall_score(y_true_eval, y_pred, zero_division=0)
                
            metrics_summary[name] = {
                "AUROC": auroc, "AUPRC": auprc, "FPR95": fpr95,
                "Best_F1": best_f1, "Accuracy": best_acc,
                "Precision": best_prec, "Recall": best_rec,
                "number_evaluated_as_hallucinations": int(np.sum(y_pred == 1)),
                "number_evaluated_as_factual": int(np.sum(y_pred == 0)),
                "n_excluded_nan": int(nan_mask.sum()),
            }
            
            print(f"{name}:")
            print(f"  -> AUROC:     {auroc:.4f}")
            print(f"  -> AUPRC:     {auprc:.4f}")
            print(f"  -> FPR@95TPR: {fpr95:.4f}")
            print(f"  -> Optimal Point Metrics (Max F1 Threshold):")
            print(f"     * F1:        {best_f1:.4f}")
            print(f"     * Accuracy:  {best_acc:.4f}")
            print(f"     * Precision: {best_prec:.4f}")
            print(f"     * Recall:    {best_rec:.4f}\n")
            
        return metrics_summary
