import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LatentMLPProbe(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, dropout_p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        return self.net(x)


from transformers import pipeline

class SemanticConsistencyBaseline:
    def __init__(self, llm, tokenizer, device="cpu",
                 nli_model_name="cross-encoder/nli-deberta-v3-small"):
        self.llm = llm
        self.tokenizer = tokenizer
        self.device = device

        device_id = -1
        if isinstance(device, str):
            if device == "cuda" and torch.cuda.is_available():
                device_id = 0
            elif device == "mps" and torch.backends.mps.is_available():
                device_id = "mps"

        # Lazy: instantiate NLI only once
        self.nli = pipeline(
            "text-classification",
            model=nli_model_name,
            device=device_id,
            top_k=None,
        )

    # ---------- helpers ----------
    @staticmethod
    def _strip_answer(text):
        """Return the prompt portion (everything before the existing answer).
        Falls back to the original text if no answer marker is found."""
        for marker in ["\nAnswer: ", "\nSummary: ", "Answer: ", "Summary: "]:
            if marker in text:
                return text.split(marker)[0] + marker.rstrip()
        return text

    def _bidirectional_entailment(self, a, b):
        """Return True iff a ⟷ b under the NLI cross-encoder."""
        if not a.strip() or not b.strip():
            return a.strip() == b.strip()
        try:
            r_ab = self.nli({"text": a, "text_pair": b}, top_k=None)
            r_ba = self.nli({"text": b, "text_pair": a}, top_k=None)
        except Exception:
            return False

        def is_entailment(result):
            top = max(result, key=lambda d: d["score"])
            return "entail" in top["label"].lower()

        return is_entailment(r_ab) and is_entailment(r_ba)

    @torch.no_grad()
    def _sample_with_logprobs(self, prompt, num_samples, max_new_tokens, temperature):
        """Generate num_samples stochastic continuations and return
        (texts, length-normalised sequence log-probs)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        texts, log_probs = [], []
        for _ in range(num_samples):
            out = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,  # paper uses plain sampling
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            seq = out.sequences[0]
            new_tokens = seq[prompt_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Per-step scores: list of length T with (1, vocab) logits
            step_logps = []
            for t, step_logits in enumerate(out.scores):
                tok_id = new_tokens[t].item()
                logp_t = F.log_softmax(step_logits[0].float(), dim=-1)[tok_id].item()
                step_logps.append(logp_t)
            if len(step_logps) == 0:
                ln_logp = -1e9
            else:
                ln_logp = float(np.mean(step_logps))  # length-normalised log-prob

            texts.append(text)
            log_probs.append(ln_logp)
        return texts, log_probs

    def compute_semantic_entropy(self, text, num_samples=10, max_new_tokens=60,
                                 temperature=1.0):
        """Semantic entropy with NLI clustering + probability-weighted masses."""
        prompt = self._strip_answer(text)
        texts, log_probs = self._sample_with_logprobs(
            prompt, num_samples=num_samples,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )

        if not texts:
            return 0.0

        # ---- Cluster by bidirectional entailment ----
        clusters = []  # list of lists of indices into `texts`
        for i, t_i in enumerate(texts):
            placed = False
            for cluster in clusters:
                rep = texts[cluster[0]]
                if self._bidirectional_entailment(rep, t_i):
                    cluster.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])

        # ---- Probability-weighted cluster masses ----
        # cluster mass = logsumexp of member length-normalised log-probs
        log_masses = []
        for cluster in clusters:
            member_lps = torch.tensor([log_probs[i] for i in cluster], dtype=torch.float64)
            log_masses.append(torch.logsumexp(member_lps, dim=0).item())
        log_masses = np.array(log_masses, dtype=np.float64)

        # Normalise to a probability distribution over clusters
        m = np.max(log_masses)
        norm = m + np.log(np.sum(np.exp(log_masses - m)) + 1e-12)
        log_probs_clusters = log_masses - norm
        probs_clusters = np.exp(log_probs_clusters)

        # Shannon entropy (nats)
        entropy = -np.sum(probs_clusters * log_probs_clusters)
        return float(entropy)

class HaloScopeLatentBaseline(nn.Module):
    def __init__(self, n_components=128, assumed_contamination=0.15, device="cuda",
                 lr=1e-3, weight_decay=3e-4, epochs=100):
        super().__init__()
        self.device = device
        self.n_components = n_components
        self.assumed_contamination = assumed_contamination
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.mean = None          # centering vector
        self.V_k = None           # [D, k] top right singular vectors
        self.mlp = None
        self.is_fitted = False

    # ---- HaloScope scoring (norm of projection onto top-k subspace) ----
    def _haloscope_score(self, X):
        """X: [N, D] tensor or ndarray, returns [N] numpy scores."""
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)
        Xc = X_np - self.mean
        proj = Xc @ self.V_k                                   # [N, k]
        return np.linalg.norm(proj, axis=1)                    # [N]

    def fit(self, wild_embeddings):
        print("\n  [HaloScope] Fitting (centered SVD → dual-threshold pseudo-labels → MLP)...")
        wild_np = wild_embeddings.detach().cpu().numpy().astype(np.float64)
        N, D = wild_np.shape
        k = min(self.n_components, D, N)

        # 1. Centre and SVD
        self.mean = wild_np.mean(axis=0, keepdims=True)        # [1, D]
        Xc = wild_np - self.mean
        # Economy SVD: Xc = U S Vt, right singular vectors are columns of V = Vt^T
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.V_k = Vt[:k].T                                    # [D, k]

        # 2. HaloScope membership scores (norm of projection)
        scores = self._haloscope_score(wild_np)                # [N]

        # 3. Dual-threshold pseudo-labels (discard middle band)
        tau = self.assumed_contamination
        hi = np.quantile(scores, 1.0 - tau)
        lo = np.quantile(scores, tau)
        pos_mask = scores >= hi
        neg_mask = scores <= lo
        keep = pos_mask | neg_mask
        if keep.sum() < 8:  # degenerate fallback
            keep = np.ones_like(keep, dtype=bool)
            pos_mask = scores >= np.median(scores)

        X_train = wild_np[keep]
        y_train = pos_mask[keep].astype(np.float32)

        # 4. 2-layer MLP, AdamW, realistic schedule
        self.mlp = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.ReLU(),
            nn.Linear(D // 2, 1),
        ).to(self.device)

        X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=self.device).unsqueeze(1)

        opt = torch.optim.AdamW(self.mlp.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        self.mlp.train()
        with torch.enable_grad():
            for _ in range(self.epochs):
                opt.zero_grad()
                logits = self.mlp(X_t)
                loss = loss_fn(logits, y_t)
                loss.backward()
                opt.step()

        self.mlp.eval()
        self.is_fitted = True
        print(f"  [HaloScope] Trained on {int(keep.sum())}/{N} samples "
              f"(pos={int(pos_mask[keep].sum())}, neg={int((~pos_mask[keep]).sum())}).")

    def score(self, test_embedding):
        if not self.is_fitted:
            return 0.5
        with torch.no_grad():
            x = test_embedding.float()
            if x.dim() == 1:
                x = x.unsqueeze(0)
            logits = self.mlp(x.to(self.device))
            return torch.sigmoid(logits).squeeze().item()

class AutoFactNLIBaseline:
    def __init__(self, device="cpu",
                 nli_model_name="cross-encoder/nli-deberta-v3-small"):
        device_id = -1
        if device == "cuda" and torch.cuda.is_available():
            device_id = 0
        elif device == "mps" and torch.backends.mps.is_available():
            device_id = "mps"

        self.nli_model = pipeline(
            "text-classification",
            model=nli_model_name,
            device=device_id,
            top_k=None,
        )

    # Sentence splitter — deliberately lightweight, no spaCy dependency
    _SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")

    def _decompose(self, answer):
        answer = (answer or "").strip()
        if not answer:
            return []
        parts = [p.strip() for p in self._SENT_SPLIT.split(answer) if p.strip()]
        return parts if parts else [answer]

    def _contradiction_prob(self, premise, hypothesis):
        try:
            result = self.nli_model({"text": premise, "text_pair": hypothesis}, top_k=None)
        except Exception as e:
            print(f"[NLI-contradiction] error: {e}")
            return 0.0
        for res in result:
            if "contradiction" in res["label"].lower():
                return float(res["score"])
        return 0.0

    @torch.no_grad()
    def score(self, context, answer):
        claims = self._decompose(answer)
        if not claims:
            return float("nan")  # nothing to check

        # Datasets without a grounding context (e.g. TruthfulQA) —
        # return NaN so the metric loop can exclude rather than biasing to 0.5.
        if not (context or "").strip():
            return float("nan")

        scores = [self._contradiction_prob(context, c) for c in claims]
        return float(max(scores)) if scores else float("nan")

class SemanticEntropyProbe:
    def __init__(self, ridge_lambda=1e-2, device="cuda"):
        self.device = device
        self.ridge_lambda = ridge_lambda
        self.W = None        # (D,)
        self.b = 0.0
        self.mean = None     # (1, D)
        self.std = None      # (1, D)
        self.is_fitted = False

    def fit(self, X, y):
        """X: (N, D) raw last-token hidden states; y: (N,) semantic-entropy targets (nats)."""
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy().astype(np.float64)
        else:
            X_np = np.asarray(X, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64)

        # Drop NaN targets (e.g. SE-fail samples) so they don't poison the fit.
        mask = ~np.isnan(y_np)
        X_np, y_np = X_np[mask], y_np[mask]
        if X_np.shape[0] < 8:
            print(f"  [SEP] Warning: only {X_np.shape[0]} valid samples — skipping fit.")
            return

        N, D = X_np.shape
        self.mean = X_np.mean(axis=0, keepdims=True)
        self.std = X_np.std(axis=0, keepdims=True) + 1e-6
        Xs = (X_np - self.mean) / self.std

        lam = self.ridge_lambda * D
        A = Xs.T @ Xs + lam * np.eye(D)
        rhs = Xs.T @ (y_np - y_np.mean())
        w = np.linalg.solve(A, rhs)
        self.W = w
        self.b = float(y_np.mean())
        self.is_fitted = True
        print(f"  [SEP] Fit on {N} samples (D={D}); train R^2 = "
              f"{1.0 - np.var((Xs @ w + self.b) - y_np) / (np.var(y_np) + 1e-9):.3f}")

    def score(self, x):
        """x: 1-D or 2-D tensor / ndarray → predicted semantic entropy."""
        if not self.is_fitted:
            return float("nan")
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy().astype(np.float64)
        else:
            x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        xs = (x - self.mean) / self.std
        return float((xs @ self.W + self.b).squeeze())

class ChainPollBaseline:
    def __init__(self, llm, tokenizer, device="cpu", K=5, temperature=0.7,
                 max_new_tokens=80):
        self.llm = llm
        self.tokenizer = tokenizer
        self.device = device
        self.K = K
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    _FINAL_YESNO = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)

    def _parse_vote(self, text):
        """Return 1.0 for YES, 0.0 for NO, NaN if undecided.
        Uses the LAST YES/NO token after the marker so CoT text doesn't leak."""
        upper = text.upper()
        if "CONCLUDE" in upper:
            tail = upper.split("CONCLUDE", 1)[1]
        else:
            tail = upper
        matches = self._FINAL_YESNO.findall(tail)
        if not matches:
            return float("nan")
        last = matches[-1].upper()
        return 1.0 if last == "YES" else 0.0

    @torch.no_grad()
    def score(self, context, answer):
        prompt = (
            f"Context: {context}\n"
            f"Answer: {answer}\n"
            f"Does the answer hallucinate or invent information not supported by the context? "
            f"Think step by step, then conclude with exactly 'YES' or 'NO'."
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        votes = []
        for _ in range(self.K):
            try:
                out = self.llm.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                new_tokens = out[0][inputs.input_ids.shape[1]:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                vote = self._parse_vote(text)
                if not np.isnan(vote):
                    votes.append(vote)
            except Exception as e:
                print(f"ChainPoll Error: {e}")

        if not votes:
            return 0.5  # abstain
        return float(np.mean(votes))
