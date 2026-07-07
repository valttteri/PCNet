import math
import torch
import torch.nn.functional as F


def alpha_mass_mask(logits, alpha=0.1):
    if alpha is None or alpha <= 0:
        return torch.zeros_like(logits, dtype=torch.bool)
    log_p = F.log_softmax(logits, dim=-1)
    log_max = log_p.max(dim=-1, keepdim=True).values
    return log_p < (log_max + math.log(alpha))


def jensen_shannon_divergence(log_p, log_q):
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = (m + 1e-12).log()
    kl_pm = (p * (log_p - log_m)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def dola_candidate_layers(n_layers, bucket="high"):
    half = n_layers // 2
    if bucket == "low":
        return list(range(0, half, 2))
    return list(range(half, n_layers, 2))


class AdditiveCorrectionHook:
    def __init__(self, layer_name, delta_h, alpha=0.15):
        self.layer_name = layer_name
        self.delta_h = delta_h.view(1, 1, -1)   # (1, 1, hidden)
        self.alpha = alpha
        self.handles = []

    def hook_fn(self):
        def _hook(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            t = t + self.alpha * self.delta_h.to(t.dtype)
            return (t,) + output[1:] if isinstance(output, tuple) else t
        return _hook

    def attach(self, model):
        for name, module in model.named_modules():
            if name == self.layer_name:
                self.handles.append(module.register_forward_hook(self.hook_fn()))
                break

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []

def _get_llm_layers(llm):
    """Return the ordered list of decoder/transformer blocks."""
    if hasattr(llm, "model") and hasattr(llm.model, "layers"):
        return llm.model.layers
    if hasattr(llm, "transformer") and hasattr(llm.transformer, "h"):
        return llm.transformer.h
    raise AttributeError("Cannot locate decoder layers on this LLM.")


def _head_dims(llm):
    cfg = llm.config
    n_heads  = getattr(cfg, "num_attention_heads", None)
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None and n_heads is not None:
        head_dim = cfg.hidden_size // n_heads
    return n_heads, head_dim


def calibrate_per_head_iti(llm, samples, n_top_heads=48, last_token_only=True):
    layers   = _get_llm_layers(llm)
    n_layers = len(layers)
    n_heads, head_dim = _head_dims(llm)
    if n_heads is None or head_dim is None:
        raise RuntimeError("Could not derive (n_heads, head_dim) from llm.config.")

    device = next(llm.parameters()).device
    sum_f  = torch.zeros(n_layers, n_heads, head_dim, device=device, dtype=torch.float32)
    sum_h  = torch.zeros_like(sum_f)
    sumsq_f = torch.zeros(n_layers, n_heads, device=device, dtype=torch.float32)
    sumsq_h = torch.zeros_like(sumsq_f)
    n_f = 0
    n_h = 0
    captured = {}

    def _make_capture(li):
        def _hook(module, args):
            x = args[0]                                  # (B, T, n_heads*head_dim)
            B, T, _ = x.shape
            captured[li] = x.detach().view(B, T, n_heads, head_dim).float()
        return _hook

    handles = []
    for li, layer in enumerate(layers):
        try:
            o_proj = layer.self_attn.o_proj
        except AttributeError:
            return {}                                     # unsupported architecture
        handles.append(o_proj.register_forward_pre_hook(_make_capture(li)))

    try:
        for inputs_tok, gt_label in samples:
            with torch.no_grad():
                _ = llm(**inputs_tok)
            for li in range(n_layers):
                if li not in captured:
                    continue
                x_h = captured[li]                        # (B, T, H, D)
                # Take last prompt token (where the answer's first token is generated).
                acts = x_h[:, -1, :, :] if last_token_only else x_h.mean(dim=1)
                # Update per-head running sums (factual or hallu).
                if gt_label == 0:
                    sum_f[li] += acts.sum(dim=0)
                else:
                    sum_h[li] += acts.sum(dim=0)
            captured.clear()
            if gt_label == 0: n_f += 1
            else:             n_h += 1
    finally:
        for hd in handles:
            hd.remove()

    if n_f == 0 or n_h == 0:
        return {}

    mean_f = sum_f / n_f
    mean_h = sum_h / n_h
    diff   = mean_f - mean_h                              # (L, H, D)
    diff_norm = diff.norm(dim=-1)                         # (L, H)
    theta = diff / (diff_norm.unsqueeze(-1) + 1e-9)

    # Second pass: variance along θ_h per class.
    captured = {}
    handles = []
    for li, layer in enumerate(layers):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(_make_capture(li)))

    try:
        for inputs_tok, gt_label in samples:
            with torch.no_grad():
                _ = llm(**inputs_tok)
            for li in range(n_layers):
                if li not in captured:
                    continue
                acts = captured[li][:, -1, :, :]                  # (B, H, D)
                proj = (acts * theta[li].unsqueeze(0)).sum(-1)    # (B, H)
                mu_proj = (mean_f[li] if gt_label == 0 else mean_h[li]) \
                    .mul(theta[li]).sum(-1)                       # (H,)
                centred = proj - mu_proj.unsqueeze(0)
                if gt_label == 0:
                    sumsq_f += (centred ** 2).sum(dim=0)
                else:
                    sumsq_h += (centred ** 2).sum(dim=0)
            captured.clear()
    finally:
        for hd in handles:
            hd.remove()

    var_f = sumsq_f / max(1, n_f)
    var_h = sumsq_h / max(1, n_h)
    sigma = (0.5 * (var_f + var_h)).sqrt()                    # pooled per-head std

    # Fisher discriminant proxy for probe accuracy.
    fisher = (diff_norm ** 2) / (var_f + var_h + 1e-6)        # (L, H)
    flat = fisher.flatten()
    n_top = min(n_top_heads, flat.numel())
    _, top_idx = torch.topk(flat, n_top)

    head_specs = {}
    for idx in top_idx.tolist():
        li = idx // n_heads
        hi = idx %  n_heads
        head_specs.setdefault(li, []).append(
            (hi, theta[li, hi].cpu().clone(), float(sigma[li, hi].item()))
        )
    return head_specs


class PerHeadITIHook:
    def __init__(self, llm, head_specs, alpha=15.0):
        self.llm = llm
        self.head_specs = head_specs
        self.alpha = alpha
        self.n_heads, self.head_dim = _head_dims(llm)
        self.handles = []

    def _make_hook(self, specs):
        n_heads, head_dim, alpha = self.n_heads, self.head_dim, self.alpha
        def _pre_hook(module, args):
            x = args[0]                                       # (B, T, H*D)
            B, T, _ = x.shape
            x_h = x.view(B, T, n_heads, head_dim)
            for h, theta, sigma in specs:
                shift = alpha * sigma * theta.to(x.device, x.dtype)
                x_h[..., h, :] = x_h[..., h, :] + shift
            return (x_h.view(B, T, -1),) + args[1:]
        return _pre_hook

    def attach(self, model=None):
        layers = _get_llm_layers(model or self.llm)
        for li, specs in self.head_specs.items():
            try:
                o_proj = layers[li].self_attn.o_proj
            except (IndexError, AttributeError):
                continue
            self.handles.append(o_proj.register_forward_pre_hook(self._make_hook(specs)))

    def remove(self):
        for hd in self.handles:
            hd.remove()
        self.handles = []


def generate_vanilla(guardrail, inputs, max_new_tokens, pad_id, stop_ids=None):
    # HF `generate` accepts eos_token_id as int or list[int]. Passing the full
    # stop list ensures Llama-3 <|eot_id|>, Qwen <|im_end|>, etc. all halt
    # generation instead of running to max_new_tokens.
    eos = list(stop_ids) if stop_ids else pad_id
    with torch.no_grad():
        return guardrail.llm.generate(
            **inputs, max_new_tokens=max_new_tokens,
            pad_token_id=pad_id, eos_token_id=eos, do_sample=False,
        )

def generate_iti(guardrail, inputs, max_new_tokens, pad_id, norm_name=None,
                 iti_dir_h=None, alpha=15.0, head_specs=None, stop_ids=None):
    if head_specs:
        hook = PerHeadITIHook(guardrail.llm, head_specs, alpha=alpha)
        hook.attach()
        out = generate_vanilla(guardrail, inputs, max_new_tokens, pad_id, stop_ids=stop_ids)
        hook.remove()
        return out, 1

    if iti_dir_h is None or not norm_name:
        return generate_vanilla(guardrail, inputs, max_new_tokens, pad_id, stop_ids=stop_ids), 0

    hook = AdditiveCorrectionHook(norm_name, iti_dir_h.squeeze(0), alpha=alpha)
    hook.attach(guardrail.llm)
    out = generate_vanilla(guardrail, inputs, max_new_tokens, pad_id, stop_ids=stop_ids)
    hook.remove()
    return out, 1

def generate_dola(guardrail, inputs, max_new_tokens, pad_id, alpha=0.1,
                  bucket="high", stop_ids=None):
    llm = guardrail.llm
    stop_set = set(stop_ids) if stop_ids else {pad_id}
    n_layers = getattr(llm.config, "num_hidden_layers",
                       getattr(llm.config, "n_layer", None))
    if n_layers is None:
        return generate_vanilla(guardrail, inputs, max_new_tokens, pad_id,
                                stop_ids=stop_ids)

    cand_layers = dola_candidate_layers(n_layers, bucket=bucket)
    norm_mod = getattr(getattr(llm, "model", None), "norm", None)
    lm_head  = llm.lm_head
    head_dtype = next(lm_head.parameters()).dtype
    generated = inputs["input_ids"].clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outs = llm(input_ids=generated, output_hidden_states=True)
            logits_final = outs.logits[:, -1, :].float()             # (B, V)
            log_pN = F.log_softmax(logits_final, dim=-1)             # (B, V)

            # Compute log p_j for each candidate layer (project hidden via norm+lm_head).
            cand_logps = []
            for j in cand_layers:
                h_j = outs.hidden_states[j + 1][:, -1:, :]            # (B, 1, H)
                if norm_mod is not None:
                    h_j = norm_mod(h_j.float()).to(head_dtype)
                else:
                    h_j = h_j.to(head_dtype)
                logits_j = lm_head(h_j).squeeze(1).float()           # (B, V)
                cand_logps.append(F.log_softmax(logits_j, dim=-1))
            cand_logps = torch.stack(cand_logps, dim=0)              # (J, B, V)

            # JSD per candidate layer; pick the layer maximising JSD with p_N.
            jsds = jensen_shannon_divergence(
                log_pN.unsqueeze(0).expand_as(cand_logps), cand_logps
            )                                                        # (J, B)
            m_idx = jsds.argmax(dim=0)                               # (B,)
            log_pM = cand_logps[m_idx, torch.arange(cand_logps.shape[1])]  # (B, V)

            # Adaptive plausibility on p_N (paper Eq. 4). Then F = log p_N − log p_M.
            forbid = alpha_mass_mask(logits_final, alpha=alpha)
            score  = log_pN - log_pM
            score  = score.masked_fill(forbid, -float("inf"))

            next_tok = torch.argmax(score, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_tok], dim=-1)
        if next_tok.item() in stop_set:
            break

    return generated

ICD_INDUCING_PREFIX = (
    "You are an unreliable assistant. You frequently invent facts, "
    "names, dates, and citations even when you are not sure they are real. "
    "Answer the following confidently, even if you must guess.\n\n"
)


def _prepend_text_to_input_ids(tokenizer, prefix_text, input_ids, device):
    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    new_text = prefix_text + decoded
    enc = tokenizer(new_text, return_tensors="pt").to(device)
    return enc["input_ids"]


def generate_icd(
    guardrail, inputs, max_new_tokens, pad_id, alpha=1.0,
    plausibility_alpha=0.1, inducing_prefix=ICD_INDUCING_PREFIX,
    stop_ids=None,
):
    llm = guardrail.llm
    tokenizer = guardrail.tokenizer
    device = inputs["input_ids"].device
    stop_set = set(stop_ids) if stop_ids else {pad_id}

    expert_ids = inputs["input_ids"].clone()
    amateur_ids = _prepend_text_to_input_ids(
        tokenizer, inducing_prefix, expert_ids, device
    )

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits_expert  = llm(input_ids=expert_ids).logits[:, -1, :].float()
            logits_amateur = llm(input_ids=amateur_ids).logits[:, -1, :].float()

        log_pe = F.log_softmax(logits_expert, dim=-1)
        log_pa = F.log_softmax(logits_amateur, dim=-1)

        forbid = alpha_mass_mask(logits_expert, alpha=plausibility_alpha)
        contrastive = (1.0 + alpha) * log_pe - alpha * log_pa
        contrastive = contrastive.masked_fill(forbid, -float("inf"))

        next_tok = torch.argmax(contrastive, dim=-1, keepdim=True)
        expert_ids = torch.cat([expert_ids, next_tok], dim=-1)
        amateur_ids = torch.cat([amateur_ids, next_tok], dim=-1)

        if next_tok.item() in stop_set:
            break

    return expert_ids


def generate_truthx(guardrail, inputs, max_new_tokens, pad_id, truthx_model, stop_ids=None):
    if truthx_model is None:
        return generate_vanilla(guardrail, inputs, max_new_tokens, pad_id, stop_ids=stop_ids), 0
        
    # We apply TruthX edit directly on forward pass using a dynamic hook
    # TruthX typically edits layer L.
    layer_name = truthx_model.cur_layer_name if hasattr(truthx_model, 'cur_layer_name') else "model.layers.15" 
    
    class TruthXHook:
        def __init__(self, layer_name, tx_m):
            self.layer_name = layer_name
            self.tx_m = tx_m
            self.handles = []
        def hook_fn(self):
            def _hook(module, inps, output):
                t = output[0] if isinstance(output, tuple) else output
                t = self.tx_m.edit(t) 
                return (t,) + output[1:] if isinstance(output, tuple) else t
            return _hook
        def attach(self, model):
            for name, module in model.named_modules():
                if name == self.layer_name:
                    self.handles.append(module.register_forward_hook(self.hook_fn()))
                    break
        def remove(self):
            for h in self.handles: h.remove()
            self.handles = []

    hook = TruthXHook(layer_name, truthx_model)
    hook.attach(guardrail.llm)
    out = generate_vanilla(guardrail, inputs, max_new_tokens, pad_id, stop_ids=stop_ids)
    hook.remove()
    return out, 1

class SADICorrectionHook:
    def __init__(self, layer_name, diff_vec, delta=0.30, top_k_ratio=0.10):
        self.layer_name = layer_name
        self.delta = delta
        d = diff_vec.view(-1)
        k = max(1, int(d.numel() * top_k_ratio))
        _, topk_idx = torch.topk(d.abs(), k)
        mask = torch.zeros_like(d)
        mask[topk_idx] = 1.0
        self.mask = mask.view(1, 1, -1)   # (1, 1, hidden)
        self.handles = []

    def hook_fn(self):
        def _hook(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            m = self.mask.to(t.device, dtype=t.dtype)
            t = t + self.delta * (t * m)
            return (t,) + output[1:] if isinstance(output, tuple) else t
        return _hook

    def attach(self, model):
        for name, module in model.named_modules():
            if name == self.layer_name:
                self.handles.append(module.register_forward_hook(self.hook_fn()))
                break

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []

def generate_sadi(guardrail, inputs, max_new_tokens, pad_id, norm_name, sadi_dir_h,
                  delta=0.30, top_k_ratio=0.10, stop_ids=None):
    if sadi_dir_h is None or not norm_name:
        return generate_vanilla(guardrail, inputs, max_new_tokens, pad_id, stop_ids=stop_ids), 0

    hook = SADICorrectionHook(norm_name, sadi_dir_h.squeeze(0), delta=delta, top_k_ratio=top_k_ratio)
    hook.attach(guardrail.llm)
    out = generate_vanilla(guardrail, inputs, max_new_tokens, pad_id, stop_ids=stop_ids)
    hook.remove()
    return out, 1

class ACTCorrectionHook:
    def __init__(self, layer_name, steer_vec, alpha=0.20, beta=0.0):
        self.layer_name = layer_name
        self.v = steer_vec.view(1, 1, -1)              # (1, 1, hidden)
        self.v_norm_sq = float(steer_vec.norm() ** 2) + 1e-8
        self.alpha = alpha
        self.beta = beta
        self.handles = []

    def hook_fn(self):
        def _hook(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            v = self.v.to(t.device, dtype=t.dtype)
            # Projection magnitude: scalar per (batch, seq) position
            proj = (t * v).sum(dim=-1, keepdim=True) / self.v_norm_sq
            # Linear probe: p_θ(h) ≈ σ(proj)
            probe = torch.sigmoid(proj)
            coeff = self.alpha * (1.0 - probe + self.beta)
            t = t + coeff * v
            return (t,) + output[1:] if isinstance(output, tuple) else t
        return _hook

    def attach(self, model):
        for name, module in model.named_modules():
            if name == self.layer_name:
                self.handles.append(module.register_forward_hook(self.hook_fn()))
                break

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []

def generate_act(guardrail, inputs, max_new_tokens, pad_id, norm_name, act_dirs,
                 alpha=0.20, beta=0.0, stop_ids=None):
    if act_dirs is None or not norm_name:
        return generate_vanilla(guardrail, inputs, max_new_tokens, pad_id, stop_ids=stop_ids), 0

    if isinstance(act_dirs, dict):
        hooks = []
        for l_name, steer_v in act_dirs.items():
            hook = ACTCorrectionHook(l_name, steer_v.squeeze(0), alpha=alpha, beta=beta)
            hook.attach(guardrail.llm)
            hooks.append(hook)
        out = generate_vanilla(guardrail, inputs, max_new_tokens, pad_id, stop_ids=stop_ids)
        for hook in hooks: hook.remove()
        return out, 1
    else:
        hook = ACTCorrectionHook(norm_name, act_dirs.squeeze(0), alpha=alpha, beta=beta)
        hook.attach(guardrail.llm)
        out = generate_vanilla(guardrail, inputs, max_new_tokens, pad_id, stop_ids=stop_ids)
        hook.remove()
        return out, 1

class AdaSteerHook:
    def __init__(self, layer_name, v_rd, v_hd, mu_hallu,
                 w_r=1.0, b_r=0.0, w_c=1.0, b_c=0.0, alpha=1.0):
        self.layer_name = layer_name
        self.v_rd = v_rd.view(1, 1, -1)
        self.v_hd = v_hd.view(1, 1, -1)
        self.mu_h = mu_hallu.view(1, 1, -1)
        self.w_r, self.b_r = float(w_r), float(b_r)
        self.w_c, self.b_c = float(w_c), float(b_c)
        self.alpha = float(alpha)
        self.handles = []

    def hook_fn(self):
        def _hook(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            v_rd = self.v_rd.to(t.device, t.dtype)
            v_hd = self.v_hd.to(t.device, t.dtype)
            mu_h = self.mu_h.to(t.device, t.dtype)
            centred = t - mu_h
            pos_rd = (centred * v_rd).sum(dim=-1, keepdim=True)
            pos_hd = (centred * v_hd).sum(dim=-1, keepdim=True)
            lambda_r = self.alpha * (self.w_r * pos_rd + self.b_r)
            lambda_c = self.alpha * (self.w_c * pos_hd + self.b_c)
            t = t + lambda_r * v_rd + lambda_c * v_hd
            return (t,) + output[1:] if isinstance(output, tuple) else t
        return _hook

    def attach(self, model):
        for name, module in model.named_modules():
            if name == self.layer_name:
                self.handles.append(module.register_forward_hook(self.hook_fn()))
                break

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []


def calibrate_adasteer(llm, samples, layer_name):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None

    target_mod = None
    for name, mod in llm.named_modules():
        if name == layer_name:
            target_mod = mod
            break
    if target_mod is None:
        return None

    captured = []
    def _capture(module, inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        captured.append(t[:, -1, :].detach().float().cpu())

    handle = target_mod.register_forward_hook(_capture)
    labels = []
    try:
        for inp, lab in samples:
            with torch.no_grad():
                _ = llm(**inp)
            labels.append(int(lab))
    finally:
        handle.remove()

    if not captured:
        return None
    H = torch.cat(captured, dim=0).numpy()                 # (N, hidden)
    y = torch.tensor(labels).numpy()
    if y.sum() == 0 or (y == 0).sum() == 0:
        return None

    mu_f = H[y == 0].mean(axis=0)
    mu_h = H[y == 1].mean(axis=0)
    delta_mean = mu_f - mu_h

    # Top-2 right singular vectors of factual rows (centred at hallu mean).
    diff = H[y == 0] - mu_h[None, :]
    try:
        _, _, Vt = torch.linalg.svd(torch.from_numpy(diff), full_matrices=False)
        Vt = Vt.numpy()
    except Exception:
        return None
    v_rd = Vt[0]
    v_hd = Vt[1] if Vt.shape[0] > 1 else delta_mean / (
        (delta_mean ** 2).sum() ** 0.5 + 1e-9)
    # Orient both with the factual direction so positive λ pushes towards factual.
    if (v_rd * delta_mean).sum() < 0: v_rd = -v_rd
    if (v_hd * delta_mean).sum() < 0: v_hd = -v_hd

    pos_rd = (H - mu_h[None, :]) @ v_rd                    # (N,)
    pos_hd = (H - mu_h[None, :]) @ v_hd                    # (N,)
    # Label = 0 → push UP λ along factual side. Fit so that high pos_* → high λ.
    # We label hallu = +1 so coefficient flips sign to push towards factual.
    lr_r = LogisticRegression(max_iter=1000)
    lr_r.fit(pos_rd.reshape(-1, 1), y)
    lr_c = LogisticRegression(max_iter=1000)
    lr_c.fit(pos_hd.reshape(-1, 1), y)
    # Negate coef so λ > 0 when sample is hallucinated (pushes towards factual).
    w_r = float(-lr_r.coef_[0, 0]); b_r = float(-lr_r.intercept_[0])
    w_c = float(-lr_c.coef_[0, 0]); b_c = float(-lr_c.intercept_[0])

    return {
        "layer_name": layer_name,
        "v_rd": torch.from_numpy(v_rd).float(),
        "v_hd": torch.from_numpy(v_hd).float(),
        "mu_hallu": torch.from_numpy(mu_h).float(),
        "w_r": w_r, "b_r": b_r,
        "w_c": w_c, "b_c": b_c,
    }


def generate_adasteer(guardrail, inputs, max_new_tokens, pad_id, norm_name,
                      adasteer_kwargs=None, alpha=1.0, stop_ids=None):
    if not adasteer_kwargs:
        return generate_vanilla(guardrail, inputs, max_new_tokens, pad_id,
                                stop_ids=stop_ids), 0
    hook = AdaSteerHook(alpha=alpha, **adasteer_kwargs)
    hook.attach(guardrail.llm)
    out = generate_vanilla(guardrail, inputs, max_new_tokens, pad_id,
                           stop_ids=stop_ids)
    hook.remove()
    return out, 1

def generate_hallucana(guardrail, inputs, max_new_tokens, pad_id, nll_threshold,
                       lookahead_k=8, top_alt=5, entropy_threshold=2.0,
                       score_alpha=1.0, stop_ids=None):
    llm = guardrail.llm
    pcnet = getattr(guardrail, 'pcnet', None)
    if pcnet is None:
        return generate_vanilla(guardrail, inputs, max_new_tokens, pad_id, stop_ids=stop_ids), 0

    stop_set = set(stop_ids) if stop_ids else {pad_id}
    generated = inputs["input_ids"].clone()
    target_layer_idx = guardrail.extraction_layer_idx

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outs = llm(input_ids=generated, output_hidden_states=True)

        logits = outs.logits[:, -1, :]
        probs  = torch.softmax(logits, dim=-1)
        # Shannon entropy of the token distribution
        log_p  = torch.log(probs + 1e-12)
        entropy = -(probs * log_p).sum(dim=-1).item()

        if entropy <= entropy_threshold:
            # Low-uncertainty step: greedy pick, no lookahead needed
            next_tok = torch.argmax(logits, dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_tok], dim=-1)
            if next_tok.item() in stop_set:
                break
            continue

        # High-uncertainty (critical) step: canary lookahead
        top_probs, top_ids = torch.topk(probs, k=top_alt, dim=-1)
        top_probs = top_probs[0]   # (top_alt,)
        top_ids   = top_ids[0]     # (top_alt,)

        best_tok   = top_ids[0].unsqueeze(0).unsqueeze(0)
        best_score = -float('inf')
        any_survivor = False

        for ci in range(top_alt):
            cand_tok = top_ids[ci].unsqueeze(0).unsqueeze(0)
            cand_gen = torch.cat([generated, cand_tok], dim=-1)

            # Greedy rollout of lookahead_k steps
            for _ in range(lookahead_k):
                with torch.no_grad():
                    roll_logits = llm(input_ids=cand_gen).logits[:, -1, :]
                n_tok  = torch.argmax(roll_logits, dim=-1).unsqueeze(-1)
                cand_gen = torch.cat([cand_gen, n_tok], dim=-1)
                if n_tok.item() in stop_set:
                    break

            # Evaluate trajectory with PCNet
            with torch.no_grad():
                roll_h = llm(input_ids=cand_gen,
                             output_hidden_states=True).hidden_states[target_layer_idx][:, -1:, :]
            traj_nll = -pcnet.log_prob(roll_h.cpu().to(torch.float32)).item()

            # Veto: discard branches beyond the NLL threshold
            if traj_nll > nll_threshold:
                continue

            # Score: logit probability weighted by faithfulness (lower NLL = higher score)
            score = float(top_probs[ci]) * (1.0 / (traj_nll + 1e-8)) ** score_alpha
            if score > best_score:
                best_score = score
                best_tok   = cand_tok
                any_survivor = True

        # If all candidates were vetoed, fall back to top-1 logit choice
        if not any_survivor:
            best_tok = top_ids[0].unsqueeze(0).unsqueeze(0)

        generated = torch.cat([generated, best_tok], dim=-1)
        if best_tok.item() in stop_set:
            break

    return generated, 1

