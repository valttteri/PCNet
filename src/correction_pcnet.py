import torch
import torch.nn.functional as F
from probabilistic_circuits import log_gaussian_jit, log_laplace_jit, log_student_jit
from correction_baselines import generate_vanilla, AdditiveCorrectionHook
import numpy as np

def _final_norm_name(llm):
    for candidate in ["model.norm", "model.final_layernorm",
                      "transformer.ln_f", "gpt_neox.final_layer_norm"]:
        obj = llm
        try:
            for part in candidate.split("."): obj = getattr(obj, part)
            return candidate
        except AttributeError:
            continue
    return ""

def _pad_id(guardrail):
    pid = guardrail.tokenizer.eos_token_id
    if isinstance(pid, list): pid = pid[0]
    return pid if pid is not None else (guardrail.tokenizer.pad_token_id or 0)

class TargetedActivationClampingHook:
    def __init__(self, thresholds_dict, clamp_ratio=0.3):
        self.thresholds_dict = thresholds_dict
        self.clamp_ratio = clamp_ratio
        self.handles = []
    def hook_fn(self, layer_name):
        def _hook(module, inputs, output):
            if layer_name in self.thresholds_dict:
                flagged = self.thresholds_dict[layer_name]
                t = output[0] if isinstance(output, tuple) else output
                cap = torch.mean(torch.abs(t)).item() * self.clamp_ratio
                t[..., flagged] = torch.clamp(t[..., flagged], min=-cap, max=cap)
                return (t,) + output[1:] if isinstance(output, tuple) else t
        return _hook
    def attach(self, model):
        for name, module in model.named_modules():
            if name in self.thresholds_dict:
                self.handles.append(module.register_forward_hook(self.hook_fn(name)))
    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []

class ActivationAmplificationHook:
    def __init__(self, thresholds_dict, amp_factor=2.0):
        self.thresholds_dict = thresholds_dict
        self.amp_factor = amp_factor
        self.handles = []
    def hook_fn(self, layer_name):
        def _hook(module, inputs, output):
            if layer_name in self.thresholds_dict:
                flagged = self.thresholds_dict[layer_name]
                t = output[0] if isinstance(output, tuple) else output
                t[..., flagged] = t[..., flagged] * self.amp_factor
                return (t,) + output[1:] if isinstance(output, tuple) else t
        return _hook
    def attach(self, model):
        for name, module in model.named_modules():
            if name in self.thresholds_dict:
                self.handles.append(module.register_forward_hook(self.hook_fn(name)))
    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []

class ActivationNoiseHook:
    """PC-Guided Contrastive Decoding: Add independent Gaussian noise to flagged anomalous dims."""
    def __init__(self, thresholds_dict, noise_std=2.0):
        self.thresholds_dict = thresholds_dict
        self.noise_std = noise_std
        self.handles = []
    def hook_fn(self, layer_name):
        def _hook(module, inputs, output):
            if layer_name in self.thresholds_dict:
                flagged = self.thresholds_dict[layer_name]
                t = output[0] if isinstance(output, tuple) else output
                noise = torch.randn_like(t[..., flagged]) * self.noise_std
                t[..., flagged] = t[..., flagged] + noise
                return (t,) + output[1:] if isinstance(output, tuple) else t
        return _hook
    def attach(self, model):
        for name, module in model.named_modules():
            if name in self.thresholds_dict:
                self.handles.append(module.register_forward_hook(self.hook_fn(name)))
    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []

def _get_leaf_means(guardrail, D):
    pc = guardrail.pc_prior
    device = next(pc.parameters()).device
    means = torch.zeros(D, device=device)
    for leaf in pc.leaves:
        if leaf.feature_idx < D:
            means[leaf.feature_idx] = leaf.mu.data
    return means

def _get_marginal_log_likelihoods(guardrail, z):
    pc = guardrail.pc_prior
    B, D = z.shape
    lls = torch.full((B, D), -1e9, device=z.device)
    for leaf in pc.leaves:
        i = leaf.feature_idx
        if i >= D: continue
        vals = z[:, i]
        sigma = F.softplus(leaf.log_sigma) + 1e-6
        nu = F.softplus(leaf.log_nu) + 1e-3
        lg = log_gaussian_jit(vals, leaf.mu, sigma)
        ll = log_laplace_jit(vals, leaf.mu, sigma)
        ls = log_student_jit(vals, leaf.mu, sigma, nu)
        w = torch.log_softmax(leaf.logits, dim=0)
        mix = torch.logsumexp(w.unsqueeze(1) + torch.stack([lg, ll, ls]), dim=0)
        lls[:, i] = mix * torch.sigmoid(leaf.gate)
    return lls

def _compute_delta_h(guardrail, delta_z):
    W2 = guardrail.projector[2].weight.data 
    W1 = guardrail.projector[0].weight.data 
    return (delta_z @ W2 @ W1).float() 

def _mic_delta(guardrail, z, leaf_means, tau_percentile):
    mll = _get_marginal_log_likelihoods(guardrail, z)
    tau = torch.quantile(mll, tau_percentile / 100.0, dim=1, keepdim=True)
    mask = mll < tau
    z_imp = z.clone()
    z_imp[mask] = leaf_means.expand_as(z)[mask]
    delta_z = z_imp - z
    if delta_z.norm() < 1e-8:
        return None, z_imp, mask, 0
    return _compute_delta_h(guardrail, delta_z), z_imp, mask, int(mask.sum())

def generate_pcnet_static(guardrail, inputs, args_dict, mode="energy"):
    global_indices = args_dict.get('global_indices', {})
    max_new_tokens = args_dict['max_new_tokens']
    pad_id = _pad_id(guardrail)
    indices = global_indices.get(mode, [])
    if not indices:
        return generate_vanilla(guardrail, inputs, max_new_tokens, pad_id, stop_ids=args_dict.get("stop_ids"))
    top_k = torch.tensor(indices, device=guardrail.device)
    llm = guardrail.llm
    head = getattr(llm, "lm_head", None) or getattr(llm, "embed_out", None)
    if head is None:
        return generate_vanilla(guardrail, inputs, max_new_tokens, pad_id, stop_ids=args_dict.get("stop_ids"))
    orig = head.weight.data[:, top_k].clone()
    head.weight.data[:, top_k] = 0.0
    out = generate_vanilla(guardrail, inputs, max_new_tokens, pad_id, stop_ids=args_dict.get("stop_ids"))
    head.weight.data[:, top_k] = orig
    return out

def generate_pc_adaptive_constrained_decoding(guardrail, inputs, args_dict, temperature=1.0, top_k_cands=3):

    max_new_tokens = args_dict.get('max_new_tokens', 150)
    pad_id = _pad_id(guardrail)
    stop_set = set(args_dict.get("stop_ids") or [pad_id])
    llm = guardrail.llm

    # Use the threshold calibrated in Phase 1 (passed via args_dict)
    threshold = args_dict.get("optimal_threshold", 0.5)
    
    generated = inputs["input_ids"].clone()
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # 1. Base Forward Pass (Get current state and raw logits)
            outputs = llm(input_ids=generated, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]
            h_current = outputs.hidden_states[-1][:, -1, :].float()
            
            # 2. Evaluate Current State Uncertainty (NLL)
            z_current = guardrail.projector(h_current)
            nll_current = -guardrail.pc_prior(z_current.unsqueeze(-1).unsqueeze(-1)).mean().item()
            
            # 3. Calculate Adaptive Alpha (Intervention Strength)
            alpha_t = torch.sigmoid(torch.tensor((nll_current - threshold) / temperature)).item()
            
            # 4. CONDITIONAL LOOKAHEAD (The massive speedup)
            if alpha_t < 0.05:
                # The state is safe. Skip lookahead, trust the LLM.
                best_tok = torch.argmax(logits, dim=-1).unsqueeze(-1)
            else:
                # The state is risky. Execute the k-candidate lookahead penalty.
                log_probs = F.log_softmax(logits, dim=-1)
                topk_logprobs, topk_ids = torch.topk(log_probs, k=top_k_cands, dim=-1)
                
                best_score = -float('inf')
                best_tok = None
                
                for i in range(top_k_cands):
                    cand_tok = topk_ids[:, i].unsqueeze(0)
                    cand_logprob = topk_logprobs[:, i].item()
                    
                    cand_seq = torch.cat([generated, cand_tok], dim=-1)
                    cand_out = llm(input_ids=cand_seq, output_hidden_states=True)
                    h_cand = cand_out.hidden_states[-1][:, -1, :].float()
                    
                    z_cand = guardrail.projector(h_cand)
                    nll_cand = -guardrail.pc_prior(z_cand.unsqueeze(-1).unsqueeze(-1)).mean().item()
                    
                    # Score uses the dynamic alpha_t from the CURRENT state to 
                    # penalize the future states.
                    score = cand_logprob - (alpha_t * nll_cand)
                    
                    if score > best_score:
                        best_score = score
                        best_tok = cand_tok
                        
            # 5. Append winning token
            generated = torch.cat([generated, best_tok], dim=-1)
            
            if best_tok.item() in stop_set:
                break
                
    return generated


def _get_module_by_name(model, name):
    """Helper to fetch a nested module from the LLM dynamically."""
    mod = model
    for part in name.split('.'):
        mod = getattr(mod, part)
    return mod