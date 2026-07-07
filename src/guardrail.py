import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from probabilistic_circuits import *

class LLM_PC_Guardrail(nn.Module):
    def __init__(self, llm_model_name, pc_in_channels=128, pc_depth=3, device="cuda"):
        super().__init__()
        self.device = device
        
        # 1. Setup 4-bit Quantization to save VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        print(f"Loading LLM: {llm_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, 
            quantization_config=bnb_config,
            torch_dtype=torch.float16, # Save VRAM
            device_map=self.device,
            trust_remote_code=True
        )
        
        # FREEZE THE LLM: We are only training the Guardrail
        for param in self.llm.parameters():
            param.requires_grad = False
        self.llm.eval()
            
        # 2. Get the hidden dimension size dynamically
        llm_hidden_size = self.llm.config.hidden_size
        
        # 3. Projection Layer (Compresses 4096 -> 128 for the PCNet)
        self.projector = nn.Sequential(
            nn.Linear(llm_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, pc_in_channels)
        ).to(self.device)
        
        # 4. Initialize Your PCNet Prior
        # Standard, structural PCNet (Much more stable for flat vectors)
        self.pc_prior = PCNet(
            n_classes=1,       # We just want 1 density output
            max_depth=4,       # Standard depth
            max_branching=3, 
            seed=42
        ).to(self.device)
        
        self.pc_in_channels = pc_in_channels

    def init_pc_network(self, sample_texts):
        print("🔨 Initializing PCNet statistics using LLM embeddings...")
        with torch.no_grad():
            embeddings = self._get_llm_embeddings(sample_texts)
            # Reshape for your spatial PCNet: (B, C) -> (B, C, 1, 1)
            embeddings_spatial = embeddings.unsqueeze(-1).unsqueeze(-1)
            self.pc_prior.init_network(embeddings_spatial)
            self.pc_prior.to(self.device)

   
    def get_raw_base_embeddings(self, texts):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=1024 
        ).to(self.device)
        
        # Explicit no_grad saves VRAM since LLM is frozen
        with torch.no_grad():
            outputs = self.llm(**inputs, output_hidden_states=True)
            final_layer_states = outputs.hidden_states[-1]
            
        sequence_lengths = inputs['attention_mask'].sum(dim=1) - 1
        batch_size = final_layer_states.shape[0]
        
        # Extract ONLY the last valid token's embedding
        last_token_embeddings = final_layer_states[torch.arange(batch_size, device=self.device), sequence_lengths]
        
        return last_token_embeddings.float()

    def _get_llm_embeddings(self, texts):
        """
        Helper for the PCNet to get the PROJECTED (128-dim) embeddings.
        """
        # Fetch the raw embeddings, then push them through the projector
        raw_embeddings = self.get_raw_base_embeddings(texts)
        return self.projector(raw_embeddings)
    # =========================================================================

    def forward(self, texts, mpe_training=False, n_samples=1):
        """
        Evaluates the anomaly score (Negative Log-Likelihood) of the given texts.
        If n_samples > 1, applies MC Dropout to calculate Epistemic Uncertainty.
        """
        if self.training:
            embeddings = self._get_llm_embeddings(texts)
            embeddings_spatial = embeddings.unsqueeze(-1).unsqueeze(-1)
            log_prob = self.pc_prior(embeddings_spatial)
            return log_prob # Shape: (B, 1, 1, 1)
        else:
            with torch.no_grad():
                base_embeddings = self._get_llm_embeddings(texts)
                
            if n_samples == 1:
                # Standard deterministic forward pass
                embeddings_spatial = base_embeddings.unsqueeze(-1).unsqueeze(-1)
                log_prob = self.pc_prior(embeddings_spatial)
                return log_prob
            else:
                # --- MC Dropout for Epistemic Uncertainty ---
                mc_log_probs = []
                for _ in range(n_samples):
                    # Force 10% dropout ON even in eval mode to simulate model doubt
                    dropped_embs = torch.nn.functional.dropout(base_embeddings, p=0.1, training=True)
                    emb_spatial = dropped_embs.unsqueeze(-1).unsqueeze(-1)
                    mc_log_probs.append(self.pc_prior(emb_spatial))
                
                # Stack results: (n_samples, Batch, ...)
                stacked_probs = torch.stack(mc_log_probs)
                mean_log_prob = stacked_probs.mean(dim=0)
                variance = stacked_probs.var(dim=0)
                
                # Store the variance so experiment.py can grab it!
                self.pc_prior._last_var = variance
                
                return mean_log_prob