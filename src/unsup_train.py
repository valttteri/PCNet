import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
import gc

from guardrail import LLM_PC_Guardrail
from baselines import LatentMLPProbe
from hf_dataloader import UnifiedDataLoader

# ==========================================
# 1. Memory Management & Data Loading
# ==========================================
def flush_memory():
    """Aggressively clears RAM and VRAM between model loads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def load_hf_training_data(dataset_name, subset=None, split="data", max_samples=500, seed=42):
    """Delegates to UnifiedDataLoader to parse and load the training pairs."""
    return UnifiedDataLoader.load_train_data(
        dataset_name=dataset_name, 
        subset=subset, 
        split=split, 
        max_samples=max_samples, 
        seed=seed
    )


# ==========================================
# 2. Core Training Routine
# ==========================================
def train_guardrail(config):
    device = config.get("device", "cuda")
    llm_name = config.get("llm_model_name", "Qwen/Qwen2.5-0.5B")
    pc_in_channels = config.get("pc_in_channels", 128)
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 8)
    lr = config.get("learning_rate", 1e-3)
    
    pc_save_path = config.get("pc_save_path")
    mlp_save_path = config.get("mlp_save_path")

    # Initialize Models
    print(f"🚀 Initializing Guardrail with {llm_name}...")
    guardrail = LLM_PC_Guardrail(llm_model_name=llm_name, pc_in_channels=pc_in_channels, device=device)

    # Load Data
    true_texts, fake_texts = load_hf_training_data(
        dataset_name=config.get("hf_dataset_name"), 
        subset=config.get("hf_subset_name"), 
        max_samples=config.get("max_train_samples", 1000),
        seed=config.get("seed", 42)
    )

    # PCNet Initialization
    init_batch = true_texts[:min(32, len(true_texts))]
    init_batch = true_texts[:min(32, len(true_texts))]
    guardrail.init_pc_network(init_batch)
    guardrail.pc_prior.to(device)

    # Optimizers
    pc_optimizer = torch.optim.Adam(
        list(guardrail.pc_prior.parameters()) + list(guardrail.projector.parameters()), 
        lr=lr, 
        weight_decay=1e-5
    )

    # Hyperparameters
    static_margin = config.get("margin", 5.0)
    alpha = config.get("alpha", 0.8) # Weight for factual NLL vs margin penalty
    best_pc_loss = float('inf')

    print(f"\n🔥 Starting Static Contrastive Training (Epochs: {epochs}, Margin: {static_margin}, Alpha: {alpha})")
    
    for epoch in range(epochs):
        random.shuffle(true_texts)
        random.shuffle(fake_texts)
        
        epoch_pos_loss = 0
        epoch_neg_loss = 0
        epoch_mlp_loss = 0
        batches = 0
        
        max_batches = min(len(true_texts), len(fake_texts))
        
        guardrail.train()
        
        for i in tqdm(range(0, max_batches, batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch_true = true_texts[i:i+batch_size]
            
            # ------------------------------------------------
            # 1. Train PCNet (Generative + Contrastive Margin)
            # ------------------------------------------------
            pc_optimizer.zero_grad()
            
            log_prob_true = guardrail(batch_true, mpe_training=False)
            loss_pos = -log_prob_true.mean()
            
            pc_total_loss = loss_pos
            pc_total_loss.backward()
            
            # Gradient clipping to prevent exploding variance
            torch.nn.utils.clip_grad_norm_(guardrail.pc_prior.parameters(), max_norm=1.0)
            pc_optimizer.step()
            batches += 1
            epoch_pos_loss+= loss_pos.item()
        # Epoch Summary
        avg_pos = epoch_pos_loss / batches
        avg_neg = epoch_neg_loss / batches
        avg_mlp = epoch_mlp_loss / batches
        avg_total = (alpha * avg_pos) + ((1 - alpha) * avg_neg)

        print(f"📊 Epoch {epoch+1} | True NLL: {avg_pos:.4f} | Margin Penalty: {avg_neg:.4f} | Total PC Loss: {avg_total:.4f} | MLP BCE: {avg_mlp:.4f}")

        # Save Best Model Logic (Saving ONLY trainable components)
        # Save Best Model Logic
        if avg_total < best_pc_loss:
            best_pc_loss = avg_total
            print(f"  🌟 New best model found! Saving full instances to {pc_save_path}...")
            
            # Save the actual objects in their entirety
            torch.save({
                'pc_prior': guardrail.pc_prior, 
                'projector': guardrail.projector
            }, pc_save_path)


# ==========================================
# 3. Dynamic Multi-Model Looping
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Model Multi-Dataset Training")
    parser.add_argument("config", help="Path of the JSON config file to use")
    parser.add_argument("seed", type=int, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = json.load(f)
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    llm_models = config.get("llm_models", ["Qwen/Qwen2.5-0.5B"])
    datasets = config.get("datasets", [{"name": "pminervini/HaluEval", "subset": "qa", "split": "data"}])
    algorithm = config.get("algorithm", "AttentionalPCNet_Guardrail")

    print(f"🚀 Starting Massive Training Suite | Models: {len(llm_models)} | Datasets: {len(datasets)}")

    for llm_name in llm_models:
        print(f"\n{'='*60}\n🤖 TRAINING FOUNDATION MODEL: {llm_name}\n{'='*60}")
        
        current_config = config.copy()
        current_config["llm_model_name"] = llm_name
        current_config["seed"] = args.seed
        
        for ds_info in datasets:
            ds_name = ds_info["name"]
            ds_subset = ds_info.get("subset", "qa")
            
            print(f"\n📚 TRAINING ON DATASET: {ds_name} (Subset: {ds_subset})")
            
            # Setup dynamic saving paths
            safe_llm_str = llm_name.replace("/", "_")
            safe_ds_str = ds_name.replace("/", "_")
            weight_dir = os.path.join("checkpoints", algorithm, str(args.seed), safe_llm_str, safe_ds_str)
            os.makedirs(weight_dir, exist_ok=True)
            pc_path = os.path.join(weight_dir, "pcnet_best_unsup.pth")
            # --- THE SKIP LOGIC ---
            if os.path.exists(pc_path):
                print(f"  ⏭️ Trained weights already exist at {weight_dir}. Skipping training!")
                continue # Jumps to the next dataset/model without training

            current_config["hf_dataset_name"] = ds_name
            current_config["hf_subset_name"] = ds_subset
            current_config["pc_save_path"] = os.path.join(weight_dir, "pcnet_best_unsup.pth")
            
            # Run the training loop!
            train_guardrail(current_config)
            
        print(f"\n🧹 Cleaning up {llm_name} from memory...")
        flush_memory()

    print("\n🎉 ALL TRAINING COMPLETE.")

if __name__ == "__main__":
    main()