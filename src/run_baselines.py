import os
import argparse
import json
import random
import numpy as np
import torch
import gc
import importlib

from transformers import AutoModelForCausalLM, AutoTokenizer
from experiment import BaselineDetectionExperiment

import hf_dataloader

def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Baseline Only Benchmark")
    parser.add_argument("config", help="Path of the JSON config file to use")
    parser.add_argument("seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = json.load(f)
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = config.get("device", "cuda")
    max_train_samples = config.get("max_train_samples", 150)
    max_samples = config.get("max_samples", 100)
    batch_size = config.get("batch_size", 4)
    
    llm_models = config.get("llm_models", ["Qwen/Qwen2.5-0.5B"])
    datasets = config.get("datasets", [{"name": "pminervini/HaluEval", "subset": "qa"}])

    master_metrics = {}

    print(f"🚀 Starting Token & SOTA Baselines Benchmark on {device.upper()}")
    print(f"Models: {len(llm_models)} | Datasets: {len(datasets)}")
    print("=" * 60)

    for llm_name in llm_models:
        print(f"\n\n{'='*60}")
        print(f"🤖 LOADING FOUNDATION MODEL: {llm_name}")
        print(f"{'='*60}")
        
        master_metrics[llm_name] = {}
        
        # Load Plain LLM and Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        llm = AutoModelForCausalLM.from_pretrained(
            llm_name, 
            device_map=device, 
            torch_dtype=torch.float16 if device=="cuda" else torch.float32
        )
        llm.eval()
        
        for ds_info in datasets:
            ds_name = ds_info["name"]
            ds_subset = ds_info.get("subset", None)
            ds_split = ds_info.get("split", "data")
            ds_key = f"{ds_name}_{ds_subset}" if ds_subset else ds_name
            
            print(f"\n📊 EVALUATING ON DATASET: {ds_name} (Subset: {ds_subset})")
            
            importlib.reload(hf_dataloader)
            UnifiedDataLoader = hf_dataloader.UnifiedDataLoader
            
            safe_llm_str = llm_name.replace("/", "_")
            safe_ds_str = ds_name.replace("/", "_")
            
            log_dir = os.path.join("logs", "baselines", safe_llm_str, safe_ds_str, str(args.seed))
            metrics_path = os.path.join(log_dir, "metrics.json")
            
            if os.path.exists(metrics_path):
                print(f"  ⏭️ Test logs already exist at {metrics_path}. Skipping evaluation!")
                with open(metrics_path, "r") as f:
                    master_metrics[llm_name][ds_key] = json.load(f)
                continue 

            dataset = UnifiedDataLoader.load_test_data(
                dataset_name=ds_name, 
                subset=ds_subset, 
                split=ds_split, 
                max_samples=max_samples,
                seed=args.seed
            )
            
            if not dataset:
                print("  ⚠️ Dataset loading failed or empty. Skipping...")
                continue
            
            experiment = BaselineDetectionExperiment(llm=llm, tokenizer=tokenizer, device=device)
            
            # Fetch Wild Mixture for HaloScope Fit
            print("  🧩 Fetching RAW 'Wild Mixture' Embeddings for HaloScope Pipeline...")
            train_true, train_fake = UnifiedDataLoader.load_train_data(
                dataset_name=ds_name,
                subset=ds_subset,
                split=ds_split,
                max_samples=max_train_samples, 
                seed=args.seed
            )
            
            num_true = min(len(train_true), int(max_train_samples * 0.85))
            num_fake = min(len(train_fake), int(max_train_samples * 0.15))
            wild_mixture = train_true[:num_true] + train_fake[:num_fake]
            
            if wild_mixture:
                raw_train_embeddings = []
                with torch.no_grad():
                    for i in range(0, len(wild_mixture), batch_size):
                        batch = wild_mixture[i:i+batch_size]
                        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
                        outputs = llm(**inputs, output_hidden_states=True)
                        hidden_states = outputs.hidden_states[-1]
                        seq_lens = (inputs['input_ids'] != tokenizer.pad_token_id).sum(dim=1) - 1
                        raw_emb = hidden_states[torch.arange(hidden_states.size(0), device=device), seq_lens, :]
                        raw_train_embeddings.append(raw_emb.detach())

                if raw_train_embeddings:
                    raw_train_embeddings = torch.cat(raw_train_embeddings, dim=0)
                    experiment.fit_haloscope(raw_train_embeddings)

                    # SEP: fit linear probe (hidden state -> semantic entropy).
                    # Cap training size to keep SE computation tractable.
                    n_sep = config.get("sep_train_samples", min(60, len(wild_mixture)))
                    sep_texts = wild_mixture[:n_sep]
                    sep_embs = raw_train_embeddings[:n_sep]
                    print(f"  🧪 Fitting SEP on {n_sep} samples (computing semantic-entropy targets)...")
                    se_targets = experiment.compute_semantic_entropy_for_fit(
                        sep_texts,
                        num_samples=config.get("sep_se_samples", 5),
                        max_new_tokens=config.get("sep_se_max_new_tokens", 40),
                    )
                    experiment.fit_sep(sep_embs, se_targets)
            
            metrics = experiment.run_benchmark(dataset, dataset_name=ds_key, llm_name=llm_name)
            master_metrics[llm_name][ds_key] = metrics
            
            os.makedirs(log_dir, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
                
            if 'dataset' in locals(): del dataset
            if 'train_true' in locals(): del train_true
            if 'train_fake' in locals(): del train_fake
            if 'wild_mixture' in locals(): del wild_mixture
            if 'raw_train_embeddings' in locals(): del raw_train_embeddings
            if 'experiment' in locals(): del experiment
            flush_memory()
            
        print(f"\n🧹 Cleaning up {llm_name} Foundation Model from memory...")
        del llm
        del tokenizer
        flush_memory()

    os.makedirs("logs", exist_ok=True)
    summary_path = os.path.join("logs", "baselines_master_benchmark_summary.json")
    
    with open(summary_path, "w") as f:
        json.dump(master_metrics, f, indent=4)
        
    print(f"\n🎉 ALL BASELINE BENCHMARKS COMPLETE. Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
