import os
import argparse
import json
import random
import numpy as np
import torch
import gc
import importlib
import pandas as pd

from guardrail import LLM_PC_Guardrail
from experiment import HallucinationExperiment
from baselines import LatentMLPProbe

# We import the dataloader module itself so we can forcefully reload it later
import hf_dataloader

from data_tools import load_custom_dataset
from logger import Logger
logs = Logger()

BINARY_TREE_PROMPT = "binary_tree"
LABELING_PROBLEM_PROMPT = "labeling_problem"

BINARY_LABEL = "binary_label"

# ==========================================
# 1. Memory Management
# ==========================================
def flush_memory():
    """Aggressively clears RAM and VRAM between model loads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ==========================================
# 2. Main Evaluation Loop
# ==========================================
def main(
    use_custom_data:bool=False,
    custom_train_data_path:str="",
    custom_eval_data_path:str="",
    weights_str:str="",
    dataset_str:str="",
    label_type:str="",
    col_names:dict={},
    metrics_id:int=1
):
    parser = argparse.ArgumentParser(description="Multi-Model Multi-Dataset Benchmark")
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
    algorithm = config.get("algorithm", "PCNet_Guardrail")
    pc_in_channels = config.get("pc_in_channels", 128)
    max_train_samples = config.get("max_train_samples", 500)
    #max_samples = config.get("max_samples", None)

    # Limit max saples to 1000 to avoid computational overhead
    # Delete: changed max_samples
    max_samples = 1000
    max_train_samples = 1000
    
    batch_size = config.get("batch_size", 8)
    
    # --- UNSUPERVISED FLAG ---
    is_unsup = config.get("unsupervised", False)
    
    llm_models = config.get("llm_models", ["Qwen/Qwen2.5-0.5B"])
    datasets = config.get("datasets", [{"name": "pminervini/HaluEval", "subset": "qa"}])

    # Alternative model name for paths
    llm_path = config.get("model_path")

    # Change/delete: run with only specific datasets.
    datasets = [
        {
            "name": "trivia_qa",
            "subset": "rc.nocontext"
        }
    ]

    # Original
    #datasets = [
    #    {
    #        "name": "coqa",
    #        "subset": None
    #    },
    #    {
    #        "name": "truthful_qa",
    #        "subset": "generation"
    #    },
    #    {
    #        "name": "trivia_qa",
    #        "subset": "rc.nocontext"
    #    },
    #    {
    #        "name": "rajpurkar/squad_v2",
    #        "subset": None,
    #        "split": "test"
    #    }
    #]

    master_metrics = {}

    print(f"[main.py] Starting Massive Benchmark Suite on {device.upper()}")
    print(f"⚙️  Mode: {'UNSUPERVISED' if is_unsup else 'SUPERVISED'}")
    print(f"Models: {len(llm_models)} | Datasets: {len(datasets)}")
    print("=" * 60)

    for llm_name in llm_models:
        print(f"\n\n{'='*60}")
        print(f"🤖 LOADING FOUNDATION MODEL: {llm_name}")
        print(f"{'='*60}")
        
        master_metrics[llm_name] = {}
        
        # 1. Initialize Base Guardrail (Loaded ONCE per LLM to save VRAM/Time)
        guardrail = LLM_PC_Guardrail(llm_model_name=llm_name, pc_in_channels=pc_in_channels, device=device)
        
        for ds_info in datasets:
            ds_name = ds_info["name"]
            ds_subset = ds_info.get("subset", None)
            ds_split = ds_info.get("split", "data")
            ds_key = f"{ds_name}_{ds_subset}" if ds_subset else ds_name
            
            print(f"\n📊 EVALUATING ON DATASET: {ds_name} (Subset: {ds_subset})")
            
            importlib.reload(hf_dataloader)
            UnifiedDataLoader = hf_dataloader.UnifiedDataLoader
            
            # 2. Re-initialize MLP Probe per dataset so weights don't carry over
            mlp_probe = LatentMLPProbe(input_dim=pc_in_channels).to(device)

            # --- String Formatting for Paths ---
            safe_llm_str = llm_name.replace("/", "_")
            safe_ds_str = ds_name.replace("/", "_")
            
            # Check if test logs already exist to skip run
            log_algo_folder = f"unsup_{algorithm}" if is_unsup else algorithm
            #log_dir = os.path.join("logs", log_algo_folder, str(args.seed), safe_llm_str, safe_ds_str)

            # Change/delete: The folder where metrics are saved 
            log_dir = os.path.join(f"all_logs/pcnet_14k/{weights_str}_weights", log_algo_folder, str(args.seed), safe_llm_str, dataset_str)
            
            # Test folder
            #log_dir = os.path.join(f"all_logs/pcnet_detection_logs_ztest", log_algo_folder, str(args.seed), safe_llm_str, safe_ds_str)

            metrics_path = os.path.join(log_dir, f"metrics{metrics_id}.json")

            print("[main.py/main()] Metrics path:", metrics_path)
            print("Path exists", os.path.exists(metrics_path))
            
            if os.path.exists(metrics_path):
                print(f"  ⏭️ Test logs already exist at {metrics_path}. Skipping evaluation!")
                with open(metrics_path, "r") as f:
                    master_metrics[llm_name][ds_key] = json.load(f)
                continue 

            # --- Dynamic Weight Loading ---

            # Change/delete: use same PCNet weights for each dataset
            #safe_ds_str = "truthful_qa"
            #safe_ds_str = "rajpurkar_squad_v2"
            #safe_ds_str = "trivia_qa"
            #safe_ds_str = "coqa"

            # Change/delete: custom weight_dir
            algorithm = "PCNet_Guardrail_14k"
            weight_dir = os.path.join("checkpoints", algorithm, str(args.seed), safe_llm_str, weights_str)
            
            pc_weight_filename = "pcnet_best_unsup.pth" if is_unsup else "pcnet_best.pth"
            pc_weight_path = os.path.join(weight_dir, pc_weight_filename)
            mlp_weight_path = os.path.join(weight_dir, "mlp_probe_best.pth") 
            
            if os.path.exists(pc_weight_path):
                print(f"Loading PCNet weights from {pc_weight_path}")
                checkpoint = torch.load(pc_weight_path, map_location=device, weights_only=False)
                guardrail.pc_prior = checkpoint['pc_prior'].to(device)
                guardrail.projector = checkpoint['projector'].to(device)
            else:
                print(f"  ⚠️ No trained PCNet weights found at {pc_weight_path}. Benchmark will fail.")
                continue 
                
            if os.path.exists(mlp_weight_path):
                print(f"  📦 Loading trained SOTA MLP weights from {mlp_weight_path}")
                mlp_probe.load_state_dict(torch.load(mlp_weight_path, map_location=device))
            else:
                print(f"  ⚠️ No trained MLP weights found. MLP baseline will be random.")

            # --- Load Data & Run ---
            if use_custom_data:
                dataset = load_custom_dataset(
                    dataset_path=custom_eval_data_path,
                    max_samples=max_samples,
                    label_type=label_type,
                    col_names=col_names,
                    train=False,
                    seed=args.seed
                )
            else:
                dataset = UnifiedDataLoader.load_test_data(
                    dataset_name=ds_name, 
                    subset=ds_subset, 
                    split=ds_split, 
                    max_samples=max_samples,
                    seed=args.seed
                )
            
            logs.info(f"Testing data size: {len(dataset)}")
            if not dataset:
                print("  ⚠️ Dataset loading failed or empty. Skipping...")
                continue

            experiment = HallucinationExperiment(guardrail_model=guardrail, mlp_probe=mlp_probe, device=device)
            
            # =========================================================
            # Fit HaloScope Latent Baseline using RAW Mixed Embeddings
            # =========================================================
            print("  🧩 Fetching RAW 'Wild Mixture' Embeddings for HaloScope Pipeline...")
            
            # Delete/change: training data for calibration
            # Load BOTH true and fake data to create the variance mixture
            if use_custom_data:
                train_true, train_fake = load_custom_dataset(
                    dataset_path=custom_train_data_path,
                    max_samples=max_train_samples,
                    label_type=label_type,
                    col_names=col_names,
                    train=True,
                    seed=args.seed
                )
            else:
                train_true, train_fake = UnifiedDataLoader.load_train_data(
                    dataset_name=ds_name,
                    subset=ds_subset,
                    split=ds_split,
                    max_samples=max_train_samples, 
                    seed=args.seed
                )
            logs.info(f"Training data: train_true, train_fake entries {len(train_true)},{len(train_fake)}")

            # Combine into a "Wild" dataset (simulating 85% true, 15% fake in the wild)
            num_true = min(len(train_true), int(max_train_samples * 0.85))
            num_fake = min(len(train_fake), int(max_train_samples * 0.15))
            wild_mixture = train_true[:num_true] + train_fake[:num_fake]
            
            if wild_mixture:
                raw_train_embeddings = []
                for i in range(0, len(wild_mixture), batch_size):
                    batch = wild_mixture[i:i+batch_size]

                    # MUST use RAW LLM hidden states, bypassing PCNet projector
                    raw_emb = guardrail.get_raw_base_embeddings(batch)
                    raw_train_embeddings.append(raw_emb)

                if raw_train_embeddings:
                    raw_train_embeddings = torch.cat(raw_train_embeddings, dim=0)
                    # Delete: skip fitting haloscope
                    #experiment.fit_haloscope(raw_train_embeddings)

                    # SEP: fit linear probe (hidden state -> semantic entropy).
                    n_sep = config.get("sep_train_samples", min(60, len(wild_mixture)))
                    sep_texts = wild_mixture[:n_sep]
                    sep_embs = raw_train_embeddings[:n_sep]
                    print(f"  🧪 Fitting SEP on {n_sep} samples (computing semantic-entropy targets)...")
                    # Delete: uncomment se_targets
                    #se_targets = experiment.compute_semantic_entropy_for_fit(
                    #    sep_texts,
                    #    num_samples=config.get("sep_se_samples", 5),
                    #    max_new_tokens=config.get("sep_se_max_new_tokens", 40),
                    #)
                    #experiment.fit_sep(sep_embs, se_targets)
            
            metrics = experiment.run_benchmark(dataset, dataset_name=ds_key, llm_name=llm_name)
            master_metrics[llm_name][ds_key] = metrics
            
            # Change/delete: commented out result saving code
            os.makedirs(log_dir, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
                
            if 'dataset' in locals(): del dataset
            if 'train_true' in locals(): del train_true
            if 'train_fake' in locals(): del train_fake
            if 'train_embeddings' in locals(): del train_embeddings
            if 'experiment' in locals(): del experiment
            if 'mlp_probe' in locals(): del mlp_probe
            flush_memory()
            
        # End of Datasets Loop
        print(f"\n🧹 Cleaning up {llm_name} Foundation Model from memory...")
        del guardrail
        flush_memory()

    # Final Save
    os.makedirs("logs", exist_ok=True)
    summary_filename = "unsup_master_benchmark_summary.json" if is_unsup else "master_benchmark_summary.json"
    summary_path = os.path.join("all_logs/pcnet_detection_logs_z", summary_filename)

    try:
        with open(summary_path, "w") as f:
            json.dump(master_metrics, f, indent=4)
    except FileNotFoundError:
        logs.error(f"Master summary path not found. (Does not matter)")
        
    print(f"\n🎉 ALL BENCHMARKS COMPLETE. Summary saved to {summary_path}")

if __name__ == "__main__":
    # Names of different PCNet weights
    gemma_2_9b_it = "gemma-2-9b-it"
    llama_3_8b = "llama-3-8b"
    mistral_7b_v03 = "mistral-7b-v0.3"
    
    # Which model we evaluate
    model_name = mistral_7b_v03
    # Which weights we use
    weights = "truthfulqa"
    # Datasets for evaluation
    test_data = ["truthfulqa", "hotpotqa", "triviaqa", "combined"]

    col_names = {"q_col": "question", "a_col": "answer_text", "lbl_col": "intermediate_p2"}

    for ds in test_data:
        main(
            use_custom_data=True,
            custom_train_data_path=f"datasets/hallu_corpus_without_activations_v1/data_samples/{model_name}/{ds}_train.csv",
            custom_eval_data_path=f"datasets/hallu_corpus_without_activations_v1/data_samples/{model_name}/{ds}_test.csv",
            weights_str=weights,
            dataset_str=ds,
            label_type=BINARY_LABEL,
            col_names=col_names,
            metrics_id=1
        )

    """
    main(
        use_custom_data=False,
        custom_train_data_path="",
        custom_eval_data_path="",
        weights_str="coqa",
        dataset_str="",
        label_type=BINARY_LABEL,
        col_names="",
        metrics_id=1
    )
    """

#srun python src/main.py src/config/sup/llama3_1_8b.json 42 
    """
    dataset_path:str,
    max_samples:int,
    label_type:str,
    col_names:dict,
    train:bool=False,
    seed:int=42,
    """