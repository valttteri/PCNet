import os
import json
import boto3
import torch
import ast
import random
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from transformers import BitsAndBytesConfig
from huggingface_hub import model_info, InferenceClient
from datasets import load_dataset
from datetime import datetime
from dotenv import load_dotenv
from tabulate import tabulate

from logger import Logger

logs = Logger()

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

BINARY_TREE_PROMPT = "binary_tree"
LABELING_PROBLEM_PROMPT = "labeling_problem"
V4_HALLU_AUDIT_SHORT_PROMPT = "v4_hallu_audit_short"

def annotation_label_metrics(dataset_path, col1_name, col2_name):
    # Compute raw agreement and Cohen's kappa score between two annotation lable columns
    logs.info("Computing metrics")
    df = pd.read_csv(dataset_path)

    col1_labels, col2_labels = [], []

    raw_matches, pos_neg_matches, undefined = 0, 0, 0
    for i in range(len(df)):
        if col1_name == "human_annotation":
            col1_label = df.loc[i, col1_name]
        else:
            col1_dict = ast.literal_eval(df.loc[i, col1_name])
            col1_label = col1_dict["label"]
            
        if col2_name == "human_annotation":
            col2_label = df.loc[i, col2_name]
        else:
            col2_dict = ast.literal_eval(df.loc[i, col2_name])
            col2_label = col2_dict["label"]



        if col1_label == "UD":
            undefined += 1
            continue

        # Matching labels
        if col1_label == col2_label:
            raw_matches += 1

        # Both labels are either pos or neg
        if col1_label[:3] == col2_label[:3]:
            pos_neg_matches += 1

        col1_labels.append(col1_label)
        col2_labels.append(col2_label)

    cohen_kappa = cohen_kappa_score(col1_labels, col2_labels)
    logs.info(f"Raw agreement between {col1_name}, {col2_name}: {raw_matches}/{len(df)-undefined}")
    logs.info(f"Pos/neg matches between {col1_name}, {col2_name}: {pos_neg_matches}/{len(df)-undefined}")
    logs.info(f"Cohen's Kappa score: {cohen_kappa}")



def create_data_subset(
    dataset_path:str,
    subset_size:int,
    output_path:str,
    columns=None,
    format_answer_col=False
):
    # Local csv file
    if dataset_path.endswith(".csv"):
        dataset = pd.read_csv(dataset_path)
    else:
        # Load dataset from huggingface
        if "trivia_qa" in dataset_path:
            dataset = load_dataset(dataset_path, "rc.nocontext", split=f"validation[:{sample_size}]")
        dataset = dataset.to_pandas()

    sample = dataset.sample(n=subset_size)
    sample = sample[columns]

    # Format the answer column (trivia_qa)
    if format_answer_col:
        sample["answer"] = sample["answer"].apply(
            lambda s: eval(
                s,
                {"__builtins__": {}},
                {"array": np.array, "object": object}
            )["aliases"].tolist()
        )

    sample.to_csv(output_path, index=False)
    logs.info(f"Saved a data sample of shape {sample.shape} to {output_path}")

def create_data_subset_14k():
    output_path = "datasets/hallu_corpus_without_activations_v1"

def format_answer_column(dataset_path:str):
    df = pd.read_csv(dataset_path)

    df["answer"] = df["answer"].apply(
        lambda s: eval(
            s,
            {"__builtins__": {}},
            {"array": np.array, "object": object}
        )["aliases"].tolist()
    )

    #print(df["answer"][0])

    df.to_csv(dataset_path, index=False)

def count_column_values(
    dataset_path:str,
    col_name:str
):
    """
    Count how many times labels 0 and 1 were given (Prompt P2)
    """
    df = pd.read_csv(dataset_path)

    zero_labels = df[df[col_name] == 0]
    one_labels = df[df[col_name] == 1]
    #print(len(zero_labels))
    #print(len(one_labels))

    print(df[col_name].value_counts())

def format_trivia_qa(df):
    """
    Function for preprocessing a triviaqa data sample

    Pipeline:
    1. Format answer-column from a weird dict to an array
    2. Remove unnecessary columns
    """
    triviaqa_cols = [
        "question",
        "question_id",
        "question_source",
        "answer",
    ]
    
    # Step 1.
    df["answer"] = df["answer"].apply(lambda x: x["value"])

    # Step 2.
    df = df[triviaqa_cols]
    logs.info("(data_tools.py) Formatted trivia_qa sample")

    return df

def format_14k():
    dataset_dir = "datasets/hallu_corpus_without_activations_v1/"
    output_path = os.path.join(dataset_dir, "data_samples")

    df = pd.read_parquet(
        f"{dataset_dir}/corpus_without_activations.parquet"
    )
    
    models = [
        "gemma-2-9b-it",
        "llama-3-8b",
        "mistral-7b-v0.3",
    ]
    datasets = ["hotpotqa", "triviaqa", "truthfulqa"]

    print(f"\nData with only has_eos == True")
    eos_tokens = df[df["has_eos"] == True].copy()

    for m in models:
        llm_subset = eos_tokens[eos_tokens["model"] == m].copy()
        train_sets = []
        test_sets = []

        for ds in datasets:
            data_subset = llm_subset[llm_subset["dataset"] == ds]
            data_subset_name = f"{m}_{ds}_{len(data_subset)}"
            print(f"model={m}, data={ds}, size={data_subset.shape}")


            hallu = data_subset[data_subset["intermediate_p2"] == 1]
            not_hallu = data_subset[data_subset["intermediate_p2"] == 0]

            train_ds, test_ds = train_test_split(
                data_subset,
                test_size=0.2,
                random_state=42,
                stratify=data_subset["intermediate_p2"],  # replace "label" with your hallu/not_hallu column
            )


            train_sets.append(train_ds)
            test_sets.append(test_ds)

            #Save per-dataset train/test splits
            train_ds.to_csv(f"{output_path}/{m}/{ds}_train.csv", index=False)
            test_ds.to_csv(f"{output_path}/{m}/{ds}_test.csv", index=False)

            print(f"{m}, {ds} train len={len(train_ds)}, test len={len(test_ds)}")

        # Combine train portions across the three datasets, then shuffle
        model_train = pd.concat(train_sets, ignore_index=True)
        model_train = model_train.sample(frac=1, random_state=42).reset_index(drop=True)

        # Combine test portions across the three datasets, then shuffle
        model_test = pd.concat(test_sets, ignore_index=True)
        model_test = model_test.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"model={m}, train size={model_train.shape}, test size={model_test.shape}")

        model_train.to_csv(f"{output_path}/{m}/combined_train.csv", index=False)
        model_test.to_csv(f"{output_path}/{m}/combined_test.csv", index=False)

    print(f"14k data subsets done.")

def get_model_and_tokenizer_kwargs(model_name:str):
    # Tokenizer kwargs
    tokenizer_kwargs = {"trust_remote_code": True}
    device = "cuda"

    # Model kwargs
    if "Hypernova" in model_name:
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
            "trust_remote_code": True,
        }
    elif "FP8" in model_name:
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": device,
            "trust_remote_code": True,
        }
    else:
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": device,
            "trust_remote_code": True,
        }

    # Chat template kwargs
    if model_name == "Qwen/Qwen3.6-35B-A3B":
        chat_template_kwargs = {
            "tokenize": True,
            "return_tensors": "pt",
            "return_dict": True,          
            "add_generation_prompt": True,
            "enable_thinking": False 
        }
    elif model_name == "IQuestLab/IQuest-Coder-V1-40B-Instruct":
        chat_template_kwargs = {
            "tokenize": True,
            "return_tensors": "pt",
            "return_dict": True,          
            "add_generation_prompt": True,
            "enable_thinking": False 
        }
    else:
        chat_template_kwargs = {
            "tokenize": True,
            "return_tensors": "pt",
            "return_dict": True,          
            "add_generation_prompt": True 
        }

    # Models that are already quantized do not need the BNB_CONFIG
    if not any(x in model_name for x in ["nvidia", "FP8"]): 
        model_kwargs["quantization_config"] = BNB_CONFIG

    return model_kwargs, chat_template_kwargs, tokenizer_kwargs

def get_model_generation_kwargs(
    model_name:str=None,
    tokenizer=None,
    annotate:bool=False
):
    known_models = [
        "Qwen/Qwen3.6-35B-A3B",
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        "MultiverseComputingCAI/Hypernova-60B-2605",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "google/gemma-4-31B-it",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
        "google/gemma-4-26B-A4B-it"
    ]
        
    if model_name not in known_models:
        logs.error(f"{model_name} is unknown.")
        return None
    
    max_new_tokens = 150
    if annotate:
        # Annotation might require more tokens than model answer generation
        max_new_tokens = 1000

    if "gemma" in model_name:
        return {
            "max_new_tokens": max_new_tokens, # For elaborations in annotation
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
        }

    if model_name == "Qwen/Qwen3.6-35B-A3B":
        # Based on instruct mode params in huggingface
        return {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "pad_token_id": tokenizer.eos_token_id
        }
    
    if model_name == "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4":
        return {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "pad_token_id": tokenizer.eos_token_id
        }
    
    # Default params for models not mentioned above
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id # Comment out for gemma models
    }

def generate_bookkeeping(
    model_name:str,
    output_path:str,
    sample_size:int,
    identifier:int | str,
):
    m_info = model_info(model_name)
    commit_id = m_info.sha

    date_today = datetime.now()
    formatted_date = date_today.strftime("%d/%m/%Y")

    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "<question>"},
        {"role": "assistant", "content": ""}
    ]

    dataset_info = {
        "name": "trivia_qa",
        "subset": "rc.nocontext",
        "split": f"validation"
    }

    log_entry = {
        "model_name": model_name,
        "model_sha": commit_id,
        "dataset": dataset_info,
        "sample_size": sample_size,
        "date": formatted_date,
        "source_code": "generate_with_models.py",
        "prompt": prompt
    }

    with open(f"{output_path}/data{identifier}_size{sample_size}_log.json", "w") as f:
        json.dump(log_entry, f)

def concat_dfs():
    df1 = pd.read_csv("datasets/sanity_check_full/binarytree_sanitycheck.csv")
    df2 = pd.read_csv("datasets/sanity_check_full/binarytree_sanity_check_missing_labels_v5_annotated.csv")

    df3 = pd.concat([df1, df2])

    df3.to_csv("datasets/sanity_check_full/binarytree_sanitycheck2.csv", index=False)

def load_custom_dataset(
    dataset_path:str,
    max_samples:int,
    prompt_type:str,
    train:bool=False,
    seed:int=42,
):
    """
    Load a custom dataset for training/evaluating PCNet instances.
    The current implementation is hard coded for triviaqa data, and follows the authors' method (hf_dataloader.py)
    """
    random.seed(seed)
    df = pd.read_csv(dataset_path)

    if prompt_type == LABELING_PROBLEM_PROMPT:
        truthful_subset = df[df["gpt-5.4_labeling_problem_p2"] == 0].reset_index(drop=True)
        hallu_subset = df[df["gpt-5.4_labeling_problem_p2"] == 1].reset_index(drop=True)
    
    # Limit the sample size
    limit = min(max_samples, min(len(truthful_subset), len(hallu_subset)))

    truthful_entries = []
    hallu_entries = []

    for i in range(limit):
        truthful_question = truthful_subset.loc[i, "question"]
        truthful_answer = truthful_subset.loc[i, "gen_answer_meta-llama_Llama-3.1-8B-Instruct"]
        
        hallu_question = hallu_subset.loc[i, "question"]
        hallu_answer = hallu_subset.loc[i, "gen_answer_meta-llama_Llama-3.1-8B-Instruct"]

        truthful_entries.append(f"Question: {truthful_question}\nAnswer: {truthful_answer}")
        hallu_entries.append(f"Question: {hallu_question}\nAnswer: {hallu_answer}")
    
    # Training condition: Return only QA-pairs
    if train:
        logs.info(
            f"(data_tools.py) Returning training data, example:\nTruthful entry: {truthful_entries[0]}\nHallu entry: {hallu_entries[0]}"
        )
        return truthful_entries, hallu_entries

    # Test condition: return QA-pair+label tuples
    combined = [(text, 0) for text in truthful_entries] + [(text, 1) for text in hallu_entries]
    random.shuffle(combined)
    logs.info(
        f"(data_tools.py) Returning testing data, example:\nEntry: {combined[0]}"
    )
    return combined

def data_test():
    df = pd.read_parquet(
        "datasets/hallu_corpus_without_activations_v1/corpus_without_activations.parquet"
    )

    print(df.shape)
    print(df["model"].value_counts())
    print(df["dataset"].value_counts())

    print(tabulate(df[:5], headers="keys", tablefmt="github"))

if __name__ == "__main__":
    triviaqa_cols = [
        "question",
        "question_id",
        "question_source",
        "answer",
        "gen_answer"
    ]

    # Column names in csv file
    human = "human_annotation"
    gemma4_31b_prompt3 = "google/gemma-4-31B-it_binary_tree3_prompt3"
    gemma4_26b_prompt3 = "google/gemma-4-26B-A4B-it_binary_tree3_prompt3"
    qwen3_next_80b_prompt3 = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8_binary_tree3_prompt3"
    gpt_6_astra = "gpt-6-astra_binary_tree3_prompt3"
    gpt_5p6_sol_p5 = "gpt-5.6-sol_binary_tree3_prompt5"
    gpt_5p6_sol_p4 = "gpt-5.6-sol_binary_tree3_prompt4"

    #data_test()

    #format_answer_column(dataset_path="datasets/triviaqa_2/data.csv")
    #create_data_subset(
    #    dataset_path="datasets/triviaqa_2/data.csv",
    #    subset_size=50,
    #    output_path="datasets/triviaqa_2/random_sample50.csv",
    #    columns=triviaqa_cols,
    #    format_answer_col=False
    #)

    #annotation_label_metrics(
    #    dataset_path="datasets/sanity_check/data1_size70_annotated.csv",
    #    col1_name=human,
    #    col2_name=gpt_5p6_sol_p5
    #)

    #count_column_values(
    #    dataset_path="datasets/triviaqa_filtered_samples/data5_size1000_annotated.csv",
    #    col_name="gpt-5.4_labeling_problem_p2"
    #)

    format_14k()

    #load_custom_dataset(
    #    dataset_path="datasets/truthfulqa_p2_prompt_test_dataset/data2_size1000_eval_annotated.csv",
    #    max_samples=500,
    #    prompt_type=LABELING_PROBLEM_PROMPT,
    #    train=False
    #)