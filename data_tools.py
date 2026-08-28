import os
import json
import boto3
import torch
import pandas as pd
import numpy as np
from transformers import BitsAndBytesConfig
from huggingface_hub import model_info, InferenceClient
from datasets import load_dataset
from datetime import datetime
from dotenv import load_dotenv

from logger import Logger

logs = Logger()

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

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

def count_column_values():
    """
    Count how many times labels 0 and 1 were given (Prompt P2)
    """
    df = pd.read_csv("datasets/triviaqa_2/data_gpt-5.4_labels.csv")

    zero_labels = df[df["gpt-5.4_promptP2_labels"] == 0]
    one_labels = df[df["gpt-5.4_promptP2_labels"] == 1]
    print(len(zero_labels))
    print(len(one_labels))

    print(df["gpt-5.4_promptP2_labels"].value_counts())

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
    df["answer"] = df["answer"].apply(lambda x: x["aliases"])

    # Step 2.
    df = df[triviaqa_cols]
    logs.info("(data_tools.py) Formatted trivia_qa sample")

    return df

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
    #elif model_name == "IQuestLab/IQuest-Coder-V1-40B-Instruct":
    #    chat_template_kwargs = {
    #        "tokenize": True,
    #        "return_tensors": "pt",
    #        "return_dict": True,          
    #        "add_generation_prompt": True,
    #        "enable_thinking": False 
    #    }
    else:
        chat_template_kwargs = {
            "tokenize": True,
            "return_tensors": "pt",
            "return_dict": True,          
            "add_generation_prompt": True 
        }

    if "nvidia" in model_name:
        pass
    else: 
        model_kwargs["quantization_config"] = BNB_CONFIG

    return model_kwargs, chat_template_kwargs, tokenizer_kwargs

def get_model_generation_kwargs(
    model_name:str=None,
    tokenizer=None
):
    known_models = [
        "Qwen/Qwen3.6-35B-A3B",
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        "MultiverseComputingCAI/Hypernova-60B-2605",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "google/gemma-4-31B-it",
        "Qwen/Qwen3-Next-80B-A3B-Instruct"
    ]
        
    if model_name not in known_models:
        logs.error(f"{model_name} is unknown.")
        return None

    if model_name == "Qwen/Qwen3.6-35B-A3B":
        # Based on instruct mode params in huggingface
        return {
            "max_new_tokens": 150,
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
            "max_new_tokens": 150,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "pad_token_id": tokenizer.eos_token_id
        }
    
    # Default params for models not mentioned above
    return {
        "max_new_tokens": 150,
        "do_sample": True,
        "temperature": 0.7,
        "repetition_penalty": 1.0,
        "pad_token_id": tokenizer.eos_token_id # Comment out for gemma models
    }

def generate_bookkeeping(model_name, output_path, sample_size):
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

    with open(f"{output_path}/log_{identifier}.json", "w") as f:
        json.dump(log_entry, f)

"""
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id, # Comment out for gemma models
        )

        #top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
"""

if __name__ == "__main__":
    triviaqa_cols = [
        "question",
        "question_id",
        "question_source",
        "answer",
        "gen_answer"
    ]

    #format_answer_column(dataset_path="datasets/triviaqa_2/data.csv")
    create_data_subset(
        dataset_path="datasets/triviaqa_2/data.csv",
        subset_size=50,
        output_path="datasets/triviaqa_2/random_sample50.csv",
        columns=triviaqa_cols,
        format_answer_col=False
    )

    #count_column_values()