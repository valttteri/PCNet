import os
import json
import boto3
import pandas as pd
import numpy as np
from datasets import load_dataset
from datetime import datetime
from dotenv import load_dotenv

from logger import Logger

logs = Logger()

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

def get_model_generation_kwargs(model_name:str=None, default:bool=False):
    known_models = [
        "Qwen/Qwen3.6-35B-A3B"
    ]
    if default:
        return {
            "max_new_tokens": 150,
            "do_sample": True,
            "temperature": 0.7,
            "repetition_penalty": 1.1,
            #"pad_token_id": tokenizer.eos_token_id # Comment out for gemma models
        }

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
            "precence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "pad_token_id": tokenizer.eos_token_id
        }

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