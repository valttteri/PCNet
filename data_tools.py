import os
import torch
import json
import boto3
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMultimodalLM, BitsAndBytesConfig, AutoProcessor
from datasets import load_dataset
from huggingface_hub import model_info, InferenceClient
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

from logger import Logger

logs = Logger()

def create_data_subset(
    dataset_path:str,
    subset_size:int,
    output_path:str,
    columns=None
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

if __name__ == "__main__":
    triviaqa_cols = [
        "question",
        "question_id",
        "question_source",
        "answer",
        "gen_answer"
    ]

    #format_answer_column(dataset_path="datasets/triviaqa_1/random_sample20.csv")
    #create_data_subset(
    #    dataset_path="datasets/triviaqa_1/data.csv",
    #    subset_size=20,
    #    output_path="datasets/triviaqa_1/random_sample20.csv",
    #    columns=triviaqa_cols
    #)
