import os
import torch
import json
import boto3
import ast
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMultimodalLM, BitsAndBytesConfig, AutoProcessor
from datasets import load_dataset
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from tabulate import tabulate

from logger import Logger
from data_tools import format_trivia_qa, get_model_generation_kwargs, generate_bookkeeping

load_dotenv()
logs = Logger()

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL="https://bedrock-mantle.eu-north-1.api.aws/v1"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

"""
Functions for annotating models answers.
"""

def annotate_data_with_api(
    dataset_path:str,
    prompt_path:str,
    model_name:str,
    save_results=False
):
    """
    Annotate QA-pairs using the Huggingface API
    """

    client = InferenceClient(api_key=HF_TOKEN)
    dataset = pd.read_csv(dataset_path)

    with open(prompt_path, "r") as f:
        prompt = f.read()

    questions = dataset["question"]
    ref_answers = dataset["answer"]
    model_answers = dataset["gen_answer"]

    labels = []
    total_tokens = 0

    for i, (q, r, m) in enumerate(zip(questions, ref_answers, model_answers)):
        user_input = f"Question: {q} ### Reference answer: {r} ### Model answer: {m}"

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": ""}
            ],
        )

        label = completion.choices[0].message.content # Label decided by model
        tokens_spent = completion.usage.total_tokens # Cost of this api call
        
        labels.append(label)
        total_tokens += tokens_spent

    if save_results:
        dataset[f"{model_name}_valtteriV2"] = labels
        dataset.to_csv(dataset_path, index=False)
        logs.info(f"Annotation ready. Cost: {total_tokens} tokens.")

def annotate_data(
    dataset_path:str,
    prompt_path:str,
    model_name:str,
    save_results=False
):
    with open(prompt_path, "r") as f:
        prompt = f.read()

    # Local dataset    
    dataset = pd.read_csv(dataset_path)
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=BNB_CONFIG,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    questions = dataset["question"]
    ref_answers = dataset["answer"]
    model_answers = dataset["gen_answer"]

    labels = []

    for i, (q, r, m) in enumerate(zip(questions, ref_answers, model_answers)):
        user_input = f"Question: {q} ### Reference answer: {r} ### Model answer: {m}"

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ""}
        ]
        tokenized_input = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,          
            add_generation_prompt=True 
        ).to(device)                   

        output_tokens = model.generate(
            **tokenized_input,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id # Comment out for gemma models
        )

        new_tokens = output_tokens[0][tokenized_input["input_ids"].shape[-1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

        if i % 5 == 0:
            print(f"{i}/{len(dataset)} QA-pairs done")

        labels.append(answer)

    if save_results:
        dataset[f"{model_name}_valtteriV2"] = labels
        dataset.to_csv(dataset_path, index=False)
        logs.info("Annotation ready.")

def annotate_data_with_openai(
    dataset_path:str,
    prompt_path:str,
    output_path:str,
    model_name:str,
    save_results=False
):
    client = OpenAI()
    dataset = pd.read_csv(dataset_path)

    with open(prompt_path, "r") as f:
        prompt = f.read()

    questions = dataset["question"]
    ref_answers = dataset["answer"]
    model_answers = dataset["gen_answer"]

    labels = []
    confidences = []
    total_tokens = 0

    # Create a chatbot using ChatCompletion.create() function
    for i, (q, r, m) in enumerate(zip(questions, ref_answers, model_answers)):

        user_input = f"Question: {q} ### Reference answer: {r} ### Model answer: {m}"
        completion = client.chat.completions.create(
          model=model_name,
          messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ""}
          ]
        )

        response = completion.choices[0].message.content 
        response_dict = ast.literal_eval(response) # Label and confidence decided by model, as Python dict
        tokens_spent = completion.usage.total_tokens # Cost of this api call
        
        labels.append(response_dict["label"])
        confidences.append(response_dict["confidence"])
        total_tokens += tokens_spent

        if i % 10 == 0:
            logs.info(f"{i}/{len(dataset)} QA-pairs annotated. Total tokens spent: {total_tokens}")

    if save_results:
        dataset[f"{model_name}_promptP2_labels"] = labels
        dataset[f"{model_name}_promptP2_confs"] = confidences

        output_filepath = f"{output_path}/data_{model_name}_labels.csv"
        dataset.to_csv(output_filepath, index=False)
        logs.info(f"Annotation ready. Cost: {total_tokens} tokens.")


if __name__ == "__main__":
    llama_3p2_1b = "meta-llama/Llama-3.2-1B-Instruct"
    llama_3p1_8b = "meta-llama/Llama-3.1-8B-Instruct"
    gemma_4_31b = "google/gemma-4-31B-it" # Try in Roihu
    qwen3_next_80b_a3b_instruct = "Qwen/Qwen3-Next-80B-A3B-Instruct" # Try via API
    gpt_oss_safeguard_120b = "openai/gpt-oss-safeguard-120b"
    gpt_oss_120b = "openai/gpt-oss-120b"
    gpt_5p4 = "gpt-5.4"

    # Bigger models
    llama_3_70b = "meta-llama/Meta-Llama-3-70B-Instruct" # Too large
    qwen_3p6_35b = "Qwen/Qwen3.6-35B-A3B" # Works

    hypernova_60B = "MultiverseComputingCAI/Hypernova-60B-2605"
    nemotron_3_super_120b = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4" 
    
    #annotate_data_with_openai(
    #    dataset_path="datasets/triviaqa_2/data.csv",
    #    prompt_path="prompts/labeling_problem_p2.txt",
    #    output_path="datasets/triviaqa_2",
    #    model_name=gpt_5p4,
    #    save_results=True
    #)

    #annotate_data(
    #    dataset_path="datasets/triviaqa_1/random_sample20.csv",
    #    prompt_path="prompts/valtteri_tree_model_v2.txt",
    #    model_name=gemma_4_31b,
    #    save_results=True
    #)

    #annotate_data_with_api(
    #    dataset_path="datasets/triviaqa_1/random_sample20.csv",
    #    prompt_path="prompts/valtteri_tree_model_v2.txt",
    #    model_name=qwen3_next_80b_a3b_instruct,
    #    save_results=False
    #)

"""
Example HF API output:

ChatCompletionOutput(
    choices=[ChatCompletionOutputComplete(finish_reason='stop', index=0, message=ChatCompletionOutputMessage(role='assistant', content='France', reasoning=None, tool_call_id=None, tool_calls=None), logprobs=None)],
    created=1785762953,
    id='chatcmpl-d053a544-f4f5-49d7-9d6d-297d0dfa2eae',
    model='gemma-4-31b',
    system_fingerprint='fp_b2c59c382b338de84cfc',
    usage=ChatCompletionOutputUsage(completion_tokens=2, prompt_tokens=34, total_tokens=36, image_tokens=0, completion_tokens_details={'reasoning_tokens': 0}, prompt_tokens_details={'cached_tokens': 0}), 
    object='chat.completion',
    time_info={'created': 1785762953.9742298, 'queue_time': 0.476599331, 'prompt_time': 0.002530569, 'completion_time': 5.032e-05, 'total_time': 0.48673486709594727}
)

Example model transfer from allas:

cd huggingface/hub
a-get veahola/models--google--gemma-4-31B-it.tar
"""