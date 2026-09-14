import os
import torch
import json
import boto3
import ast
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMultimodalLM, BitsAndBytesConfig, AutoProcessor
from datasets import load_dataset
from huggingface_hub import model_info, InferenceClient
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from tabulate import tabulate

from logger import Logger
from data_tools import format_trivia_qa, get_model_generation_kwargs, get_model_and_tokenizer_kwargs, generate_bookkeeping

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
Functions for generating answers with LLMs
"""

def generate_answers_with_api(
    model_name:str,
    dataset_path:str,
    output_path:str,
    sample_size:int,
    verbose:bool,
    save_results:bool
):
    client = InferenceClient(api_key=HF_TOKEN)

    #client = OpenAI(
    #    base_url="https://router.huggingface.co/v1",
    #    api_key=HF_TOKEN
    #)

    questions = ["Can you access this website: https://da.wikipedia.org/wiki/Johnny_Depp ? If YES, return only the FIRST PARAGRAPH on the website. If NOT, return only 'No, I can not'."] 

    for i, q in enumerate(questions):

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{q}"},
                {"role": "assistant", "content": ""}
            ],
            max_tokens=128 # For openai
        )

        #print(completion)
        print("Response:", completion.choices[0].message.content)
        print("Total tokens:", completion.usage.total_tokens)

def generate_answers(
    model_name:str,
    dataset_path:str,
    output_path:str,
    sample_size:int,
    identifier:int,
    gguf_file:str=None,
    verbose:bool=False,
    save_results:bool=False,
):
    """
    Generate answers for a dataset. Pipeline:

    1. Load model
    2. Load dataset
    3. Generate answers until a desired amount of complete answers are ready
    4. Save results to a csv file and a log entry to a json file
    """
    device = "cuda"

    model_kwargs, chat_template_kwargs, tokenizer_kwargs = get_model_and_tokenizer_kwargs(model_name=model_name)

    # Load tokenizer and model
    if "gemma" in model_name:
        tokenizer = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto"
        )
    else:
        # Non-gemma models
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    logs.info(f"Loaded model {model_name}")
    eos_tokens = model.generation_config.eos_token_id
    if len(eos_tokens) == 0:
        logs.error("No EOS Tokens found")
        return
    logs.info(f"eos tokens: {eos_tokens}")    

    # Local csv file
    if dataset_path.endswith(".csv"):
        dataset = pd.read_csv(dataset_path)
    else:
        # Load dataset from huggingface and do preprocessing
        if "trivia_qa" in dataset_path:
            dataset = load_dataset(dataset_path, "rc.nocontext", split=f"validation")
            dataset = dataset.to_pandas()

            dataset = format_trivia_qa(dataset)

    dataset = dataset.sample(frac=1).reset_index(drop=True)
    questions = dataset["question"]

    logs.info(f"Dataset length: {dataset.shape}")

    # Arrays for answers, and 1/0 values indicating if the answers were completed
    answers, is_completed = [], []

    # Generate an answer for each question
    for i, q in enumerate(questions):
        #logs.info(f"Iteration number {i+1}")
        if sum(is_completed) == sample_size:
            logs.info(f"{sum(is_completed)} complete answers generated after {len(is_completed)} iterations")
            break
        if len(answers) == 10 and sum(is_completed) <= 3:
            logs.info(f"Model is producing mostly incomplete answers: {len(answers)} anwers, {sum(is_completed)} complete")
            break

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{q}"},
            {"role": "assistant", "content": ""}
        ]
        # Tokenize the input prompt
        tokenized_input = tokenizer.apply_chat_template(
            messages,
            **chat_template_kwargs
        ).to(device)                   

        # Get model-specific generation args
        generation_kwargs = get_model_generation_kwargs(
            model_name=model_name,
            tokenizer=tokenizer,
            annotate=False
        )
        # Generate tokens
        output_tokens = model.generate(
            **tokenized_input,
            **generation_kwargs
        )

        new_tokens = output_tokens[0][tokenized_input["input_ids"].shape[-1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Generated answers ends in an eos token
        if new_tokens[-1] in eos_tokens:
            is_completed.append(1)
        else:
            is_completed.append(0)
        answers.append(answer)
        
        if (i+1) % 10 == 0:
            print(f"{i+1} Answers generated, {sum(is_completed)} are complete")

    # The subset of data we want
    data_subset = dataset.iloc[:len(answers)]

    # Print information for user
    if verbose:
        for i, a in enumerate(answers):
            logs.info("###########")
            logs.info(f"question {i}: {questions.iloc[i]}")
            logs.info(f"answer {i}, complete={is_completed[i]}: {a}\n")
    
    # Save the generated answers and a log entry
    if save_results:
        logs.info(f"Dataset length before filtering: {len(data_subset)}")

        safe_model_name = model_name.replace("/", "_")
        
        data_subset.loc[:, f"gen_answer_{safe_model_name}"] = answers
        data_subset.loc[:, "is_complete"] = is_completed
        
        # Drop all rows that contain an incomplete answer
        data_subset = data_subset[data_subset["is_complete"] == 1]
        data_subset = data_subset.drop(columns=["is_complete"])

        path_to_results = f"{output_path}/data{identifier}_size{sample_size}.csv"

        logs.info(f"Dataset length after filtering: {len(data_subset)}")

        data_subset.to_csv(f"{output_path}/data{identifier}_size{sample_size}.csv", index=False)    
        generate_bookkeeping(
            model_name=model_name,
            output_path=output_path,
            sample_size=sample_size,
            identifier=identifier
        )
        logs.info(f"Saved results to: {path_to_results}")

if __name__ == "__main__":
    llama_3p2_1b = "meta-llama/Llama-3.2-1B-Instruct"
    llama_3p1_8b = "meta-llama/Llama-3.1-8B-Instruct"
    gemma_4_31b = "google/gemma-4-31B-it" # Try in Roihu
    gpt_oss_safeguard_120b = "openai/gpt-oss-safeguard-120b"
    gpt_oss_120b = "openai/gpt-oss-120b"
    gpt_5p4 = "gpt-5.4"

    # Bigger models
    qwen3_next_80b_a3b_instruct = "Qwen/Qwen3-Next-80B-A3B-Instruct" # Try via API
    llama_3_70b = "meta-llama/Meta-Llama-3-70B-Instruct" # Too large
    qwen_3p6_35b = "Qwen/Qwen3.6-35B-A3B" # Works

    nvidia_nemotron_3_super_120b = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
    iquest_coder_v1_40b = "IQuestLab/IQuest-Coder-V1-40B-Instruct" 
    

    generate_answers(
        model_name=gemma_4_31b,
        dataset_path="trivia_qa",
        output_path="datasets/sanity_check",
        sample_size=70,
        identifier=1,
        gguf_file=None,
        verbose=False,
        save_results=True
    )

    #generate_answers_with_api(
    #    model_name=qwen3_next_80b_a3b_instruct,
    #    dataset_path="",
    #    output_path="",
    #    sample_size=0,
    #    verbose=False,
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