import os
import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMultimodalLM, BitsAndBytesConfig, AutoProcessor
from datasets import load_dataset
from huggingface_hub import model_info, InferenceClient
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

from logger import Logger

load_dotenv()
logs = Logger()

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL="https://bedrock-mantle.eu-north-1.api.aws/v1"

def generate_answers(
    model_name:str,
    dataset_path:str,
    output_path:str,
    sample_size:int,
    verbose:bool,
    save_results:bool
):
    """
    Generate answers for a dataset.
    """
    device = "cuda"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if "llama" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
    elif "gemma" in model_name:
        tokenizer = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto"
        )
    logs.info(f"Loaded model {model_name}")

    # Local csv file
    if dataset_path.endswith(".csv"):
        dataset = pd.read_csv(dataset_path)
    else:
        # Load dataset from huggingface
        if "trivia_qa" in dataset_path:
            dataset = load_dataset(dataset_path, "rc.nocontext", split=f"validation[:{sample_size}]")
        dataset = dataset.to_pandas()

    questions = dataset["question"]
    answers = [] 

    # Generate an answer for each question
    for i, q in enumerate(questions):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{q}"},
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
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.1
            #pad_token_id=tokenizer.eos_token_id,
        )

        new_tokens = output_tokens[0][tokenized_input["input_ids"].shape[-1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

        if i % 25 == 0:
            print(f"{i}/{len(dataset)} QA-pairs done")

        answers.append(answer)
    
    dataset["gen_answer"] = answers

    # Print information for user
    if verbose:
        logs.info(f"questions: {questions}")
        
        for i, a in enumerate(answers):
            logs.info("###########")
            logs.info(f"answer {i}: {a}")
    
    # Save the generated answers and a log entry
    if save_results:
        dataset.to_csv(f"{output_path}/data.csv", index=False)
    
        generate_bookkeeping(
            model_name=model_name,
            output_path=output_path,
            sample_size=sample_size
        )

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
        "split": f"validation[:{sample_size}]"
    }

    log_entry = {
        "model_name": model_name,
        "model_sha": commit_id,
        "dataset": dataset_info,
        "date": formatted_date,
        "source_code": "generate_data.py",
        "prompt": prompt
    }

    with open(f"{output_path}/log.json", "w") as f:
        json.dump(log_entry, f)

def generate_answers_with_api(
    model_name:str,
    dataset_path:str,
    output_path:str,
    sample_size:int,
    verbose:bool,
    save_results:bool
):
    #client = InferenceClient(api_key=HF_TOKEN)

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN
    )

    questions = ["Who are you?", "Where are you from?", "Name a European country."]

    for i, q in enumerate(questions):

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{q}"},
                {"role": "assistant", "content": ""}
            ],
            max_tokens=128
        )

        #print(completion)
        print("Response:", completion.choices[0].message.content)
        print("Total tokens:", completion.usage.total_tokens)

def generate_answers_with_aws():
    client = OpenAI()

    questions = ["Where is France located?", "When is Finland's independence day?", "What is the most popular dog breed?"]

    for q in questions:

        response = client.chat.completions.create(
            model="openai.gpt-oss-safeguard-120b",
            messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{q}"},
                    {"role": "assistant", "content": ""}
                ],
        )
    
        print(response)


if __name__ == "__main__":
    llama_3p2_1b = "meta-llama/Llama-3.2-1B-Instruct"
    gemma_4_31b = "google/gemma-4-31B-it" # Works in Roihu
    qwen3_next_80b_a3b_instruct = "Qwen/Qwen3-Next-80B-A3B-Instruct" # Works in Roihu
    gpt_oss_safeguard_120b = "openai/gpt-oss-safeguard-120b"

    #generate_answers_with_api(
    #    model_name=gpt_oss_safeguard_120b,
    #    dataset_path="",
    #    output_path="",
    #    sample_size=10,
    #    verbose=False,
    #    save_results=False
    #)


    #generate_answers(
    #    model_name=gemma_4_31b,
    #    dataset_path="trivia_qa",
    #    output_path="datasets/triviaqa_1",
    #    sample_size=5,
    #    verbose=True,
    #    save_results=False
    #)

    generate_answers_with_aws()

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
"""