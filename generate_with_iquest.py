import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_tools import format_trivia_qa, generate_bookkeeping
from logger import Logger

"""
Separate function for running an IQuest model.
The models requires transformers==4.xx.x, making it incompatible with generate_with_models.py
"""

logs = Logger()

def generate_answers_with_iquest(
    model_name:str,
    dataset_path:str,
    output_path:str,
    sample_size:str,
    identifier:int,
    verbose:bool=False,
    save_results:bool=False,
):
    """
    Generate model answers with an IQuest model
    """

    device="cuda"

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    eos_tokens = model.generation_config.eos_token_id

    if len(eos_tokens) == 0:
        logs.error("No EOS Tokens found")
        return

    logs.info(f"Loaded model {model_name} with eos tokens {eos_tokens}")

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

    logs.info(f"Dataset shape: {dataset.shape}")

    # Arrays for answers, and 1/0 values indicating if the answers were completed
    answers, is_completed = [], []

    # Prepare the input
    for i, q in enumerate(questions):
        logs.info(f"Iteration number {i+1}")
        if sum(is_completed) == sample_size:
            logs.info(f"{sum(is_completed)} complete answers generated after {len(is_completed)} iterations")
            break
        if len(answers) == 10 and sum(is_completed) <= 5:
            logs.info(f"Model is producing mostly incomplete answers: {len(answers)} anwers, {sum(is_completed)} complete")
            break

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
        ).to(model.device)

        # Generate response
        output_tokens = model.generate(
            **tokenized_input,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=150,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

        new_tokens = output_tokens[0][tokenized_input["input_ids"].shape[-1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=False)

        if new_tokens[-1] in eos_tokens:
            is_completed.append(1)
        else:
            is_completed.append(0)
        answers.append(answer)
        
        if (i+1) % 5 == 0:
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
        data_subset.loc[:, "gen_answer"] = answers
        data_subset.loc[:, "is_complete"] = is_completed
        
        # Drop all rows that contain an incomplete answer
        data_subset = data_subset[data_subset["is_complete"] == 1]
        data_subset = data_subset.drop(columns=["is_complete"])

        logs.info(f"Dataset length after filtering: {len(data_subset)}")

        return

        data_subset.to_csv(f"{output_path}/sample_{identifier}_size{sample_size}.csv", index=False)

        generate_bookkeeping(
            model_name=model_name,
            output_path=output_path,
            sample_size=sample_size,
            identifier=identifier
        )

if __name__ == "__main__":
    iquest_coder_v1_40b = "IQuestLab/IQuest-Coder-V1-40B-Instruct" 

    generate_answers_with_iquest(
        model_name=iquest_coder_v1_40b,
        dataset_path="trivia_qa",
        output_path="datasets/triviaqa_filtered_samples",
        sample_size=20,
        identifier=1,
        verbose=True,
        save_results=True
    )