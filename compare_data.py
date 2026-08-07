import os
import json
import pandas as pd
import numpy as np
from glob import glob
from tabulate import tabulate
from logger import Logger

logs = Logger()

def compare_dataframes(model_name: str, dataset_name: str):
    """
    Compare two dataframes (my results & paper results)
    Originally created for comparing results made with correction_pipeline.py

    
    """
    method_names = [
        "ITI [Gated]",
        "ITI [Blind]",
        "DoLa [Gated]",
        "DoLa [Blind]",
        "ICD [Gated]",
        "ICD [Blind]",
        "SADI [Gated]",
        "SADI [Blind]",
        "AdaSteer [Gated]",
        "AdaSteer [Blind]",
        "HalluCana [Gated]",
        "HalluCana [Blind]"
    ]

    # df names
    my_df_name = glob(f"correction_pipeline_logs/global_summary_run_3*.csv") # Seed 42
    paper_df_name = f"paper_logs/correction/42/{model_name}/{dataset_name}/comparison/method_summary.csv"

    # Load dataframes
    df_me = pd.read_csv(my_df_name[0], header=0, index_col=0)
    df_paper = pd.read_csv(paper_df_name, header=0, index_col=0)
    

    df_me_subset = df_me[(df_me["dataset"] == dataset_name) & (df_me["method"].isin(method_names))]
    df_me_subset = df_me_subset.drop(columns=["seed"])
    

    df_paper_subset = df_paper[(df_paper["dataset"] == dataset_name) & (df_paper["method"].isin(method_names))]
    df_paper_subset = df_paper_subset.drop(columns=["official_pre_token_f1", "official_post_token_f1", "official_delta_token_f1"])


    df_me_subset.to_csv("comparisons/df_me_subset.csv")
    df_paper_subset.to_csv("comparisons/df_paper_subset.csv")

    #df_subset = df_paper[["llm", "dataset", "method"]]
    #print(tabulate(df_subset, headers="keys", tablefmt="github"))

    #print(df_me_subset.shape, df_paper_subset.shape)

    df_comp = df_me_subset.compare(df_paper_subset, result_names=("me", "paper"))
    df_comp.to_csv(f"comparisons/df_comp_{dataset_name}.csv")

def average_of_dataframes(paper_mode: bool=False, dataset: str=None):
    """
    Compute average of dataframes produced by correction_pipeline.py
    """
    print("dataset:", dataset)
    method_names = [
        "ITI [Gated]",
        "ITI [Blind]",
        "DoLa [Gated]",
        "DoLa [Blind]",
        "ICD [Gated]",
        "ICD [Blind]",
        "SADI [Gated]",
        "SADI [Blind]",
        "AdaSteer [Gated]",
        "AdaSteer [Blind]",
    ]

    if paper_mode:
        df1_name = f"paper_logs/correction/42/meta-llama_Llama-3.2-1B-Instruct/{dataset}/comparison/method_summary.csv" # Seed 42
        df2_name = f"paper_logs/correction/43/meta-llama_Llama-3.2-1B-Instruct/{dataset}/comparison/method_summary.csv" # Seed 43
        df3_name = f"paper_logs/correction/44/meta-llama_Llama-3.2-1B-Instruct/{dataset}/comparison/method_summary.csv" # Seed 44
    else:
        df1_name = glob(f"correction_pipeline_logs/global_summary_run_3*.csv")[0] # Seed 42
        df2_name = glob(f"correction_pipeline_logs/global_summary_run_4*.csv")[0] # Seed 43
        df3_name = glob(f"correction_pipeline_logs/global_summary_run_5*.csv")[0] # Seed 44


    df1 = pd.read_csv(df1_name, header=0, index_col=0)
    df2 = pd.read_csv(df2_name, header=0, index_col=0)
    df3 = pd.read_csv(df3_name, header=0, index_col=0)

    df1 = df1[df1["method"].isin(method_names)]
    df2 = df2[df2["method"].isin(method_names)]
    df3 = df3[df3["method"].isin(method_names)]


    print("df shapes:", df1.shape, df2.shape, df3.shape)

    ## Split into numeric and non-numeric columns
    numeric_cols = df2.select_dtypes(include="number").columns
    non_numeric_cols = df2.select_dtypes(exclude="number").columns

    # Average numeric columns, cell-wise
    df_avg_numeric = pd.DataFrame(
       (df1[numeric_cols].values + df2[numeric_cols].values + df3[numeric_cols].values) / 3,
       index=df1.index,
       columns=numeric_cols
   )
    print(df_avg_numeric.shape)

    # Reattach non-numeric columns
    df_avg = pd.concat([df2[non_numeric_cols], df_avg_numeric], axis=1)

    # Restore original column order
    df_avg = df_avg[df2.columns]
    df_avg.to_csv(f"comparisons/paper_correction_avg_{dataset}_1.csv")
    #print(tabulate(df_avg_numeric, headers="keys", tablefmt="github"))

def average_of_dataframes_mc(**kwargs):
    """
    Compute average of datafames produced by correction_pipeline_mc
    """
    df1 = pd.read_csv(kwargs["df1_name"], header=0, index_col=0)
    df2 = pd.read_csv(kwargs["df2_name"], header=0, index_col=0)
    df3 = pd.read_csv(kwargs["df3_name"], header=0, index_col=0)

    numeric_cols = df1.select_dtypes(include="number").columns
    non_numeric_cols = df1.select_dtypes(exclude="number").columns

    # Average numeric columns, cell-wise
    df_avg_numeric = pd.DataFrame(
       (df1[numeric_cols].values + df2[numeric_cols].values + df3[numeric_cols].values) / 3,
       index=df1.index,
       columns=numeric_cols
    )

    # Reattach non-numeric columns
    df_avg = pd.concat([df2[non_numeric_cols], df_avg_numeric], axis=1)

    # Restore original column order
    df_avg = df_avg[df2.columns]

    try:
        df_avg.to_csv(f"comparisons/paper_correction_mc_1.csv")
        print("Saved results to a csv file")
    except Exception as e:
        print(f"Saving results failed:", e)
    
def remove_extra_index():
    df = pd.read_csv("datasets/truthfulqa_paper_gen.csv", header=0, index_col=0)
    df.to_csv("datasets/truthfulqa_paper_gen.csv", index=False)

def average_of_json_files(
    file_paths:list,
    model_name:str,
    dataset_name:str,
    output_path:str
):
    """
    Compute the average of json files generated by main.py
    """

    avg_values = {}

    for i, fp in enumerate(file_paths):
        with open(fp, "r") as file:
            data = json.load(file)

        for j, (method, metrics) in enumerate(data.items()):
            if method == "Experiment_Info":
                avg_values[method] = data[method]
                continue
            # Create an empty dict on the 1st iteration
            if method not in avg_values:
                avg_values[method] = {}

            for metric, value in metrics.items():
                if metric in avg_values[method]:
                    avg_values[method][metric].append(value)
                else:
                    # Create an empty list on the 1st iteration
                    avg_values[method][metric] = [value]

                # Last iteration: calculate average
                if i == len(file_paths) - 1:
                    avg_values[method][metric] = float(np.mean(avg_values[method][metric]))

    output_dir = f"{output_path}/{model_name}_{dataset_name}.json"
    with open(output_dir, "w") as f:
        json.dump(avg_values, f)

    logs.info("Saved average values to json")
    

def test():
    arr = [0.54624, 0.5453439999999999, 0.5460799999999999]
    mean = np.mean(arr)
    print(mean)

if __name__ == "__main__":
    #main(
    #    model_name="meta-llama_Llama-3.2-1B-Instruct",
    #    dataset_name="coqa"
    #)
    #datasets = os.listdir("paper_logs/correction/44/meta-llama_Llama-3.2-1B-Instruct")
    
    #for ds in datasets:
    #    average_of_dataframes(paper_mode=True, dataset=ds)

    #average_of_dataframes_mc(
    #    df1_name="2_last_correction_pipeline_mc_logs/global_mc_summary_seed42.csv",
    #    df2_name="2_last_correction_pipeline_mc_logs/global_mc_summary_seed43.csv",
    #    df3_name="2_last_correction_pipeline_mc_logs/global_mc_summary_seed44.csv"
    #)
    
    json_file_paths = [
        "all_logs/pcnet_detection_logs2/PCNet_Guardrail/42/meta-llama_Llama-3.2-1B-Instruct/truthful_qa/42/metrics.json",
        "all_logs/pcnet_detection_logs2/PCNet_Guardrail/43/meta-llama_Llama-3.2-1B-Instruct/truthful_qa/43/metrics.json",
        "all_logs/pcnet_detection_logs2/PCNet_Guardrail/44/meta-llama_Llama-3.2-1B-Instruct/truthful_qa/44/metrics.json"

    ]

    paper_detection_result_paths = [
        "all_logs/paper_logs/detection/42/meta-llama_Llama-3.1-8B-Instruct/coqa/metrics.json",
        "all_logs/paper_logs/detection/43/meta-llama_Llama-3.1-8B-Instruct/coqa/metrics.json",
        "all_logs/paper_logs/detection/44/meta-llama_Llama-3.1-8B-Instruct/coqa/metrics.json"
    ]

    #test()

    average_of_json_files(
        file_paths=paper_detection_result_paths,
        model_name="meta-llama_Llama-3.1-8B-Instruct",
        dataset_name="coqa",
        output_path="comparisons/detection_logs_paper",
    )