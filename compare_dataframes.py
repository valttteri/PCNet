import os
import pandas as pd
from glob import glob
from tabulate import tabulate

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

if __name__ == "__main__":
    #main(
    #    model_name="meta-llama_Llama-3.2-1B-Instruct",
    #    dataset_name="coqa"
    #)
    datasets = os.listdir("paper_logs/correction/44/meta-llama_Llama-3.2-1B-Instruct")
    
    for ds in datasets:
        average_of_dataframes(paper_mode=True, dataset=ds)


    #average_of_dataframes(paper_mode=True)