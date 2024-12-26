import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def find_max_value(df,exclude_substrings=None, exclude_datasets=None,score_type=None):

    if exclude_substrings is None:
        exclude_substrings = []

    if exclude_datasets is None:
        exclude_datasets = []

    df = df[~df["dataset"].isin(exclude_datasets)].reset_index(drop=True)

    filtered_columns = [
        col for col in df.columns
        if not any(substr in col for substr in exclude_substrings) and col != "dataset"
        and (score_type is None or score_type in col)
    ]
    df = df[["dataset"] + filtered_columns]
    # find the maximum value
    max_value = df.drop(columns=["dataset"]).max().max() 
    
    # find the location (row and column) of the max value
    max_location = (df.drop(columns=["dataset"]) == max_value).stack().idxmax()
    
    # extract the dataset value and column name
    dataset_value = df.loc[max_location[0], "dataset"]  # Row index -> corresponding "dataset" column
    column_name = max_location[1]  
    
    print(f"Highest score: {max_value}")
    print(f"Dataset: {dataset_value}")
    print(f"Column: {column_name}")

# score type: b1 - bleu 1 , b2 - bleu2
def preprocess_dataframe(df, score_type, model_name, exclude_substrings=None):
    """
    Preprocess the DataFrame to filter only columns ending with '-b1' and rename them with the model name.

    """
    if exclude_substrings is None:
        exclude_substrings = []

    filtered_columns = [
        col for col in df.columns
        if col.endswith(f'-{score_type}') and not any(substr in col for substr in exclude_substrings)
    ]
    # filter columns that end with '-b1' along with the 'dataset' column
    filtered_df = df[['dataset'] + filtered_columns ]
    
    # rename the columns to include the model name
    filtered_df = filtered_df.rename(columns={col: f"{model_name}_{col}" for col in filtered_df.columns if col != 'dataset'})
    
    return filtered_df

def merge_dataframes(dfs):
    """
    Merge multiple DataFrames on the 'dataset' column.
    
    """
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='dataset')
    return merged_df


def compare_setting(dataframe, score_type,setting, title="Comparison by Type", dataset_column='dataset', exclude_datasets=None, exclude_columns=None,baseline=None):
    """
    Creates a line graph comparing models based on a specific column type (e.g., 'base-b1').

    """
    if exclude_datasets is None:
        exclude_datasets = []
        
    if exclude_columns is None:
        exclude_columns = []
    
    filtered_dataframe = dataframe[~dataframe[dataset_column].isin(exclude_datasets)].reset_index(drop=True)
    filtered_columns = [col for col in filtered_dataframe.columns #if setting in "".join(col.split('_')[-1]) and col not in exclude_columns]
                        if (
                            (setting == "target-sent" and setting in col and f"{setting}-target" not in col and f"{setting}-subject" not in col)
                            or (setting != "target-sent" and setting in "".join(col.split('_')[-1]))
                            )
                            and col not in exclude_columns
                        ]

    # line graph
    plt.figure(figsize=(12, 6))
    for col in filtered_columns:
        label_name = col.split('_')[0]
        if "llama-instr-few-shot" in label_name:
            label_name = "llama-instruct-" + "".join(col.split('_')[1].split('-')[-2])
        
        plt.plot(filtered_dataframe[dataset_column], filtered_dataframe[col], marker='o', label=label_name)

        # find the highest score for the current model
        max_idx = filtered_dataframe[col].idxmax()
        max_score = filtered_dataframe[col].iloc[max_idx]
        max_dataset = filtered_dataframe[dataset_column].iloc[max_idx]

        plt.text(
            x=max_dataset,
            y=max_score,
            s=f"{max_score:.3f}",
            fontsize=10,
            color="black",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="yellow", alpha=0.7)
        )

    # add baseline
    if baseline:
        if baseline in filtered_dataframe.columns:
            plt.plot(
                filtered_dataframe[dataset_column],
                filtered_dataframe[baseline],
                color='red',
                linestyle='--',
                linewidth=1.5,
                label=f"baseline"
            )

    # set labels, title, and legend
    plt.xlabel("Dataset")
    plt.ylabel(f"{score_type} Score")
    plt.title(title)
    plt.legend(title="Models", loc="best")

    # display grid
    plt.grid(True)
    plt.show()

def compare_prompt_setting(dataframe, score_type, dataset_column, column_type, title="Comparison by Type"):

    filtered_columns = [col for col in dataframe.columns if col.endswith(column_type)]

    plt.figure(figsize=(12, 6))
    for col in filtered_columns:
        plt.plot(dataframe[dataset_column], dataframe[col], marker='o', label="-".join(col.split("-")[:-1]))

        # highest score for the current prompt
        max_idx = dataframe[col].idxmax()
        max_score = dataframe[col].iloc[max_idx]
        max_dataset = dataframe[dataset_column].iloc[max_idx]

        # annotate the highest score on the graph
        plt.text(
            x=max_dataset,
            y=max_score,
            s=f"{max_score:.3f}",
            fontsize=10,
            color="black",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="yellow", alpha=0.7)
        )

    plt.xlabel("Dataset")
    plt.ylabel(f"{score_type} Score")
    plt.title(title)
    plt.legend(title="Prompts", loc="best")

    # Display grid and plot
    plt.grid(True)
    plt.show()

def compare_to_base(
    df,
    settings_to_compare,
    score_type,
    dataset_column='dataset',
    title="Comparison with Base Setting",
    exclude_columns=None
):

    if exclude_columns is None:
        exclude_columns = []

    # Extract base columns for all models
    base_columns = [
        col for col in df.columns
        if "base" in col and score_type in col and col not in exclude_columns
    ]
    """    # Filter columns for settings to compare
    compare_columns = [
        col for col in df.columns
        if any(setting in col for setting in settings_to_compare)
        and score_type in col
        and col not in exclude_columns
    ]
    """
    # Filter columns for settings to compare with additional conditions
    compare_columns = [
        col for col in df.columns
        if (
            (  # Handle the specific case for "target-sent"
                "target-sent" in settings_to_compare
                and "target-sent" in col
                and f"target-sent-target" not in col
                and f"target-sent-subject" not in col
            )
            or (  # General case for other settings
                any(setting in col for setting in settings_to_compare if setting != "target-sent")
                and score_type in col
            )
        )
        and col not in exclude_columns
    ]

    # Extract relevant data
    filtered_df = df[[dataset_column] + base_columns + compare_columns].dropna()

    for base_col in base_columns:
        model_name = base_col.split('_')[0] 
        for col in compare_columns:
            if model_name in col:  # Match the base column to the correct model's settings
                filtered_df[f"Difference_{col}"] = filtered_df[col] - filtered_df[base_col]

    # Plot the comparison
    plt.figure(figsize=(14, 8))
    # Assign unique colors for each setting
    colors = plt.cm.tab10(np.linspace(0, 1, len(compare_columns)))
    bar_width = 0.2  # Width of each bar
    x = np.arange(len(filtered_df[dataset_column]))  # X-axis positions

    for i, col in enumerate(compare_columns):
        # Extract model name and setting type
        #model_name = col.split('_')[0]  # Get the part before the first underscore
        #setting_type = col.split('_')[1] if '_' in col else col  # Get the part after the first underscore
    
        # Clean up and format label
        label_name = col.split('_')[0]
        if "llama-instr-few-shot" in col:
            label_name = "llama-instruct-" + col.split('_')[1].split('-')[-2]  # Simplify llama-instr label
    
        # Plot bars
        differences = filtered_df[f"Difference_{col}"]
        plt.bar(
            x + i * bar_width,  # Offset each bar group
            differences,
            color=colors[i],
            width=bar_width,
            label=label_name  # Use cleaned-up label for legend
        )

    # Add labels and title
    plt.xlabel("Dataset",fontsize=14)
    plt.ylabel("Score Difference",fontsize=14)
    plt.title(title,fontsize=16)
    plt.xticks(x + bar_width * (len(compare_columns) - 1) / 2, filtered_df[dataset_column], rotation=45)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
    plt.legend(title="Settings", loc="best")
    plt.tight_layout()
    plt.show()

    return filtered_df