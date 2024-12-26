import argparse
from dataset_utils import load_dataset_from_csv
from prompt_utils import formatting_prompt_func, base_prompt_fewshot, base_prompt_zeroshot_test
from transformers import StoppingCriteriaList, BartTokenizer, BartForConditionalGeneration, GenerationConfig
from model_utils import RefinedEndSentenceStoppingCriteria, generate_predictions_with_llama_instr, generate_predictions_with_bart, generate_predictions_with_llama
from tqdm.notebook import tqdm
import pandas as pd
from dataset_utils import create_results_df, tokenize_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

setting_ds_dict = {
    #"base": ["c2s", "c2sp", "c4s", "c4sp"],
    #"masked": ["c2s", "c2sp", "c4s", "c4sp"],
    "target-phrase": ["c2s", "c2sp", "c4s", "c4sp"],
    "target-sent": ["c2s", "c2sp", "c4s", "c4sp"],
    "target-sent-target": ["c2s", "c2sp", "c4s", "c4sp"],
}

example_num_versions = ["n3", "n6"]

# set GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(model_name):
    """
    Main function to load datasets, format prompts, and generate predictions.

    """
    print(f"Using model: {model_name}")
    # python generate_elaborations.py --model llama-instruct
    if model_name == "llama-instruct":   
        torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct', cache_dir="../models/llama/") 
        tokenizer.pad_token = tokenizer.eos_token
        EOS_TOKEN = tokenizer.eos_token
        model =  AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B-Instruct', cache_dir="../models/llama/", 
                                                      device_map={'':torch.cuda.current_device()})
            
    for setting, ds_types in setting_ds_dict.items():
        for ds_type in ds_types:
            print(f"Processing setting: {setting}, dataset type: {ds_type}")
            dataset = load_dataset_from_csv(ds_type, setting)
            
            if model_name == "llama-instruct":
                for num_examples in example_num_versions:
                    print(f"Formatting prompts with {num_examples} examples.")
                    formatted_test_dataset = formatting_prompt_func(dataset["test"], EOS=EOS_TOKEN, 
                                                                    base_prompt=base_prompt_fewshot, setting=setting,num_examples=num_examples)
                    generate_predictions_with_llama_instr(device, model, tokenizer, 
                                                          dataset, formatted_test_dataset, 
                                                          ds_type, setting, num_examples)
            elif model_name == "bart-ft":
                output_name =f"{ds_type}-{setting}"
                checkpoint_dir = f"../models/bart/bart-news-ft/results/results-bart-{output_name}"
                # find the last checkpoint number
                checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
                if not checkpoints:
                    raise FileNotFoundError(f"No checkpoints found in directory: {checkpoint_dir}")
                latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
                latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                
                tokenizer = BartTokenizer.from_pretrained('facebook/bart-base',use_fast=False, cache_dir="../models/bart/")
                model = BartForConditionalGeneration.from_pretrained(latest_checkpoint_path)
                tokenized_dataset, formatting_func = tokenize_dataset(dataset, ds_type, setting, tokenizer)
                    
                generate_predictions_with_bart(device, model, tokenizer, dataset, formatting_func, ds_type, setting)
        
            
            elif model_name == "llama-ft":
                output_name =f"{ds_type}-{setting}"
                checkpoint_dir = f"../models/llama/llama-news-ft/results/results-llama-alpaca-{output_name}"
                checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
                if not checkpoints:
                    raise FileNotFoundError(f"No checkpoints found in directory: {checkpoint_dir}")
                latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
                latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                model = AutoModelForCausalLM.from_pretrained(latest_checkpoint_path)
                tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint_path)
                EOS_TOKEN = tokenizer.eos_token
                formatted_test_dataset = formatting_prompt_func(dataset["test"], EOS=EOS_TOKEN, 
                                                                    base_prompt=base_prompt_zeroshot_test, setting=setting,test=True)
                generate_predictions_with_llama(device, model, tokenizer, 
                                                          dataset, formatted_test_dataset, 
                                                          ds_type, setting)

                

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate elaborations using a specified model.")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Specify the model name (e.g., 'llama-instruct', 'llama-ft','bart-ft')."
    )
    args = parser.parse_args()
    main(args.model)
