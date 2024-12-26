import argparse
from dataset_utils import load_dataset_from_csv
from prompt_utils import formatting_prompt_func
from transformers import StoppingCriteriaList, BartForConditionalGeneration, BartTokenizer
from model_utils import RefinedEndSentenceStoppingCriteria, finetune_llama, finetune_bart
from tqdm.notebook import tqdm
import pandas as pd
from dataset_utils import create_results_df, tokenize_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

setting_ds_dict = {
    #"base": ["c2s", "c2sp", "c4s", "c4sp"],
    #"masked": ["c2s", "c2sp", "c4s", "c4sp"],
    "target-phrase": ["c2s", "c2sp", "c4s", "c4sp"],
    "target-sent": ["c2s", "c2sp", "c4s", "c4sp"],
    "target-sent-target":  ["c2s", "c2sp", "c4s", "c4sp"],
}


# set GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(model_name):
    """
    Main function to load datasets, format prompts, and fine-tune models.

    """
    print(f"Using model: {model_name}")
    # python finetune_model.py --model llama 
        
            
    for setting, ds_types in setting_ds_dict.items():
        for ds_type in ds_types:
            print(f"Processing setting: {setting}, dataset type: {ds_type}")
            dataset = load_dataset_from_csv(ds_type, setting)
            if model_name == "llama":
                torch.cuda.empty_cache()
                tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B', cache_dir="../models/llama/") 
                tokenizer.pad_token = tokenizer.eos_token
                model =  AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B', 
                                                              cache_dir="../models/llama/",
                                                              device_map={'':torch.cuda.current_device()})

                print(f"Fine-tuning model on {ds_type}-dataset.")
                finetune_llama(device, model, tokenizer, dataset, ds_type, setting)
            elif model_name == "bart":
                torch.cuda.empty_cache()
                tokenizer = BartTokenizer.from_pretrained('facebook/bart-base',use_fast=False, cache_dir="../models/bart/") 
                model = BartForConditionalGeneration.from_pretrained('facebook/bart-base', 
                                                                     cache_dir="../models/bart/", device_map ={'':torch.cuda.current_device()})

                tokenized_dataset, formatting_func = tokenize_dataset(dataset, ds_type, setting, tokenizer)
                print(f"Fine-tuning model on {ds_type}-dataset.")
                finetune_bart(device, model, tokenizer, tokenized_dataset, ds_type, setting)
                

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fine-tune a specified model.")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Specify the model name (e.g., 'llama','bart')."
    )
    args = parser.parse_args()
    main(args.model)
