import json
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import re
from transformers import pipeline, StoppingCriteria, StoppingCriteriaList
import torch

# for llama models
class RefinedEndSentenceStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, sentence_end_tokens):
        super().__init__()
        self.tokenizer = tokenizer
        self.sentence_end_token_ids = [
            self.tokenizer.convert_tokens_to_ids(token) for token in sentence_end_tokens
        ]
        self.eos_token_id = tokenizer.eos_token_id  # Include eos_token_id

    def is_valid_stop(self, input_ids):
        # Get the last token and the one before it
        if len(input_ids[0]) < 2:
            return False  # Not enough tokens to decide
        last_token_id = input_ids[0, -1].item()
        second_last_token_id = input_ids[0, -2].item()

        # Decode tokens to check context
        last_token = self.tokenizer.decode([last_token_id])
        second_last_token = self.tokenizer.decode([second_last_token_id])

        # Stop if it's a sentence-ending token and not part of an abbreviation
        if (
            last_token in [".", "!", "?"]  # Check if it's a sentence-ending token
            and len(second_last_token) > 1  # Ensure not part of an abbreviation
            and not second_last_token.isupper()  # Ensure it's not "U.S." or similar
        ):
            return True

        # Include end-of-sequence token
        return last_token_id == self.eos_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return self.is_valid_stop(input_ids)


from tqdm.notebook import tqdm
import pandas as pd
from dataset_utils import create_results_df

def extract_response(text, prefix = "### Assistant:"):
    if prefix in text:
        return text.split(prefix, 1)[1].strip()
    return None


def generate_predictions_with_llama_instr(device, model, tokenizer, dataset, formatted_test_dataset, ds_type, setting, num_examples):

    output_name = f"{ds_type}-{setting}"
    search_type = {"beam-search":{"num_beams":4, "early_stopping":True, 
                              "filename":f"../data/gen_predictions/predictions_llama-instruct-few-shot-{output_name}-{num_examples}.csv"},
              "greedy":{"num_beams":1, "early_stopping":False,
                        "filename":f"../data/gen_predictions/predictions_llama-instruct-few-shot-{output_name}-greedy-{num_examples}.csv"}
    }

    # stopping criteria
    sentence_end_tokens = [".","\n","!", "?"]
    stopping_criteria = StoppingCriteriaList([RefinedEndSentenceStoppingCriteria(tokenizer, sentence_end_tokens)])

    # move to GPU
    model.to(device)
    model.eval()
    model.config.use_cache = True

    for search_t in search_type.keys():
        df_results = create_results_df(dataset)
        for idx, row in tqdm(df_results.iterrows(),total=len(df_results)):
            if row["pred_elaboration"]=="":
                inputs = tokenizer(
                    formatted_test_dataset[idx], #input_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2500 # 1024 for short, 2500 for long
                ).to(device)
                
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=32, 
                        min_length=10,
                        do_sample=False, 
                        temperature=None,  # not used in greedy decoding
                        top_p=None,# not used in greedy decoding
                        num_beams = search_type[search_t]["num_beams"],
                        early_stopping = search_type[search_t]["early_stopping"],
                        num_return_sequences=1,
                        no_repeat_ngram_size=3,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                        stopping_criteria=stopping_criteria
                    )
                
                generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                response = extract_response(generated_text) 
                df_results.at[idx,"pred_elaboration"] = response
        
        df_results.to_csv(search_type[search_t]["filename"], index=False)
        print(f"Saved {search_type[search_t]['filename']}")

import os
import shutil

def clear_directory(directory_path):
    """
    Deletes all files and subdirectories in the specified directory.

    """
    if os.path.exists(directory_path):
        # loop through all files and subdirectories in the directory
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            # check if it's a file or a directory
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  
        print(f"Cleared contents of: {directory_path}")
    else:
        print(f"Directory does not exist: {directory_path}")

def calculate_logging_steps(dataset, batch_size, logging_frequency):
    """
    Calculate the steps per epoch and logging steps for a given dataset and training configuration.
    """
    # the number of training samples
    num_samples = len(dataset["train"])
    
    # steps per epoch
    steps_per_epoch = num_samples // batch_size
    
    logging_steps = max(1, steps_per_epoch // logging_frequency)  # Avoid zero division or too low values
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Logging steps: {logging_steps}")
    
    return steps_per_epoch, logging_steps

import matplotlib.pyplot as plt

def plot_loss_curves(trainer):
    """
    Plot training and validation loss curves from the Trainer's log history.

    """
    # extract loss values from the trainer's log history
    train_loss = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    eval_loss = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    steps = range(len(train_loss))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, train_loss, label='Training Loss')
    plt.plot(steps[:len(eval_loss)], eval_loss, label='Validation Loss', linestyle='--')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def remove_tags(text):
    """
    Removes all tag-like elements (e.g., <tag>) from the given text.

    """
    if isinstance(text, str):
        return re.sub(r'<[^>]*>', '', text).strip()
    return text