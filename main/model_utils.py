import json
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList, DataCollatorForLanguageModeling, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, GenerationConfig
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from prompt_utils import formatting_prompt_func, base_prompt_zeroshot_train
import re
import os 


# FINE-TUNING
def finetune_bart(device, model, tokenizer, tokenized_dataset, ds_type, setting):
    
    # disable parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    model.train()

    output_name = f"{ds_type}-{setting}"

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"../models/bart/bart-news-ft/results/results-bart-{output_name}", 
        eval_strategy="epoch",
        save_strategy = "epoch",
        logging_strategy="steps",
        logging_steps=25,
        learning_rate=1e-5, # paper: 1e-4
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2, 
        num_train_epochs=3,
        save_total_limit=1,
        weight_decay=0.01,
        warmup_steps = 2,
        optim = "paged_adamw_8bit",
        load_best_model_at_end=True,
        greater_is_better=False, # the lower the loss the better
        fp16=True,
        logging_dir=f"../models/bart/bart-news-ft/logs/logs-bart-{output_name}",
        predict_with_generate=True,
        metric_for_best_model='eval_loss'
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
    )
    
    model.config.use_cache = False
    trainer.train()

    log_history = trainer.state.log_history
    eval_logs = [log for log in log_history if "eval_loss" in log]
    best_eval_log = min(eval_logs, key=lambda x: x["eval_loss"])
    print(f"Best validation loss: {best_eval_log['eval_loss']}")
    print(f"Epoch of the best model: {best_eval_log['epoch']}")
 


def finetune_llama(device, model, tokenizer, dataset, ds_type, setting):
    # disable parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # data collator
    data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False, response_template="### Assistant:")
    EOS_TOKEN = tokenizer.eos_token
    
    ds_maxlength_dict = {
        "c2s": 250,
        "c2sp": 280,
        "c4s": 280,
        "c4sp": 320
    }
    
    formatting_func_dict = {
        "base": lambda examples: formatting_prompt_func(examples, EOS=EOS_TOKEN, base_prompt=base_prompt_zeroshot_train, 
                                                        setting="base", num_examples=None),
        "masked": lambda examples: formatting_prompt_func(examples, EOS=EOS_TOKEN, base_prompt=base_prompt_zeroshot_train,  
                                                          setting="masked", num_examples=None),
        "target-phrase":lambda examples: formatting_prompt_func(examples, EOS=EOS_TOKEN, base_prompt=base_prompt_zeroshot_train, 
                                                                setting="target-phrase", num_examples=None),
        "target-sent": lambda examples: formatting_prompt_func(examples, EOS=EOS_TOKEN, base_prompt=base_prompt_zeroshot_train,  
                                                               setting="target-sent", num_examples=None),
        "target-sent-target":lambda examples: formatting_prompt_func(examples, EOS=EOS_TOKEN, base_prompt=base_prompt_zeroshot_train, 
                                                setting="target-sent-target", num_examples=None)
    }
    
    output_name = f"{ds_type}-{setting}"
    formatting_func = formatting_func_dict[setting]
    
    model.train()
    
    training_args = SFTConfig(
        output_dir=f"../models/llama/llama-news-ft/results/results-llama-alpaca-{output_name}",
        max_seq_length=ds_maxlength_dict[ds_type],
        eval_strategy="epoch",  
        save_strategy="epoch",  
        logging_strategy="steps",  
        logging_steps=6, 
        learning_rate=1e-6,  
        per_device_train_batch_size=32, 
        per_device_eval_batch_size=32,  
        num_train_epochs=3,  
        weight_decay=0.01,  
        warmup_steps=2,    
        optim="paged_adamw_8bit",  
        fp16=True,  
        logging_dir=f"../models/llama/llama-news-ft/logs/logs-llama-alapaca-{output_name}", 
        push_to_hub=False,
        group_by_length=True,  
        save_total_limit=1,
        packing=False,
        load_best_model_at_end=True, 
        greater_is_better=False,
        metric_for_best_model="eval_loss"

    )
    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"], 
        eval_dataset=dataset["validation"], 
        args=training_args,   
        formatting_func = formatting_func, 
        data_collator=data_collator,
        )

    model.config.use_cache = False
    trainer.train()
    
    log_history = trainer.state.log_history
    eval_logs = [log for log in log_history if "eval_loss" in log]
    best_eval_log = min(eval_logs, key=lambda x: x["eval_loss"])
    print(f"Best validation loss: {best_eval_log['eval_loss']}")
    print(f"Epoch of the best model: {best_eval_log['epoch']}")
    

# GENERATION
def generate_predictions_with_llama(device, model, tokenizer, dataset, formatted_test_dataset, ds_type, setting):
    
    model.eval()
    model.config.use_cache = True
    model.to(device)
    output_name = f"{ds_type}-{setting}"
    search_type = {"beam-search":{"num_beams":4, "early_stopping":True, 
                              "filename":f"../data/gen_predictions/predictions_llama-ft-{output_name}.csv"},
              "greedy":{"num_beams":1, "early_stopping":False,
                        "filename":f"../data/gen_predictions/predictions_llama-ft-{output_name}-greedy.csv"}
    }

    # stopping criteria
    sentence_end_tokens = [".","\n","!", "?"]
    stopping_criteria = StoppingCriteriaList([RefinedEndSentenceStoppingCriteria(tokenizer, sentence_end_tokens)])
    
    for search_t in search_type.keys():
    
        df_results = create_results_df(dataset)
        
        for idx, row in tqdm(df_results.iterrows(),total=len(df_results)):
            if row["pred_elaboration"]=="":
                inputs = tokenizer(
                    formatted_test_dataset[idx], 
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512 
                ).to(device) 
                
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=32,  
                        min_length=10,
                        do_sample=False,  
                        temperature=None,  
                        top_p=None,
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
        print("Saved ", search_type[search_t]["filename"])

def generate_predictions_with_bart(device, model, tokenizer, dataset, formatting_func, ds_type, setting):
    
    model.eval()
    model.config.use_cache = True
    model.to(device)
    output_name = f"{ds_type}-{setting}"
    search_type = {"beam-search":{"num_beams":4, "early_stopping":True, 
                              "filename":f"../data/gen_predictions/predictions_bart-ft-{output_name}.csv"},
              "greedy":{"num_beams":1, "early_stopping":False,
                        "filename":f"../data/gen_predictions/predictions_bart-ft-{output_name}-greedy.csv"}
    
    }
    for search_t in search_type.keys():

        df_results = create_results_df(dataset)
        generation_config = GenerationConfig(
            bos_token_id = 0,
            decoder_start_token_id = 2,
            early_stopping = search_type[search_t]["early_stopping"], 
            do_sample = False,
            max_new_tokens = 32,
            eos_token_id = 2, 
            forced_bos_token_id = 0,
            forced_eos_token_id = 2,
            no_repeat_ngram_size = 3,
            num_beams = search_type[search_t]["num_beams"], 
            pad_token_id = 1,
            num_return_sequences=1
        )
        if formatting_func:
            test_dataset = formatting_func(dataset["test"])
        else:
            test_dataset = dataset["test"]["source_text"]
    
        with torch.no_grad(): 
            for idx, example in tqdm(enumerate(test_dataset),total=len(test_dataset)):
                # tokenize the text
                input_ids = tokenizer(example, return_tensors="pt")
                # move input_ids to the same device as the model
                input_ids = {key: value.to(device) for key, value in input_ids.items()}
                
                # generate predictions
                output_ids = model.generate(**input_ids, generation_config=generation_config ) 
                elaboration = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                df_results.at[idx,"pred_elaboration"] = elaboration
    
    
        df_results.to_csv(search_type[search_t]["filename"], index=False)
        print(f"Saved {search_type[search_t]['filename']}")

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

import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
import os

class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self):
        """ Load model from paraphrase finetuning """
        self.model.load_state_dict(torch.load('models/bart_score.pth', map_location=self.device))

    def score(self, srcs, tgts, batch_size):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

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