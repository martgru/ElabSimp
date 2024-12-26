from datasets import load_dataset
import pandas as pd
import numpy as np
import os

data_path = "../data/elaborations"

def load_dataset_from_csv(ds_type, setting):

    if setting == "masked":
        datasets = {
            "c2s":{
            'train': os.path.join(data_path,"train","train_ds_c2s_masked_llama.csv"),      
            'validation': os.path.join(data_path,"validation","valid_ds_c2s_masked_llama.csv"),  
            'test': os.path.join(data_path,"test","test_ds_c2s_masked_llama.csv")         
            },
            "c2sp":{
                'train': os.path.join(data_path,"train","train_ds_c2sp_masked_llama.csv"),      
                'validation': os.path.join(data_path,"validation","valid_ds_c2sp_masked_llama.csv"),  
                'test': os.path.join(data_path,"test","test_ds_c2sp_masked_llama.csv")         
            },
            "c4s":{
                'train': os.path.join(data_path,"train","train_ds_c4s_masked_llama.csv"),      
                'validation': os.path.join(data_path,"validation","valid_ds_c4s_masked_llama.csv"),  
                'test': os.path.join(data_path,"test","test_ds_c4s_masked_llama.csv")         
            },
            "c4sp":{
                'train': os.path.join(data_path,"train","train_ds_c4sp_masked_llama.csv"),      
                'validation': os.path.join(data_path,"validation","valid_ds_c4sp_masked_llama.csv"),  
                'test': os.path.join(data_path,"test","test_ds_c4sp_masked_llama.csv")         
            },
        }
    else:

        datasets = {
            "c2s":{
                'train': os.path.join(data_path,"train","train_ds_c2s.csv"),      
                'validation': os.path.join(data_path,"validation","valid_ds_c2s.csv"),  
                'test': os.path.join(data_path,"test","test_ds_c2s.csv")         
            },
            "c2sp":{
                'train': os.path.join(data_path,"train","train_ds_c2sp.csv"),      
                'validation': os.path.join(data_path,"validation","valid_ds_c2sp.csv"),  
                'test': os.path.join(data_path,"test","test_ds_c2sp.csv")         
            },
            "c4s":{
                'train': os.path.join(data_path,"train","train_ds_c4s.csv"),      
                'validation': os.path.join(data_path,"validation","valid_ds_c4s.csv"),  
                'test': os.path.join(data_path,"test","test_ds_c4s.csv")         
            },
            "c4sp":{
                'train': os.path.join(data_path,"train","train_ds_c4sp.csv"),      
                'validation': os.path.join(data_path,"validation","valid_ds_c4sp.csv"),  
                'test': os.path.join(data_path,"test","test_ds_c4sp.csv")         
            },
            "c2o":{
                'train': os.path.join(data_path,"train","train_ds_c2o.csv"),      
                'validation': os.path.join(data_path,"validation","valid_ds_c2o.csv"),  
                'test': os.path.join(data_path,"test","test_ds_c2o.csv")         
            },
            "c2op":{
                'train': os.path.join(data_path,"train","train_ds_c2op.csv"),      
                'validation': os.path.join(data_path,"validation","valid_ds_c2op.csv"),  
                'test': os.path.join(data_path,"test","test_ds_c2op.csv")         
            },
            "c4o":{
                'train': os.path.join(data_path,"train","train_ds_c4o.csv"),      
                'validation': os.path.join(data_path,"validation","valid_ds_c4o.csv"),  
                'test': os.path.join(data_path,"test","test_ds_c4o.csv")         
            },
            "c4op":{
                'train': os.path.join(data_path,"train","train_ds_c4op.csv"),      
                'validation': os.path.join(data_path,"validation","valid_ds_c4op.csv"),  
                'test': os.path.join(data_path,"test","test_ds_c4op.csv")         
            },
            
            
        }
     # dataframes with elaboration targets
    if "o" in ds_type:
        df_train = pd.read_csv(os.path.join(data_path, "train",f"train_ds_c2o_elab_targets.csv"))
        df_valid = pd.read_csv(os.path.join(data_path, "validation",f"validation_ds_c2o_elab_targets.csv"))
        df_test = pd.read_csv(os.path.join(data_path, "test",f"test_ds_c2o_elab_targets.csv"))
    else: 
        df_train = pd.read_csv(os.path.join(data_path, "train",f"train_ds_{ds_type}_elab_targets.csv"))
        df_valid = pd.read_csv(os.path.join(data_path, "validation",f"validation_ds_{ds_type}_elab_targets.csv"))
        df_test = pd.read_csv(os.path.join(data_path, "test",f"test_ds_{ds_type}_elab_targets.csv"))

    dataset = load_dataset('csv', data_files=datasets[ds_type])
    col_names = []

    if setting == "target-phrase":
        col_names.append("target_sentence_target")
    elif setting == "target-sent":
        col_names.append("target_sentence_4o")
    elif setting == "target-sent-target":
        col_names.append("target_sentence_4o")
        col_names.append("target_sentence_target")
    elif setting == "subject":
        col_names.append("subject")

    if setting != "base" or setting != "masked":
        for col_name in col_names:
            dataset["train"] = dataset["train"].add_column(col_name, df_train[col_name])
            dataset["validation"] = dataset["validation"].add_column(col_name, df_valid[col_name])
            dataset["test"] = dataset["test"].add_column(col_name, df_test[col_name])
    
    return dataset

def tokenize_dataset(dataset, ds_type, setting, tokenizer):
    
    def format_sent_target(examples):
        texts = []
        contexts = examples['source_text']
        target_sents = examples['target_sentence_4o']
        targets = examples['target_sentence_target'] 
        for context, sent, target in zip(contexts, target_sents, targets):
            if sent is not None and target is not None: 
                texts.append(context + " target_sentence=" + sent + ", " + target)
            elif sent is None and target is not None: 
                texts.append(context + " " + target)
            else:
                texts.append(context)
        return texts

    def format_target(examples, target_type):
        texts = []
        contexts = examples['source_text'] 
        targets = examples[target_type] 
        for context, target in zip(contexts, targets):
            if target is not None: 
                if target_type == "target_sentence_4o":
                    texts.append(context + " target_sentence="  + target) 
                elif target_type == "target_sentence_target":
                    texts.append(context + " "  + target) 
                elif target_type == "subject":
                    texts.append(context + " subject="  + target) 
            else:
                texts.append(context)
        return texts

    if ds_type == "c2s":
        MAX_LEN = 200
    elif ds_type == "c2sp" or ds_type == "c4s":
        MAX_LEN = 250
    elif ds_type == "c4sp":
        MAX_LEN = 300
    elif "o" in ds_type:
        MAX_LEN = 512

    formatting_func = None

    if setting == "target-phrase":
        formatting_func = lambda examples: format_target(examples, target_type="target_sentence_target")
    elif setting == "target-sent":
        formatting_func = lambda examples: format_target(examples, target_type="target_sentence_4o")
    elif setting == "target-sent-target":
        formatting_func = format_sent_target
    elif setting == "subject":
        formatting_func = lambda examples: format_target(examples, target_type="subject")

    def tokenize_func(examples):

        if formatting_func:
            inputs = tokenizer(formatting_func(examples), truncation=True, padding="max_length", max_length=MAX_LEN)
        else:
            inputs = tokenizer(examples['source_text'], truncation=True, padding="max_length", max_length=MAX_LEN)

        labels = tokenizer(examples['elaboration_sentence'],truncation=True, padding="max_length", max_length=32)
        inputs['labels'] = np.array(labels['input_ids']) 
        return inputs
        

    tokenized_dataset = dataset.map(tokenize_func, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return tokenized_dataset, formatting_func


def create_results_df(dataset):
    df_results = pd.DataFrame({
        'source_text': dataset['test']['source_text'],
        'elaboration_sentence': dataset['test']['elaboration_sentence'],
        'target_sentence_target': dataset['test']['target_sentence_target'] if 'target_sentence_target' in dataset['test'].column_names else "",
        'target_sentence': dataset['test']['target_sentence_4o'] if 'target_sentence_4o' in dataset['test'].column_names else "",
        'subject': dataset['test']['subject'] if 'subject' in dataset['test'].column_names else "",
        'pred_elaboration': ""
    })
    return df_results

def create_scores_df(df_gen):
    df_scores = pd.DataFrame({
        'source_text': df_gen['source_text'] if 'source_text' in df_gen else None,
        'target_sentence': (
            df_gen['target_sentence_4o'] if 'target_sentence_4o' in df_gen
            else df_gen['target_sentence'] if 'target_sentence' in df_gen
            else None
        ),
        'target_sentence_target': df_gen['target_sentence_target'] if 'target_sentence_target' in df_gen else None,
        'subject': df_gen['subject'] if 'subject' in df_gen else None,
        'target-phrase': df_gen['target-phrase'] if 'target-phrase' in df_gen else None,
        'elaboration_sentence': df_gen['elaboration_sentence'],
        'pred_elaboration': df_gen['pred_elaboration'],
    })
    return df_scores
