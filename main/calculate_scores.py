import argparse
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from dataset_utils import create_scores_df
from bert_score import BERTScorer
from model_utils import BARTScorer
import numpy as np
from tqdm import tqdm
import pandas as pd


models = ["llama-ft","bart-ft","llama-instruct-few-shot"]

setting_ds_dict = {
   # "base": ["c2s","c2sp","c4s","c4sp"],
   # "masked": ["c2s","c2sp","c4s","c4sp"],
    "target-phrase":["c2s","c2sp","c4s","c4sp"],
    "target-sent":["c2s","c2sp","c4s","c4sp"],
    "target-sent-target":["c2s","c2sp","c4s","c4sp"],
}

num_examples = ["n3","n6"]

bart_ft_res = pd.read_csv("../data/results/bart-ft-results.csv")
llama_ft_res = pd.read_csv("../data/results/llama-ft-results.csv")
llama_instr_res = pd.read_csv("../data/results/llama-instruct-few-shot-results.csv")


def calculate_bleu(model, tokenizer, setting_key, ds, num_examples=None):

    smoothing_function = SmoothingFunction().method1
    
    df_res = pd.read_csv(f"../data/results/{model}-results.csv")

    if num_examples:
        for num_example in num_examples:
            all_refs = []
            all_preds = []
            output_name = f"{ds}-{setting_key}-{num_example}"
            df_gen = pd.read_csv(f"../data/gen_predictions/predictions_{model}-{output_name}.csv")
            for index, row in tqdm(df_gen.iterrows(), total=len(df_gen)):
                ref = row['elaboration_sentence']
                prediction = row['pred_elaboration'] 
        
                tokenized_ref = tokenizer(ref).split()
                tokenized_pred = tokenizer(prediction).split()
                all_refs.append([tokenized_ref]) 
                all_preds.append(tokenized_pred)

            bleu1_score = corpus_bleu(all_refs, all_preds, weights=(1.0, 0, 0, 0), smoothing_function=smoothing_function)  # 1-gram
            bleu2_score = corpus_bleu(all_refs, all_preds, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)  # 2-gram
            idx = df_res.index[df_res["dataset"] == ds].tolist()[0]
            
            df_res.at[idx, f"{setting_key}-{num_example}-b1"] = round(bleu1_score*100,3)
            df_res.at[idx, f"{setting_key}-{num_example}-b2"] = round(bleu2_score*100,3)
            df_res.to_csv(f"../data/results/{model}-results.csv",index=False)
            print(f"Results saved for {model}-{output_name}")
    else:
        all_refs = []
        all_preds = []
        output_name = f"{ds}-{setting_key}"
        df_gen = pd.read_csv(f"../data/gen_predictions/predictions_{model}-{output_name}.csv")
        
        for index, row in tqdm(df_gen.iterrows(), total=len(df_gen)):
            ref = row['elaboration_sentence']
            prediction = row['pred_elaboration'] 
    
            tokenized_ref = tokenizer(ref).split()
            tokenized_pred = tokenizer(prediction).split()
            all_refs.append([tokenized_ref]) 
            all_preds.append(tokenized_pred)
    
        bleu1_score = corpus_bleu(all_refs, all_preds, weights=(1.0, 0, 0, 0), smoothing_function=smoothing_function)  # 1-gram
        bleu2_score = corpus_bleu(all_refs, all_preds, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)  # 2-gram

        idx = df_res.index[df_res["dataset"] == ds].tolist()[0]
        df_res.at[idx, f"{setting_key}-b1"] = round(bleu1_score*100,3)
        df_res.at[idx, f"{setting_key}-b2"] = round(bleu2_score*100,3)
        df_res.to_csv(f"../data/results/{model}-results.csv",index=False)
        print(f"Results saved for {model}-{output_name}")

def calculate_bertscore(model, scorer, setting_key, ds, num_examples=None):
    
    df_res = pd.read_csv(f"../data/results/{model}-results.csv")
    if num_examples:
        for num_example in num_examples:
            bert_scores_f1 = []
            output_name = f"{ds}-{setting_key}-{num_example}" 
            df_gen = pd.read_csv(f"../data/gen_predictions/predictions_{model}-{output_name}.csv")
            
            for index, row in tqdm(df_gen.iterrows(), total=len(df_gen)):
                elaboration = row['elaboration_sentence']
                prediction = row['pred_elaboration']
                
                P, R, F1 = scorer.score(
                    cands=[prediction],  
                    refs=[elaboration],              
                )
                
                bert_scores_f1.append(F1.mean().item())
            avg_f1 = np.mean(bert_scores_f1)
            idx = df_res.index[df_res["dataset"] == ds].tolist()[0]
            df_res.at[idx, f"{setting_key}-{num_example}-bsf1"] = round(avg_f1,3)
            df_res.to_csv(f"../data/results/{model}-results.csv",index=False)
            print(f"Results saved for {model}-{output_name}")
    else:
        bert_scores_f1 = []
        output_name = f"{ds}-{setting_key}" 
        df_gen = pd.read_csv(f"../data/gen_predictions/predictions_{model}-{output_name}.csv")
        
        for index, row in tqdm(df_gen.iterrows(), total=len(df_gen)):
            elaboration = row['elaboration_sentence']
            prediction = row['pred_elaboration']
            
            P, R, F1 = scorer.score(
                cands=[prediction],  
                refs=[elaboration],              
            )
            
            bert_scores_f1.append(F1.mean().item())
        avg_f1 = np.mean(bert_scores_f1)
        idx = df_res.index[df_res["dataset"] == ds].tolist()[0]
        df_res.at[idx, f"{setting_key}-bsf1"] = round(avg_f1,3)
        print(f"{model}-{output_name}: {round(avg_f1,3)}")
        df_res.to_csv(f"../data/results/{model}-results.csv",index=False)
        print(f"Results saved for {model}-{output_name}")

def calculate_bartscore(model, scorer, setting_key, ds, num_examples=None):

    df_res = pd.read_csv(f"../data/results/{model}-results.csv")

    if num_examples:
        for num_example in num_examples:
            bart_scores = []
            output_name = f"{ds}-{setting_key}-{num_example}"
            df_gen = pd.read_csv(f"../data/gen_predictions/predictions_{model}-{output_name}.csv")
            for index, row in tqdm(df_gen.iterrows(), total=len(df_gen)):
                reference = row['elaboration_sentence']  # reference text (r)
                hypothesis = row['pred_elaboration']    # generated text (h)
                
                # precision (r → h)
                precision_score = scorer.score(
                    srcs=[reference],  # r as source
                    tgts=[hypothesis], # h as target
                    batch_size=1
                )[0]
                
                # recall (h → r)
                recall_score = scorer.score(
                    srcs=[hypothesis],  # h as source
                    tgts=[reference],   # r as target
                    batch_size=1
                )[0]
                
                # f1 score as the average of precision and recall
                f1_score = (precision_score + recall_score) / 2
                bart_scores.append(f1_score)
            avg_score = np.mean(bart_scores)
            idx = df_res.index[df_res["dataset"] == ds].tolist()[0]
            df_res.at[idx, f"{setting_key}-{num_example}-bartscore"] = round(avg_score,3)
            print(f"{model}-{ds}-{setting_key}-{num_example}: {round(avg_score,3)}")
            df_res.to_csv(f"../data/results/{model}-results.csv",index=False)
            print(f"Results saved for {model}-{output_name}")
    else:
        bart_scores = []
        output_name = f"{ds}-{setting_key}"
        df_gen = pd.read_csv(f"../data/gen_predictions/predictions_{model}-{output_name}.csv")
        for index, row in tqdm(df_gen.iterrows(), total=len(df_gen)):
            reference = row['elaboration_sentence']  # reference text (r)
            hypothesis = row['pred_elaboration']    # generated text (h)
            
            # precision (r → h)
            precision_score = scorer.score(
                srcs=[reference],  # r as source
                tgts=[hypothesis], # h as target
                batch_size=1
            )[0]
            
            # recall (h → r)
            recall_score = scorer.score(
                srcs=[hypothesis],  # h as source
                tgts=[reference],   # r as target
                batch_size=1
            )[0]
            
            # f1 score as the average of precision and recall
            f1_score = (precision_score + recall_score) / 2
            bart_scores.append(f1_score)
        avg_score = np.mean(bart_scores)
        idx = df_res.index[df_res["dataset"] == ds].tolist()[0]
        df_res.at[idx, f"{setting_key}-bartscore"] = round(avg_score,3)
        print(f"{model}-{ds}-{setting_key}: {round(avg_score,3)}")
        df_res.to_csv(f"../data/results/{model}-results.csv",index=False)
        print(f"Results saved for {model}-{output_name}")
        


def main(score_type):
    """
    Main function to load datasets, format prompts, and generate predictions.

    """
    print(f"Calculating: {score_type} scores.")
    if score_type == "bleu": 
        tokenizer = Tokenizer13a()
        for model in models: 
            for setting_key, ds_values in setting_ds_dict.items():
                for ds in ds_values:     
                    if model == "llama-instruct-few-shot":
                        calculate_bleu(model, tokenizer, setting_key, ds, num_examples)
                    else:
                        calculate_bleu(model, tokenizer, setting_key, ds)
    elif score_type == "bert": 
        scorer = BERTScorer(model_type='bert-base-uncased',device='cuda:0')
        for model in models: 
            for setting_key, ds_values in setting_ds_dict.items():
                for ds in ds_values:     
                    if model == "llama-instruct-few-shot":
                        calculate_bertscore(model, scorer, setting_key, ds, num_examples)
                    else:
                        calculate_bertscore(model, scorer, setting_key, ds)

    elif score_type == "bart": 
        scorer = BARTScorer(device='cuda:0')
        for model in models: 
            for setting_key, ds_values in setting_ds_dict.items():
                for ds in ds_values:     
                    if model == "llama-instruct-few-shot":
                        calculate_bartscore(model, scorer, setting_key, ds, num_examples)
                    else:
                        calculate_bartscore(model, scorer, setting_key, ds)
    
        





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get scores for predicted elaborations.")
    parser.add_argument(
        "--score", 
        type=str, 
        required=True, 
        help="Specify the score type (e.g., 'bleu', 'bert','bart')."
    )
    args = parser.parse_args()
    main(args.score)
