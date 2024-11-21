import json
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import re

class LlamaAssistant:
    def __init__(self, model_name: str, cache_dir: str = "../models/llama/"):
        # quantization config
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=self.quantization_config,
            cache_dir=cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            #early_stopping=True, 
            max_new_tokens=32,
            #min_new_tokens=6,
            #num_beams = 4,
            #length_penalty=1.2,
            return_full_text=False,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )

    def create_prompt(self, system_prompt: str, context: str) -> str:
        
        messages = [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": "Got it!"},
            {"role": "user", "content": context}
        ]
        return messages

    def create_extended_prompt(self, system_prompt: str, context: str, target: str) -> str:
        # ELAB AND CONTEXT TOGETHER
        #combined_content = f"Text: {context}\nExplanation sentence: {elaboration_sent}"
        # ELAB AND CONTEXT SEPERATELY
        #combined_content = f"Context: {context}\nExplanation sentence: {elaboration_sent}"
        # ELAB TARGET PROVIDED
        combined_content = f"Context: {context}\nExplanation target: {target}"
        
        # ZERO-SHOT for elaboration generation
        messages = [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": "Got it!"},
            {"role": "user", "content": combined_content}
        ]
        
        """
        # FEW-SHOT for elaboration generation
        messages = [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": "Got it!"},
            # Example 1
            {"role": "user", "content": "Text: Solar panels are becoming increasingly popular. These renewable energy sources can significantly reduce electricity costs for households. They are environmentally friendly.\nExplanation sentence:These renewable energy sources can significantly reduce electricity costs for households." },
            {"role": "assistant", "content": '"target": "solar panels"'},
            # Example 2
            {"role": "user", "content": "Text:There are several famous landmarks in Paris. For instance, the Eiffel Tower and the Louvre Museum attract millions of visitors annually.\nExplanation sentence:For instance, the Eiffel Tower and the Louvre Museum attract millions of visitors annually. " },
            {"role": "assistant", "content": '"target": "famous landmarks in Paris"'},
            {"role":"user", "content":"Text:Volcanoes are natural openings in the Earth's crust. They allow molten rock, gases, and ash to escape from below the surface. This process has shaped many of the world's landscapes over millions of years.\nExplanation sentence:Volcanoes are natural openings in the Earth's crust. "},
            {"role":"assistant","content": '"target": "volcanoes"'},
            {"role": "user", "content": combined_content}
        ]
        """
        """
        # FEW-SHOT for target identification
        messages = [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": "Got it!"},
            # Example 1
            {"role": "user", "content": "Text: Solar panels are becoming increasingly popular. These renewable energy sources can significantly reduce electricity costs for households. They are environmentally friendly.\nExplanation sentence:These renewable energy sources can significantly reduce electricity costs for households." },
            {"role": "assistant", "content": '"target": "solar panels"'},
            # Example 2
            {"role": "user", "content": "Text:There are several famous landmarks in Paris. For instance, the Eiffel Tower and the Louvre Museum attract millions of visitors annually.\nExplanation sentence:For instance, the Eiffel Tower and the Louvre Museum attract millions of visitors annually. " },
            {"role": "assistant", "content": '"target": "famous landmarks in Paris"'},
            {"role":"user", "content":"Text:Volcanoes are natural openings in the Earth's crust. They allow molten rock, gases, and ash to escape from below the surface. This process has shaped many of the world's landscapes over millions of years.\nExplanation sentence:Volcanoes are natural openings in the Earth's crust. "},
            {"role":"assistant","content": '"target": "volcanoes"'},
            {"role": "user", "content": combined_content}
        ]"""
        return messages

    def generate_explanation(self, system_prompt: str, context: str, target: str) -> dict:
        """Generate a single explanation sentence given a context."""
        prompt = self.create_extended_prompt(system_prompt, context, target)
        response = self.generator(prompt)
        return response

    def find_explanation_target(self, system_prompt: str, context: str, elaboration_sent: str ) -> dict:
        """Generate a single explanation sentence given a context."""
        prompt = self.create_extended_prompt(system_prompt, context, elaboration_sent)
        response = self.generator(prompt)
        return response


    @staticmethod
    def extract_response(response) -> tuple:
        """Extract the sentence and target from the model's response."""
        generated_text = response[0]["generated_text"]

        def fix_json_format(generated_text):
            
            # remove any extra commas after the "sentence" field or before the "target" field
            generated_text = re.sub(r',\s*,', ',', generated_text)
            generated_text = re.sub(r'(:\s*".+?")\s*,\s*,', r'\1,', generated_text)
            # remove unnecessary escape characters for single quotes inside double-quoted strings
            generated_text = re.sub(r'\\\'', "'", generated_text)
            # fix missing colons between keys and values (e.g., "target" "value")
            generated_text = re.sub(r'("sentence"|\"target")\s*(")', r'\1: \2', generated_text)
            # wrap unquoted target values with double quotes
            generated_text = re.sub(r'("target":\s*)([^\s"][^,}]*)', r'\1"\2"', generated_text)
            # correct any malformed keys like "target):" or "target}:"
            generated_text = re.sub(r'("target)\)(:)', r'\1":', generated_text)  
            generated_text = re.sub(r'("target)}(:)', r'\1":', generated_text)
            # ddd missing closing brace if necessary
            #if generated_text.count("{") > generated_text.count("}"):
                #generated_text += "}"
            # add a comma between "sentence" and "target" if missing
            if '"sentence"' in generated_text and '"target"' in generated_text:
                generated_text = re.sub(r'("sentence":\s*".+?")\s*("target":)', r'\1, \2', generated_text)
            
            return generated_text

        def postprocess(text):
            # remove any leading or trailing symbols like quotes, brackets, colons, braces, parentheses, etc.
            cleaned_text = re.sub(r'^[\s"\'{}()\[\]:,]*|[\s"\'{}()\[\]:,]*$', '', text)
            
            # remove any embedded unnecessary characters (extra spaces are preserved within the text)
            cleaned_text = re.sub(r'[{}()\[\]:]+', '', cleaned_text)
            
            return cleaned_text

        def extract_target(generated_text):
            # search for "sentence" and "target" in the text
            #sentence_match = re.search(r'sentence', generated_text)
            target_match = re.search(r'target', generated_text)
            
            # get the start and end positions of the keywords
            #sentence_part_start = sentence_match.end()
            #sentence_part_end = target_match.start()
            target_part_start = target_match.end()
            
            # extract everything between "sentence" and "target"
            #sentence = generated_text[sentence_part_start:sentence_part_end].strip()
            
            # extract everything after "target"
            target = generated_text[target_part_start:].strip()
            
            # postprocess the extracted values
            #cleaned_sentence = postprocess(sentence)
            cleaned_target = postprocess(target)
        
            #return cleaned_sentence, cleaned_target
            return cleaned_target
        
        def extract_sentence(generated_text):
            # search for "sentence" and "target" in the text
            sentence_match = re.search(r'sentence', generated_text)
            
            # get the start and end positions of the keywords
            sentence_part_start = sentence_match.end()
            
            # extract everything after "sentence"
            sentence = generated_text[sentence_part_start:].strip()
            
            # postprocess the extracted values
            cleaned_sentence = postprocess(sentence)        
            #return cleaned_sentence, cleaned_target
            return cleaned_sentence

        #formatted_generated_text = fix_json_format(generated_text)

        try:
            # attempt to load the fixed JSON
            data = json.loads(generated_text)
            sentence = data.get("sentence", "")
            #target = data.get("target", "")
                
        except json.JSONDecodeError:
            #print("Warning: Could not fix JSON format:", formatted_generated_text)
            #sentence, target = extract_sentence_and_target(generated_text)
            sentence = extract_sentence(generated_text)

        return sentence        
        #return sentence, target





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