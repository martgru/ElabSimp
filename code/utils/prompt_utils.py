versions = {
    "n3":['Definition','Example','Background'],
    "n6":['Definition','Example','Background', 'Supplementation', 'Analogy', 'Speculation'],
    "n9":['Definition', 'Example', 'Analogy', 'Background', 'Reason', 'Contrast', 'Result', 'Speculation', 'Supplementation']
}


# for llama-instruct
def insert_examples(examples_dict, setting):
    """
    Insert examples
    """
    if setting == "masked":
        examples_string = "\n".join(
            f"context text: '{example['masked']}'\nAssistant: '{example['assistant']}'\n"
            for category in examples_dict.values()
            for example in category  
        )
    elif setting == "target-phrase":
        examples_string = "\n".join(
            f"context text: '{example['context_text']}'\ntarget_phrase='{example['target_phrase']}'\nAssistant: '{example['assistant']}'\n"
            for category in examples_dict.values()
            for example in category  
        )
    elif setting == "target-sent":
        examples_string = "\n".join(
            f"context text: '{example['context_text']}'\ntarget_sentence='{example['target_sentence']}'\nAssistant: '{example['assistant']}'\n"
            for category in examples_dict.values()
            for example in category  
        )
    elif setting == "target-sent-target":
        examples_string = "\n".join(
            f"context text: '{example['context_text']}'\ntarget_sentence='{example['target_sentence']}\ntarget_phrase='{example['target_phrase']}'\nAssistant: '{example['assistant']}'\n"
            for category in examples_dict.values()
            for example in category  
        )
    else:
        examples_string = "\n".join(
            f"context text: '{example['context_text']}'\nAssistant: '{example['assistant']}'\n"
            for category in examples_dict.values()
            for example in category  
        )
    return examples_string

def create_user_message(context, setting, target=None, target_sentence=None):
    if setting == "masked":
        return f"Return the explanation sentence that could replace the `<explanatory sentence>` tag in the following text: '{context}'."
        
    elif target and setting == "target-phrase":
            return f"Return the explanation sentence for the following context text: '{context}'. The explanation sentence should specifically clarify the target_phrase={target}."

    elif target and setting == "target-sent":
            return f"Return the explanation sentence for the following context text: '{context}'. The explanation sentence should specifically clarify the target_sentence={target}"

    elif target and target_sentence and setting == "target-sent-target":
            return f"Return the explanation sentence for the following context text: '{context}'. The explanation sentence should specifically clarify the target_sentence={target_sentence} by referring to the target_phrase={target}."
    else:
        return f"Return an explanation sentence for the following context text: '{context}'."

# for llama-instruct
def formatting_prompt_func(examples, EOS, base_prompt, setting, num_examples=None, test=False):
    
    contexts = examples["source_text"]
    elab_sentences = examples["elaboration_sentence"]
    texts = []
    if num_examples:
        filtered_dict = {key: value for key, value in examples_dict.items() if key in versions[num_examples]}
    
    if setting == "target-phrase":
        targets = examples["target_sentence_target"]
        if num_examples:
            for context, target in zip(contexts, targets):
                text = base_prompt.format(insert_examples(filtered_dict, setting), create_user_message(context, setting, target)) 
                texts.append(text)

        elif num_examples is None and test:
            for context, target in zip(contexts, targets):
                text = base_prompt.format(create_user_message(context, setting, target)) 
                texts.append(text)
        else:
            for context, target, elab_sent in zip(contexts, targets, elab_sentences):
                text = base_prompt.format(create_user_message(context, setting, target), elab_sent) + EOS
                texts.append(text)
        return texts
        
    elif setting == "target-sent":
        target_sents = examples["target_sentence_4o"]
        if num_examples:
            for context, target_sent in zip(contexts, target_sents):
                text = base_prompt.format(insert_examples(filtered_dict, setting), create_user_message(context, setting, target=None, target_sentence=target_sent)) 
                texts.append(text)

        elif num_examples is None and test:
            for context, target_sent in zip(contexts, target_sents):
                text = base_prompt.format(create_user_message(context, setting, target=None, target_sentence=target_sent)) 
                texts.append(text)
        else: 
            for context, target_sent, elab_sent in zip(contexts, target_sents, elab_sentences):
                text = base_prompt.format(create_user_message(context, setting, target=None, target_sentence=target_sent),elab_sent) + EOS
                texts.append(text)
        return texts
        
    elif setting == "target-sent-target":
        targets = examples["target_sentence_target"]
        target_sents = examples["target_sentence_4o"]
        if num_examples:
            for context, target, target_sent in zip(contexts, targets, target_sents):
                text = base_prompt.format(insert_examples(filtered_dict, setting), create_user_message(context, setting, target, target_sent)) 
                texts.append(text)

        elif num_examples is None and test:
            for context, target, target_sent in zip(contexts, targets, target_sents):
                text = base_prompt.format(create_user_message(context, setting, target, target_sent)) 
                texts.append(text)
        else:
            for context, target, target_sent, elab_sent in zip(contexts, targets, target_sents, elab_sentences):
                text = base_prompt.format(create_user_message(context, setting, target, target_sent),elab_sent) + EOS
                texts.append(text)
        return texts
        
    else:
        # base & masked
        if num_examples:
            for context in contexts:
                text = base_prompt.format(insert_examples(filtered_dict, setting),create_user_message(context, setting)) 
                texts.append(text)

        elif num_examples is None and test:
            for context in contexts:
                text = base_prompt.format(create_user_message(context, setting)) 
                texts.append(text)
        else:
            for context, elab_sent in zip(contexts,elab_sentences):
                text = base_prompt.format(create_user_message(context, setting), elab_sent) + EOS
                texts.append(text)     
        return texts



import random
def insert_random_examples(examples_dict, num_examples=3):
    """
    Insert random examples from the examples dictionary.
    """
    # flatten the examples from all categories into a single list
    all_examples = [
        f"context text: '{example['context_text']}'\nAssistant: '{example['assistant']}'\n"
        for category in examples_dict.values()
        for example in category
    ]
    
    # select random examples
    selected_examples = random.sample(all_examples, min(num_examples, len(all_examples)))
    
    # join the selected examples into a single string
    return "\n".join(selected_examples)


base_prompt_fewshot = """### User: You are an expert in clarifying unclear,complex term or concept in a given text. Your task is to generate exactly ONE short concise explanation sentence (made up of around 10 words or fewer) in plain English for a given context text. The tone should be plain and simple! Do not add any comments to your answer! 
For example:\n
{}
{}\n### Assistant:"""

base_prompt_zeroshot_train = """### User: You are an expert in clarifying unclear, complex terms or concepts in a given text. Your task is to generate exactly ONE short concise explanation sentence (made up of around 10 words or fewer) in plain English for a given context text. The tone should be plain and simple! {}\n### Assistant: {}"""

base_prompt_zeroshot_test = """### User: You are an expert in clarifying unclear, complex term or concept in a given text. Your task is to generate exactly ONE short concise explanation sentence (made up of around 10 words or fewer) in plain English for a given context text. The tone should be plain and simple! {}\n### Assistant:"""


examples_dict = {
    "Definition":[
        {"context_text":"She teaches at the University of Utah. In 1974, Wiessner recorded conversations among the Ju/'hoansi Bushmen. They live in a vast area of 124 miles in southwestern Africa. Their lives have changed since the 1970s.",
                 "target_phrase": "Bushmen",
                 "target_sentence":"In 1974, Wiessner recorded conversations among the Ju/'hoansi Bushmen.",
                 "masked":"She teaches at the University of Utah. In 1974, Wiessner recorded conversations among the Ju/'hoansi Bushmen. <explanatory sentence> They live in a vast area of 124 miles in southwestern Africa. Their lives have changed since the 1970s.",
                  "assistant": "The Bushmen are a group of people who hunt animals and gather wild berries and plants to eat.",
                 },
    ],
    "Example":[
        {"context_text":"There are differences in how the increases would work. The differences have to do with how the cost of living would be measured. The minimum wage in Alaska would be based on prices in Alaska. South Dakota would raise the minimum wage based on changes to a national measure of the cost of living.",
                 "target_phrase":"the cost of living",
                 "target_sentence":"The differences have to do with how the cost of living would be measured. ",
                 "masked":"There are differences in how the increases would work. The differences have to do with how the cost of living would be measured. <explanatory sentence> The minimum wage in Alaska would be based on prices in Alaska. South Dakota would raise the minimum wage based on changes to a national measure of the cost of living.",
                   "assistant":"The cost of living looks at prices for things like food, clothes and housing.",
                 },
    ],
    "Analogy": [
        {"context_text":"When Border first started doing art, he worked with paper and clay. A few years ago, he found a dead elk. He loaded the elk into his car, Borders said, laughing. 'I almost got arrested doing this.'",
                 "target_phrase":"elk",
                 "target_sentence":"A few years ago, he found a dead elk.",
                 "masked":"When Border first started doing art, he worked with paper and clay. A few years ago, he found a dead elk. <explanatory sentence> He loaded the elk into his car, Borders said, laughing. 'I almost got arrested doing this.'",
                   "assistant":"Elk are similar to deer, but larger.",
                 },
    ],
    "Background":[
        {"context_text":"The light of the fire changed how their bodies made a chemical called melatonin. Firelight let people stay awake after the sun went down.",
                 "target_phrase":"a chemical called melatonin",
                 "target_sentence":"The light of the fire changed how their bodies made a chemical called melatonin.",
                 "masked":"The light of the fire changed how their bodies made a chemical called melatonin. <explanatory sentence> Firelight let people stay awake after the sun went down.",
                  "assistant":"Melatonin makes people feel sleepy when it gets dark."
                 },
    ],
        
    "Reason":[
        {"context_text":"Three days later, he became sicker and was rushed back to Texas Health Presbyterian Hospital Dallas. He was in a room by himself in the hospital. Duncan was extremely ill. Because doctors did not realize Duncan had Ebola, many are afraid.",
                 "target_phrase":"was in a room by himself",
                 "target_sentence":"He was in a room by himself in the hospital.",
                 "masked":"Three days later, he became sicker and was rushed back to Texas Health Presbyterian Hospital Dallas. He was in a room by himself in the hospital. <explanatory sentence> Duncan was extremely ill. Because doctors did not realize Duncan had Ebola, many are afraid.",
                "assistant": "He must be kept away from the other patients because the disease could spread."
                 },
        
    ],
    "Contrast":[
        {"context_text":"And jellyfish don't have bones. Their simple bodies look like an open umbrella or a bell.",
                 "target_phrase":"don't have bones",
                 "target_sentence":"And jellyfish don't have bones.",
                 "masked":"And jellyfish don't have bones. <explanatory sentence> Their simple bodies look like an open umbrella or a bell.",
            "assistant":"They have arms called tentacles.",
                 },
    ],
        
    "Result":[
        {"context_text":"Climate change is a shift in weather patterns. It's thought to be caused in part by humans burning fuels.",
                 "target_phrase":"burning fuels",
                 "target_sentence":"It's thought to be caused in part by humans burning fuels.",
                 "masked":"Climate change is a shift in weather patterns. It's thought to be caused in part by humans burning fuels.  <explanatory sentence>",
                  "assistant":"That leads to global warming."
                 },
    ],
    "Speculation":[
        {"context_text":"He works at the hospital where Emily was treated. Less government money could mean less experimental therapies and research. The number of specialists in children's hospitals across the country has dropped, he added.",
                 "target_phrase":"less experimental therapies and research",
                 "target_sentence":"Less government money could mean less experimental therapies and research. ",
                 "masked":"He works at the hospital where Emily was treated. Less government money could mean less experimental therapies and research. <explanatory sentence> The number of specialists in children's hospitals across the country has dropped, he added.",  
                   "assistant":"And that could hurt patients, he said.",
                 },
    ],
    "Supplementation":[
	    {"context_text":"She is mystery writer Agatha Christie. J.K. Rowling is the best-selling author of recent memory. Yet no woman has been chosen to be put on British money.",
                 "target_phrase":"J.K. Rowling",
                 "target_sentence":"J.K. Rowling is the best-selling author of recent memory.",
                 "masked": "She is mystery writer Agatha Christie. J.K. Rowling is the best-selling author of recent memory. <explanatory sentence>  Yet no woman has been chosen to be put on British money.", 
                   "assistant":"She created Harry Potter."
                   },
    ]

}