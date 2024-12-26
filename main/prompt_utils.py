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

base_prompt_zeroshot_train = """### User: You are an expert in clarifying unclear, complex term or concept in a given text. Your task is to generate exactly ONE short concise explanation sentence (made up of around 10 words or fewer) in plain English for a given context text. The tone should be plain and simple! {}\n### Assistant: {}"""

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

"""

examples_dict = {
    "Definition":[
        {"context_text":"The company faced severe challenges during the economic downturn. Many workers were laid off. The remaining staff had to adapt to a new schedule.",
                 "target_phrase": "laid off",
                 "target_sentence":"Many workers were laid off.",
                  "assistant": "Being laid of means losing your job because the company has to save money",
                 },
        {"context_text":"Japan is known for its rich cultural heritage and advanced technology. Its landscapes range from cherry blossom gardens to towering Mount Fuji.",
                 "target_phrase": "Mount Fuji",
                 "target_sentence":"Its landscapes range from cherry blossom gardens to towering Mount Fuji.",
                  "assistant": "Mount Fuji is the tallest mountain in Japan.",
                 },
    ],
    "Example":[
        {"context_text":"Japan is known for its rich cultural heritage and advanced technology. Its landscapes range from cherry blossom gardens to towering Mount Fuji.",
                 "target_phrase":"cultural heritage",
                 "target_sentence":'Japan is known for its rich cultural heritage and advanced technology.',
                   "assistant":"This heritage includes traditional arts like tea ceremony or calligraphy.",
                 },
        {"context_text":"The zoo is home to many exotic animals. Visitors can see creatures from all over the world. Special tours allow guests to learn more about their habitats and diets.",
                 "target_phrase":"creatures from all over the world",
                 "target_sentence":"Visitors can see creatures from all over the world.",
                   "assistant":"These include lions, pandas, and kangaroos.",
                 },
    ],
    "Background":[
        {"context_text":"The festival is a time for celebration and joy. It is a tradition that has been followed for centuries. But in recent years, rising costs have made it harder for some families to participate fully.",
                 "target_phrase":"a tradition",
                 "target_sentence":"It is a tradition that has been followed for centuries.",
                  "assistant":"Families gather to share meals and exchange gifts. "
                 },
        {"context_text":"The old tree stood in the center of the village. People would meet there to share news and celebrate events.",
                 "target_phrase":"meet there",
                 "target_sentence":"People would meet there to share news and celebrate events.",
                  "assistant":"It was a gathering place for the community."
                 },
    ],
        
    "Reason":[
        {"context_text":"The city decided to shut down all major roads during the storm. Emergency teams worked quickly to clear debris from smaller streets. ",
                 "target_phrase":"shut down all major roads",
                 "target_sentence":"The city decided to shut down all major roads during the storm.",
                "assistant": "Officials worried that the bad weather conditions would cause accidents."
                 },
        {"context_text":"Firefighters worked tirelessly through the night to control the blaze. They focused on protecting nearby homes to prevent further damage.",
                 "target_phrase":"control the blaze",
                 "target_sentence":"Firefighters worked tirelessly through the night to control the blaze.",
                "assistant": "The wind made the fire harder to contain."
                 },
    ],
    "Flow":[
        {"context_text":"The villagers started storing water in large containers. They were preparing for the dry season ahead. Weather experts had predicted unusually low rainfall this year.",
                 "target_phrase":"the dry season",
                 "target_sentence":"They were preparing for the dry season ahead.",
            "assistant":"But why was the dry season expected to be so severe?",
                 },
        {"context_text":"The chef carefully layered the ingredients in the pan. First, the vegetables were sautéed, followed by the spices. Finally, the broth was added to bring everything together.",
                 "target_phrae":"vegetables were sautéed",
                 "target_sentence":"First, the vegetables were sautéed",
            "assistant":"That’s how the dish got its rich and aromatic flavor.",
                 },
    ],
        
    "Result":[
        {"context_text":"One of the most thrilling events in winter sports is ski jumping. Ski jumping is a winter sport where athletes glide down a ramp and jump to achieve maximum distance.",
                 "target_prase":"glide down a ramp and jump",
                 "target_sentence":"Ski jumping is a winter sport where athletes glide down a ramp and jump to achieve maximum distance.",
                  "assistant":"As they glide down, they gain speed, which helps them jump higher into the air."
                 },
        {"context_text":"The storm caused heavy rain throughout the night. Many roads were submerged, making it impossible for cars to pass. Emergency crews worked to rescue stranded families.",
                 "target_phrase":"roads were submerged",
                 "target_sentence":"Many roads were submerged, making it impossible for cars to pass.",
                  "assistant":"Rivers flooded nearby neighborhoods."
                 },
    ],
    "Speculation":[
        {"context_text":"Scientists observed unusual patterns in the migration of birds this year. Many flocks changed their routes unexpectedly. Researchers are studying the phenomenon to understand its cause.",
                 "target_phrase":"changed their routes",
                 "target_sentence":"Many flocks changed their routes unexpectedly.",
                   "assistant":"This could be due to changes in weather or food availability.",
                 },
        {"context_text":"The town experienced a sudden increase in power outages last month. Some residents suggested it might be because of aging infrastructure. Officials promised to investigate the issue thoroughly.",
                 "target_phrase":"increase in power outages",
                 "target_sentence":"The town experienced a sudden increase in power outages last month.",
                   "assistant":" Others believed the recent storms could have damaged power lines.",
                 },
    ]
    
}

examples_dict_from_validation_dataset = {
    "Definition":[
        {"context_text":"Together they slowed down the large group of trucks carrying the machine parts. They were not able to stop the trucks for long, though. This one was carrying a giant water evaporator. After being blocked, it continued on its way when police arrested 20 of the protesters.",
                 "target_phrase": "group of trucks",
                 "target_sentence":"Together they slowed down the large group of trucks carrying the machine parts.",
                  "assistant": "Trucks that travel in groups are known as a convoy.",
                 },
        {"context_text":"She teaches at the University of Utah. In 1974, Wiessner recorded conversations among the Ju/'hoansi Bushmen. They live in a vast area of 124 miles in southwestern Africa. Their lives have changed since the 1970s.",
                 "target_phrase": "Bushmen",
                 "target_sentence":"In 1974, Wiessner recorded conversations among the Ju/'hoansi Bushmen.",
                  "assistant": "The Bushmen are a group of people who hunt animals and gather wild berries and plants to eat.",
                 },
    ],
    "Example":[
        {"context_text":"There are differences in how the increases would work. The differences have to do with how the cost of living would be measured. The minimum wage in Alaska would be based on prices in Alaska. South Dakota would raise the minimum wage based on changes to a national measure of the cost of living.",
                 "target_phrase":"the cost of living",
                 "target_sentence":"The differences have to do with how the cost of living would be measured. ",
                   "assistant":"The cost of living looks at prices for things like food, clothes and housing.",
                 },
        {"context_text":"When Border first started doing art, he worked with paper and clay. A few years ago, he found a dead elk. He loaded the elk into his car, Borders said, laughing. 'I almost got arrested doing this.'",
                 "target_phrase":"elk",
                 "target_sentence":"A few years ago, he found a dead elk.",
                   "assistant":"Elk are similar to deer, but larger.",
                 },
    ],
    "Background":[
        {"context_text":"Scientists studied the birds, called bar-headed geese. They were on their way south for the winter. The scientists followed the birds across the world's tallest mountains. They found that the geese do not fly in a straight line.",
                 "target_phrase":"for the winter",
                 "target_sentence":"They were on their way south for the winter.",
                  "assistant":"Geese often spend the summer in one place and then move on for the winter so they can stay warm."
                 },
        {"context_text":"The light of the fire changed how their bodies made a chemical called melatonin. Firelight let people stay awake after the sun went down.",
                 "target_phrase":"a chemical called melatonin",
                 "target_sentence":"The light of the fire changed how their bodies made a chemical called melatonin.",
                  "assistant":"Melatonin makes people feel sleepy when it gets dark."
                 },
    ],
        
    "Reason":[
        {"context_text":"Three days later, he became sicker and was rushed back to Texas Health Presbyterian Hospital Dallas. He was in a room by himself in the hospital. Duncan was extremely ill. Because doctors did not realize Duncan had Ebola, many are afraid.",
                 "target_phrase":"was in a room by himself",
                 "target_sentence":"He was in a room by himself in the hospital.",
                "assistant": "He must be kept away from the other patients because the disease could spread."
                 },
        {"context_text":"A new program in Chicago found jobs for some teenagers. A study showed that if teens have jobs, the number of violent crimes they do may go down. So they picked teenagers. Then they gave each one a job for eight weeks.",
                 "target_phrase":"gave each one a job",
                 "target_sentence":"Then they gave each one a job for eight weeks.",
                "assistant": "Scientists wanted to see if having a job changes the way somebody acts."
                 },
    ],
    "Opposition":[
        {"context_text":"And jellyfish don't have bones. Their simple bodies look like an open umbrella or a bell.",
                 "target_phrase":"don't have bones",
                 "target_sentence":"And jellyfish don't have bones.",
            "assistant":"They have arms called tentacles.",
                 },
    ],
        
    "Result":[
        {"context_text":"Climate change is a shift in weather patterns. It's thought to be caused in part by humans burning fuels.",
                 "target_phrase":"burning fuels",
                 "target_sentence":"It's thought to be caused in part by humans burning fuels.",
                  "assistant":"That leads to global warming."
                 },
    ],
    "Speculation":[
        {"context_text":"He works at the hospital where Emily was treated. Less government money could mean less experimental therapies and research. The number of specialists in children's hospitals across the country has dropped, he added.",
                 "target_phrase":"less experimental therapies and research",
                 "target_sentence":"Less government money could mean less experimental therapies and research. ",
                   "assistant":"And that could hurt patients, he said.",
                 },
    ]
    
}

"""