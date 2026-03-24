import shutil
import os
import json
import random
from collections import defaultdict

base_dir = 'dataset/SNLI-VE'
dir_type = 'train'
max_num = 10000 if dir_type == 'train' else 1000
input_file = f"{base_dir}/snli_ve_{dir_type}.jsonl"
output_dir = f"{base_dir}/{dir_type}"
final_data = defaultdict(list)

base_prompt = 'Question: Given the image as a premise and the given sentence as a hypothesis, determine their relationship: entailment, neutral, or contradiction, based on the relationship conveyed by the given premise image and hypothesis. Entailment: The image provides enough evidence to conclude the hypothesis is true. Contradiction: The image provides enough evidence to conclude the hypothesis is false. Neutral: The image lacks sufficient evidence to confirm or refute the hypothesis. Choice list: [entailment, neutral, contradiction]. Your answer is:'

def classify_subtask(entry):
    sentence1 = entry["sentence1"].lower()
    
    object_keywords = ["man", "woman", "lady", "boy", "girl", "dog", "cat", "car", "tree"]
    action_keywords = ["running", "jumping", "playing", "walking", "sitting", "standing"]
    relation_keywords = ["with", "beside", "holding", "next to", "near"]
    commonsense_keywords = ["thinking", "knowing", "believing", "teaching", "learning", "pretending",
    "guessing", "assuming", "expecting", "predicting", "understanding",
    "happy", "sad", "excited", "worried", "scared", "nervous", "angry",
    "wanting", "planning", "deciding", "trying", "intending", "hoping"]
    # ["thinking", "knowing", "believing", "teaching", "learning", "pretending"]
    
    if any(word in sentence1 for word in object_keywords):
        if any(word in sentence1 for word in commonsense_keywords):
            return "Commonsense & High-Level Inference"
        return "Object-Centric Entailment"
    elif any(word in sentence1 for word in action_keywords):
        if any(word in sentence1 for word in commonsense_keywords):
            return "Commonsense & High-Level Inference"
        return "Action-Based Entailment"
    elif any(word in sentence1 for word in relation_keywords):
        if any(word in sentence1 for word in commonsense_keywords):
            return "Commonsense & High-Level Inference"
        return "Scene Context & Object Relations"
    elif any(word in sentence1 for word in commonsense_keywords):
        return "Commonsense & High-Level Inference"
    else:
        return "Uncategorized"

def generate_question(subtask):
    questions = {
        "Object-Centric Entailment": "What is the main object in the image?",
        "Action-Based Entailment": "What action is happening in the image?",
        "Scene Context & Object Relations": "What is the relationship between the objects in the image?",
        "Commonsense & High-Level Inference": "What can we infer from the scene?"
    }
    return questions.get(subtask, "What is happening in the image?")


with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line.strip())
        subtask = classify_subtask(entry)
        
        formatted_entry = {
            "id": entry["pairID"],
            "image": f"{base_dir}/images/{entry['Flickr30K_ID']}.jpg",
            "subtask": subtask,
            "conversations": [
                {
                    "from": "human",
                    "value": f"Premise Image: <image>\n Hypothesis: {entry['sentence2']}\n {base_prompt}" 
                },
                {
                    "from": "gpt",
                    "value": entry['gold_label']
                }
            ]
        }
        final_data[subtask].append(formatted_entry)

os.makedirs(f'{base_dir}/{dir_type}', exist_ok=True)
os.makedirs(f'{base_dir}/images', exist_ok=True)
del final_data['Uncategorized']

for idx, subtask in enumerate(sorted(list(final_data.keys()))):
    sampled_subtask = random.sample(final_data[subtask], min(len(final_data[subtask]), max_num))
    print(f"{subtask} original {len(final_data[subtask])} sampled {len(sampled_subtask)}")
    random.shuffle(sampled_subtask)
    with open(f"{base_dir}/{dir_type}/dataset-{idx//4}{idx%4}.json", 'w', encoding='utf-8') as f:
        json.dump(sampled_subtask, f, indent=4)

    for item in sampled_subtask:
        source_path = item["image"].replace('images', 'raw_images/images')
        target_path = item["image"]
        if os.path.exists(source_path):
            shutil.copy(source_path, target_path)
        else:
            print(source_path)

print("Subtask-based SNLI-VE dataset generated successfully.")

