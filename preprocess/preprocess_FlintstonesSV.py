import os
import json
import numpy as np
from collections import defaultdict

np.random.seed(42)

dir = 'dataset/FlintstonesSV'
with open(dir+'/full/full.json', 'r') as fp:
    full_data = json.load(fp)

meta_data = full_data['metadata']
full_data = full_data['data']

def extract_descriptions(context):
    # Split the context by '{image#' and remove the empty first element
    parts = context.split('{image#')[1:]
    
    # Extract the caption part from each split
    descriptions = []
    for part in parts[:-1]:
        # Find the position of the colon
        colon_pos = part.find(':')
        if colon_pos != -1:
            # Extract the text after the colon and before the next '{image#' or end of string
            description = part[colon_pos+1:].strip()
            # Remove the image number and colon from the beginning
            description = description.split(':', 1)[-1].strip()
            descriptions.append(description)
    
    return descriptions

def process_sample(item):
    context_descriptions = extract_descriptions(item['task_instance']['context'])
    response = item['response'].strip()
    return context_descriptions + [response]

# Create a map of descriptions to sample indices
description_to_samples = defaultdict(set)

for idx, item in enumerate(full_data):
    for description in process_sample(item):
        description_to_samples[description].add(idx)

# Create groups of samples that share descriptions
sample_groups = []
processed_samples = set()

for idx, item in enumerate(full_data):
    if idx in processed_samples:
        continue
    
    group = set()
    to_process = [idx]
    
    while to_process:
        current_idx = to_process.pop()
        if current_idx in processed_samples:
            continue
        
        group.add(current_idx)
        processed_samples.add(current_idx)
        
        for description in process_sample(full_data[current_idx]):
            related_samples = description_to_samples[description]
            to_process.extend(related_samples - processed_samples)
    # if len(sample_groups) == 6:
    #     breakpoint()
    sample_groups.append(list(group))

# Shuffle the groups
# np.random.shuffle(sample_groups)
# Split groups into train and test, aiming for about 80% train
train_samples = []
test_samples = []
target_train_ratio = 0.8
current_ratio = 0
# breakpoint()
# for group in sample_groups:
#     if current_ratio < target_train_ratio:
#         train_samples.extend(group)
#     else:
#         test_samples.extend(group)
    
#     current_ratio = len(train_samples) / (len(train_samples) + len(test_samples))
train_samples.extend(sample_groups[0])
for i in range(len(sample_groups)):
    if i == 0:
        continue
    test_samples.extend(sample_groups[i])
current_ratio = len(train_samples) / (len(train_samples) + len(test_samples))
print(f"Train set size: {len(train_samples)}, Test set size: {len(test_samples)}")
print(f"Actual train ratio: {current_ratio:.2f}")

# Create directories
for subset in ['train', 'test']:
    subset_folder = os.path.join(dir, subset)
    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)

task_idx = 0
train_json_data = []
test_json_data = []

# Process and save data
for idx, item in enumerate(full_data):
    new_item = {}
    new_item['id'] = item['sample_id']
    new_item['image'] = [os.path.join(dir, 'full/images', img) for img in item['task_instance']['images_path']]
    question = item['task_instance']['context']
    for i in range(len(new_item['image'])):
        rmv_i = '{image#%d}' % (i+1)
        rmv_t = '{table#%d}' % (i+1)
        question = question.replace(rmv_i, '<image>')
        question = question.replace(rmv_t, '<image>')
    
    new_item['conversations'] = [
        {
            "from": "human",
            "value": meta_data['task_instruction'][item['task_instruction_id']] + question
        },
        {
            "from": "gpt",
            "value": item['response']
        }
    ]
    
    if idx in test_samples:
        test_json_data.append(new_item)
    else:
        train_json_data.append(new_item)


print(len(train_json_data))
print(len(test_json_data))

if len(train_json_data) > 10000:
    train_json_data = np.random.choice(train_json_data, size=10000, replace=False).tolist()
if len(test_json_data) > 2000:
    test_json_data = np.random.choice(test_json_data, size=2000, replace=False).tolist()

print(len(train_json_data))
print(len(test_json_data))

with open(f'{dir}/train/dataset-{task_idx}.json', 'w') as json_file:
    json.dump(train_json_data, json_file, indent=4)
with open(f'{dir}/test/dataset-{task_idx}.json', 'w') as json_file:
    json.dump(test_json_data, json_file, indent=4)