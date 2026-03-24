from PIL import Image
import os
import json
import numpy as np
from collections import defaultdict

np.random.seed(42)

dir = 'dataset/MIT-States'
subdirs = ['MIT-States_PropertyCoherence', 'MIT-States_StateCoherence']

def process_dataset(subdir):
    with open(os.path.join(dir, subdir, 'full/full.json'), 'r') as fp:
        full_data = json.load(fp)

    meta_data = full_data['metadata']
    full_data = full_data['data']

    task_data = defaultdict(list)

    for item in full_data:
        new_item = {}
        new_item['id'] = item['sample_id']
        new_item['image'] = [os.path.join(dir, subdir, 'full/images', img) for img in item['task_instance']['images_path']]
        try:
            for img in new_item['image']:
                image = Image.open(img)
        except:
            print(f"Error opening image: {img}")
            continue

        question = item['task_instance']['context']
        choice_list = item['task_instance']['choice_list']
        choice_string = ', '.join(f'{choice_list[i]}' for i in range(len(choice_list)))

        for i in range(len(new_item['image'])):
            question = question.replace(f'{{image#{i+1}}}', '<image>')
            question = question.replace(f'{{table#{i+1}}}', '<image>')

        new_item['conversations'] = [
            {
                "from": "human",
                "value": meta_data['task_instruction'][item['task_instruction_id']] + question + f'\nChoice list:[{choice_string}]. Your answer is:'
            },
            {
                "from": "gpt",
                "value": item['response']
            }
        ]

        task_data[item['task_instruction_id']].append(new_item)

    return task_data

def split_and_sample(data, train_ratio=0.8):
    np.random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

# Process both datasets
all_task_data = {subdir: process_dataset(subdir) for subdir in subdirs}

# Split each task into train and test for each dataset
train_data = []
test_data = []

for subdir, task_data in all_task_data.items():
    for task_id, items in task_data.items():
        print(subdir, task_id, len(items))
        train_task, test_task = split_and_sample(items)
        train_data.extend(train_task)
        test_data.extend(test_task)

# Shuffle and sample final datasets
np.random.shuffle(train_data)
np.random.shuffle(test_data)

if len(train_data) > 10000:
    train_data = np.random.choice(train_data, size=10000, replace=False).tolist()
if len(test_data) > 2000:
    test_data = np.random.choice(test_data, size=2000, replace=False).tolist()

print(f"Final train set size: {len(train_data)}")
print(f"Final test set size: {len(test_data)}")

# Save the data
os.makedirs(os.path.join(dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(dir, 'test'), exist_ok=True)

with open(os.path.join(dir, 'train', 'dataset-0.json'), 'w') as json_file:
    json.dump(train_data, json_file, indent=4)
with open(os.path.join(dir, 'test', 'dataset-0.json'), 'w') as json_file:
    json.dump(test_data, json_file, indent=4)