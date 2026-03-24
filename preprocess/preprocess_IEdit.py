from PIL import Image
import os
import json
import numpy as np
from collections import defaultdict

np.random.seed(42)

dir = 'dataset/IEdit'
with open(dir+'/full/full.json', 'r') as fp:
    full_data = json.load(fp)
    
subset_folder = os.path.join(dir, 'train')
if not os.path.exists(subset_folder):
    os.makedirs(subset_folder)
    
subset_folder = os.path.join(dir, 'test')
if not os.path.exists(subset_folder):
    os.makedirs(subset_folder)

meta_data = full_data['metadata']
full_data = full_data['data']

# Create a dictionary to store data for each task
task_data = defaultdict(list)

for item in full_data:
    new_item = {}
    new_item['id'] = item['sample_id']
    new_item['image'] = [os.path.join(dir, 'full/images', img) for img in item['task_instance']['images_path']]
    
    # try:
    #     for img_path in new_item['image']:
    #         image = Image.open(img_path)
    # except Exception as e:
    #     print(e)
    #     print(img_path)
    #     continue
    
    question = item['task_instance']['context']
    for i in range(len(new_item['image'])):
        question = question.replace(f'{{image#{i+1}}}', '<image>')
        question = question.replace(f'{{table#{i+1}}}', '<image>')
    
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
    
    # Add the item to the corresponding task list
    task_data[item['task_instruction_id']].append(new_item)

# Function to split and sample data
def split_and_sample(data, train_size, test_size):
    total_len = len(data)
    test_ratio = test_size / (train_size + test_size)
    
    idx_list = list(range(total_len))
    test_idx = np.random.choice(idx_list, size=int(total_len * test_ratio), replace=False).tolist()
    
    train_data = [item for i, item in enumerate(data) if i not in test_idx]
    test_data = [item for i, item in enumerate(data) if i in test_idx]
    
    if len(train_data) > train_size:
        train_data = np.random.choice(train_data, size=train_size, replace=False).tolist()
    if len(test_data) > test_size:
        test_data = np.random.choice(test_data, size=test_size, replace=False).tolist()
    
    return train_data, test_data

# Process each task
train_data_all = []
test_data_all = []



for task_id, task_items in task_data.items():
    print(task_id, len(task_items))
    print(f"Task {task_id}: {len(task_items)} items")
    train_size = int(len(task_items) * 0.8)  # 80% or 200, whichever is smaller
    test_size = len(task_items) - train_size  # remaining or 50, whichever is smaller
    
    train_task, test_task = split_and_sample(task_items, train_size, test_size)
    
    train_data_all.extend(train_task)
    test_data_all.extend(test_task)

# Shuffle the combined data
np.random.shuffle(train_data_all)
np.random.shuffle(test_data_all)

print(f"Total train items: {len(train_data_all)}")
print(f"Total test items: {len(test_data_all)}")

# Save the data
def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

save_json(train_data_all, os.path.join(dir, 'train', 'dataset-0.json'))
save_json(test_data_all, os.path.join(dir, 'test', 'dataset-0.json'))