import numpy as np
import os
import json
from PIL import Image

np.random.seed(42)

dir = 'dataset/Fashion200K'
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

# Create four sublists for different tasks
task1_data = []  # Guessing number
task2_data = []  # T/F with two input images
task3_data = []  # T/F with three input images
task4_data = []  # T/F with four input images

for item in full_data:
    new_item = {}
    new_item['id'] = item['sample_id']
    new_item['image'] = [os.path.join(dir, 'full/images', img) for img in item['task_instance']['images_path']]
    
    # Check if all images exist
    # try:
    #     for img_path in new_item['image']:
    #         image = Image.open(img_path)
    # except:
    #     print(f"Image not found: {img_path}")
    #     continue
    
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
    
    # Categorize the item based on task type
    if any(choice.isdigit() for choice in choice_list):
        task1_data.append(new_item)
    elif len(new_item['image']) == 2 and 'True' in choice_list and 'False' in choice_list:
        task2_data.append(new_item)
    elif len(new_item['image']) == 3 and 'True' in choice_list and 'False' in choice_list:
        task3_data.append(new_item)
    elif len(new_item['image']) == 4 and 'True' in choice_list and 'False' in choice_list:
        task4_data.append(new_item)

# Function to split data into train and test sets
def split_data(data, test_ratio=0.2):
    total_len = len(data)
    idx_list = list(range(total_len))
    test_idx = np.random.choice(idx_list, size=int(total_len * test_ratio), replace=False).tolist()
    
    train_data = [item for i, item in enumerate(data) if i not in test_idx]
    test_data = [item for i, item in enumerate(data) if i in test_idx]
    
    return train_data, test_data

# Split each task data into train and test sets
train_task1, test_task1 = split_data(task1_data)
train_task2, test_task2 = split_data(task2_data)
train_task3, test_task3 = split_data(task3_data)
train_task4, test_task4 = split_data(task4_data)

# Combine all train and test data
# train_data_0 = train_task1 + train_task4
# test_data_0 = test_task1 + test_task4
# train_data_1 = train_task2 + train_task3
# test_data_1 = test_task2 + test_task3

# Shuffle the combined data
# np.random.shuffle(train_data)
# np.random.shuffle(test_data)

print(f"Task 1 (Guessing number) items: {len(task1_data)} {len(train_task1)} {len(test_task1)}")
print(f"Task 2 (T/F with two images) items: {len(task2_data)} {len(train_task2)} {len(test_task2)}")
print(f"Task 3 (T/F with three images) items: {len(task3_data)} {len(train_task3)} {len(test_task3)}")
print(f"Task 4 (T/F with four images) items: {len(task4_data)} {len(train_task4)} {len(test_task4)}")

# print(f"Total train items: {len(train_data)}")
# print(f"Total test items: {len(test_data)}")

# print(len(train_json_data))
# print(len(test_json_data))

# if len(train_json_data) > 10000:
#     train_json_data = np.random.choice(train_json_data, size=10000, replace=False).tolist()
# if len(test_json_data) > 2000:
#     test_json_data = np.random.choice(test_json_data, size=2000, replace=False).tolist()

# print(len(train_data_0))
# print(len(test_data_0))
# print(len(train_data_1))
# print(len(test_data_1))

# with open(f'{dir}/train/dataset-0.json', 'w') as json_file:
#     json.dump(train_data_0, json_file, indent=4)
# with open(f'{dir}/test/dataset-0.json', 'w') as json_file:
#     json.dump(test_data_0, json_file, indent=4)

# with open(f'{dir}/train/dataset-1.json', 'w') as json_file:
#     json.dump(train_data_1, json_file, indent=4)
# with open(f'{dir}/test/dataset-1.json', 'w') as json_file:
#     json.dump(test_data_1, json_file, indent=4)

with open(f'{dir}/train/dataset-0.json', 'w') as json_file:
    json.dump(train_task1, json_file, indent=4)
with open(f'{dir}/test/dataset-0.json', 'w') as json_file:
    json.dump(test_task1, json_file, indent=4)

with open(f'{dir}/train/dataset-1.json', 'w') as json_file:
    json.dump(train_task2, json_file, indent=4)
with open(f'{dir}/test/dataset-1.json', 'w') as json_file:
    json.dump(test_task2, json_file, indent=4)
    
with open(f'{dir}/train/dataset-2.json', 'w') as json_file:
    json.dump(train_task3, json_file, indent=4)
with open(f'{dir}/test/dataset-2.json', 'w') as json_file:
    json.dump(test_task3, json_file, indent=4)

with open(f'{dir}/train/dataset-3.json', 'w') as json_file:
    json.dump(train_task4, json_file, indent=4)
with open(f'{dir}/test/dataset-3.json', 'w') as json_file:
    json.dump(test_task4, json_file, indent=4)