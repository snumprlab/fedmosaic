from PIL import Image
from io import BytesIO
import requests
import os
import json
import glob
import shutil
import numpy as np

np.random.seed(42)

dir = 'dataset/NLVR2'
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

# split into two

total_len = len(full_data)

full_datas = [full_data[:total_len//2], full_data[total_len//2:]]

train_test_ratio = 0.2

task_idx = 0
for full_data in full_datas:
    total_len = len(full_data)
    idx_list = list(range(total_len))
    test_idx = np.random.choice(idx_list, size=int(total_len*train_test_ratio), replace=False).tolist()

    train_json_data = []
    test_json_data = []

    for idx in range(total_len):
        item = full_data[idx]
        new_item = {}
        new_item['id'] = item['sample_id']
        new_item['image'] = [os.path.join(dir, 'full/images', img) for img in item['task_instance']['images_path']]
        invalid = False
        # for img in new_item['image']:
        #     image = Image.open(img)
        #     if image.width == 1 and image.height == 1:
        #         invalid = True
        #         break
        # if invalid:
        #     continue
        
        question = item['task_instance']['context']
        for i in range(len(new_item['image'])):
            rmv_i = '{image#%d}'% (i+1)
            rmv_t = '{table#%d}'% (i+1)
            question = question.replace(rmv_i, '<image>')
            question = question.replace(rmv_t, '<image>')
        
        new_item['conversations'] = [
            {
                "from": "human",
                "value": meta_data['task_instruction'][item['task_instruction_id']] + question + '\nChoice list:[True, False]. Your answer is:'
            },
            {
                "from": "gpt",
                "value": item['response']
            }
        ]
        
        if idx in test_idx:
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
    
    task_idx += 1