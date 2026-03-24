from collections import defaultdict
from PIL import Image
from io import BytesIO
import random
import requests
import os
import json
import glob
import shutil
import numpy as np

np.random.seed(42)

dir = 'dataset/NLVR2_feddat'
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

categories = {
    "cardinality_existential": ["Cardinality (hard)", "Cardinality (soft)", "Existential"],
    "coordination_coreference": ["Coordination", "Coreference"],
    "spatial_comparative": ["Spatial Relations", "Comparative"],
    "presupposition_negation_universal": ["Presupposition", "Negation", "Universal"]
    }

def categorize_sample(sample):
    context = sample["task_instance"]["context"].lower()

    if any(word in context for word in ["not", "never", "no longer", "fail to", "lack of", "without", "absent", "none", "nothing", "nowhere", "nobody", "neither", "hardly", "scarcely", "barely", "avoid", "miss", "except", "prevent", "forbid", "deny", "refuse", "stop", "restrict", "prohibit", "limit", "omit", "fail", "neglect", "every", "each", "always", "whenever", "whichever"]):
        return "presupposition_negation_universal"
    elif any(word in context for word in ["near", "left", "right", "closer", "far", "next to", "beside", "above", "below", "between", "in front of", "behind"]):
        return "spatial_comparative"
    elif any(word in context for word in ["at most", "at least", "more than", "no more than", "each", "fewer than", "exactly"]):
        return "cardinality_existential"
    elif any(word in context for word in ["and", "each other", "only one", "all", "both", "either"]):
        return "coordination_coreference"

    return random.choice(list(categories.keys()))  

subtasks = defaultdict(list)
#{key: [] for key in categories.keys()}

for sample in full_data:
    category = categorize_sample(sample)
    subtasks[category].append(sample)

total_len = len(full_data)

train_test_ratio = 0.2


for task_idx, key in enumerate(sorted(list(subtasks.keys()))):
    full_data = subtasks[key]
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

    print(key, "original len(train_json_data)", len(train_json_data))
    print(key, "original len(test_json_data)", len(test_json_data))

    if len(train_json_data) > 10000:
        train_json_data = np.random.choice(train_json_data, size=10000, replace=False).tolist()
    if len(test_json_data) > 1000:
        test_json_data = np.random.choice(test_json_data, size=1000, replace=False).tolist()

    print(key, "sampled len(train_json_data)", len(train_json_data))
    print(key, "sampled len(test_json_data)", len(test_json_data))
    
    with open(f'{dir}/train/dataset-{task_idx}.json', 'w') as json_file:
        json.dump(train_json_data, json_file, indent=4)
    with open(f'{dir}/test/dataset-{task_idx}.json', 'w') as json_file:
        json.dump(test_json_data, json_file, indent=4)
    
