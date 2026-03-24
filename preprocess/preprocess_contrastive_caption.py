from datasets import load_dataset
import random
import os
import json
from PIL import Image
import numpy as np

np.random.seed(42)
random.seed(42)

ds = load_dataset("TIGER-Lab/Mantis-Instruct", "contrastive_caption", cache_dir='/mnt/disk1/thkim/FederatedCL/dataset/')
print(len(ds))
print(ds.keys())
print(len(ds['train']))
breakpoint()

output_folder = 'dataset/Mantis_contrastive_caption'
json_data_list_train = []
json_data_list_test = []

# train_num = 4000
# test_num = 993

train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')
image_folder = os.path.join(output_folder, 'images')

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
    
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
    

total_indices = list(range(len(ds['train'])))
test_idx = np.random.choice(total_indices, size=int(len(total_indices)*0.2), replace=False).tolist()

count_valid = 0
choice_list = ['Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5', 'Image 6', 'Image 7', 'Image 8']

for idx, item in enumerate(ds['train']):
    images = item['images']
    conv = item['conversation']
    
    new_images = []
    valid = True
    for img in images:
        orig_path = img['path']
        if 'sharegpt4v/llava/llava_pretrain/images' in orig_path:
            img_path = orig_path.replace('sharegpt4v/llava/llava_pretrain/images', 'dataset/llava_pretrain/datasets--liuhaotian--LLaVA-Pretrain/snapshots/70f9d1e5e1a697fe35830875cfc7de1dd590d727')
        elif 'sharegpt4v/coco/train2017' in orig_path:
            img_path = orig_path.replace('sharegpt4v/coco/train2017', 'dataset/llava_pretrain/train2017')
        # if 'sharegpt4v/web-landmark' in orig_path:
        #     img_path = orig_path.replace('sharegpt4v/web-landmark', 'dataset/llava_pretrain/data/web-landmark')
        # elif 'sharegpt4v/web-celebrity' in orig_path:
        #     img_path = orig_path.replace('sharegpt4v/web-celebrity', 'dataset/llava_pretrain/data/web-celebrity')
        # elif 'sharegpt4v/wikiart' in orig_path:
        #     img_path = orig_path.replace('sharegpt4v/wikiart', 'dataset/llava_pretrain/data/wikiart')
        # elif 'sharegpt4v/sam/images' in orig_path:
        #     img_path = orig_path.replace('sharegpt4v/sam/images', 'dataset/llava_pretrain/data/sam')
        else:
            # print(orig_path)
            valid = False
            break
    
        new_img_path = os.path.join(image_folder, f"{orig_path.split('/')[-2]}_{orig_path.split('/')[-1]}")
        img = Image.open(img_path)
        img = img.convert('RGB')
        img.save(new_img_path)
        
        new_images.append(new_img_path)
    if not valid:
        continue
    
    for conv_id in range(int(len(conv)/2)):
        if conv[2*conv_id]['role'] == 'user':
            prompt = conv[2*conv_id]['content']
        else:
            continue
        if conv[2*conv_id + 1]['role'] == 'assistant':
            answer = conv[2*conv_id + 1]['content']
        else:
            continue
        
        if 'caption' not in prompt.lower() and 'which' not in prompt.lower():
            continue
        prompt = prompt.replace('<image>','')
        cur_choice_list = choice_list[:len(new_images)]
        if 'first' in answer or '1' in answer or "initial" in answer:
            new_answer = cur_choice_list[0]
        elif 'second' in answer or '2' in answer:
            new_answer = cur_choice_list[1]
        elif 'third' in answer or '3' in answer or 'tertiary' in answer:
            new_answer = cur_choice_list[2]
        elif 'fourth' in answer or '4' in answer:
            new_answer = cur_choice_list[3]
        elif 'fifth' in answer or '5' in answer:
            new_answer = cur_choice_list[4]
        elif 'sixth' in answer or '6' in answer:
            new_answer = cur_choice_list[5]
        elif 'seventh' in answer or '7' in answer:
            new_answer = cur_choice_list[6]
        elif 'eighth' in answer or '8' in answer:
            new_answer = cur_choice_list[7]
        elif 'end' in answer or 'final' in answer or "last" in answer:
            new_answer = cur_choice_list[-1]
        else:
            print(answer)
            continue
        
        json_data = {
            "id": item['id'] + f'_{conv_id}',
            "image": new_images,
            "conversations": [
                {
                    "from": "human",
                    "value": '<image>'*len(new_images) + "\n"+ prompt + f' You must choose your answer from the Choice List.\nChoice list:[{",".join(cur_choice_list)}]. Your answer is: '
                },
                { 
                    "from": "gpt",
                    "value": new_answer
                }
            ]
        }
        if idx in test_idx:
            json_data_list_test.append(json_data)
        else:
            json_data_list_train.append(json_data)
    if valid:
        count_valid += 1

print(count_valid)
print(len(json_data_list_train))
print(len(json_data_list_test))

breakpoint()

if len(json_data_list_train) > 10000:
    json_data_list_train = np.random.choice(json_data_list_train, size=10000, replace=False).tolist()
if len(json_data_list_test) > 2000:
    json_data_list_test = np.random.choice(json_data_list_test, size=2000, replace=False).tolist()

        
# with open(f'{train_folder}/dataset-0.json', 'w') as json_file:
#     json.dump(json_data_list_train, json_file, indent=4)
# with open(f'{test_folder}/dataset-0.json', 'w') as json_file:
#     json.dump(json_data_list_test, json_file, indent=4)
    
with open(f'{train_folder}/dataset-1.json', 'w') as json_file:
    json.dump(json_data_list_train, json_file, indent=4)
with open(f'{test_folder}/dataset-1.json', 'w') as json_file:
    json.dump(json_data_list_test, json_file, indent=4)