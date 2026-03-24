from datasets import load_dataset
import random
import os
import json
from PIL import Image
import numpy as np

np.random.seed(42)
random.seed(42)

# export HF_HOME=/disk1/thkim/datasets/

ds = load_dataset("TIGER-Lab/Mantis-Instruct", "imagecode")
print(len(ds))
print(ds.keys())
print(len(ds['train']))
breakpoint()

output_folder = 'dataset/imagecode'

train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')
image_folder = os.path.join(output_folder, 'images')

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
    
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

answers = list('ABCDEFGHIJ')
json_datalist1 = []
json_datalist2 = []
json_datalist3 = []
json_datalist4 = []
for idx, item in enumerate(ds['train']):
    images = item['images']
    conv = item['conversation']
    
    if 'Answer: Image ' in conv[1]['content']:
        # <image><image>...
        answer_idx = int(conv[1]['content'].split(' ')[-1]) - 1
        new_images = [images[0]['path'], images[3]['path'], images[6]['path'], images[9]['path']]
        if answer_idx <= 2:
            new_images[0] = images[answer_idx]['path']
            answer = 'Image 1'
        elif answer_idx <= 4:
            new_images[1] = images[answer_idx]['path']
            answer = 'Image 2'
        elif answer_idx <= 6:
            new_images[2] = images[answer_idx]['path']
            answer = 'Image 3'
        else:
            new_images[3] = images[answer_idx]['path']
            answer = 'Image 4'
        for i in range(len(new_images)):
            path = new_images[i].split('/')
            path[0] = image_folder
            new_path = '/'.join(path)
            ###
            # img = Image.open(new_images[i])
            # img = img.convert('RGB')
            # if not os.path.exists('/'.join(path[:-1])):
            #     os.makedirs('/'.join(path[:-1]))
            # img.save(new_path)
            ###
            new_images[i] = new_path
        
        new_instruction = conv[0]['content'].replace('<image><image><image><image><image><image><image><image><image><image>', 'Image 1:<image>\nImage 2:<image>\nImage 3:<image>\nImage 4:<image>')
        if '10' in new_instruction:
            new_instruction = new_instruction.replace('10', '4')
        
        new_instruction += '\nYou must choose from the choice list.\nChoice list:[Image 1, Image 2, Image 3, Image 4].'
    else:
        answer_idx = answers.index(conv[1]['content'][-1])
        new_images = [images[0]['path'], images[3]['path'], images[6]['path'], images[9]['path']]
        if answer_idx <= 2:
            new_images[0] = images[answer_idx]['path']
            answer = 'Image A'
        elif answer_idx <= 4:
            new_images[1] = images[answer_idx]['path']
            answer = 'Image B'
        elif answer_idx <= 6:
            new_images[2] = images[answer_idx]['path']
            answer = 'Image C'
        else:
            new_images[3] = images[answer_idx]['path']
            answer = 'Image D'
        for i in range(len(new_images)):
            path = new_images[i].split('/')
            path[0] = image_folder
            new_path = '/'.join(path)
            ###
            # img = Image.open(new_images[i])
            # img = img.convert('RGB')
            # if not os.path.exists('/'.join(path[:-1])):
            #     os.makedirs('/'.join(path[:-1]))
            # img.save(new_path)
            ###
            new_images[i] = new_path
        
        new_instruction = conv[0]['content'].replace(
                    "A. <image>\nB. <image>\nC. <image>\nD. <image>\nE. <image>\n"
                    "F. <image>\nG. <image>\nH. <image>\nI. <image>\nJ. <image>", 
                    "Image A. <image>\nImage B. <image>\nImage C. <image>\nImage D. <image>")
        if '10' in new_instruction:
            new_instruction = new_instruction.replace('10', '4')
        
        new_instruction += '\nYou must choose from the choice list.\nChoice list:[Image A, Image B, Image C, Image D].'
    json_data = {
        "id": f"{item['id']}",
        "image": new_images,
        "conversations": [
            {
                "from": "human",
                "value": new_instruction
            },
            { 
                "from": "gpt",
                "value": answer
            }
        ]
    }

    # MSR-VTT-videoTrainValVideo open-images video-storytelling-videowedding YouCook
    if 'MSR-VTT-videoTrainValVideo' in new_images[0]:
        json_datalist1.append(json_data)
    elif 'open-images' in new_images[0]:
        json_datalist2.append(json_data)
    elif 'video-storytelling' in new_images[0]:
        json_datalist3.append(json_data)
    elif 'YouCook' in new_images[0]:
        json_datalist4.append(json_data)
    else:
        print(new_images[0])
        breakpoint()

print(len(json_datalist1))
print(len(json_datalist2))
print(len(json_datalist3))
print(len(json_datalist4))

random.shuffle(json_datalist1)
random.shuffle(json_datalist2)
random.shuffle(json_datalist3)
random.shuffle(json_datalist4)


json_train_datalist1 = json_datalist1[:int(len(json_datalist1)*0.83)] + json_datalist2[:int(len(json_datalist2)*0.83)]
json_test_datalist1 = json_datalist1[int(len(json_datalist1)*0.83):] + json_datalist2[int(len(json_datalist2)*0.83):]
json_train_datalist2 = json_datalist3[:int(len(json_datalist3)*0.83)] + json_datalist4[:int(len(json_datalist4)*0.83)]
json_test_datalist2 = json_datalist3[int(len(json_datalist3)*0.83):] + json_datalist4[int(len(json_datalist4)*0.83):]

print(len(json_train_datalist1))
print(len(json_test_datalist1))
print(len(json_train_datalist2))
print(len(json_test_datalist2))
        
with open(f'{train_folder}/dataset-0.json', 'w') as json_file:
    json.dump(json_train_datalist1, json_file, indent=4)
with open(f'{test_folder}/dataset-0.json', 'w') as json_file:
    json.dump(json_test_datalist1, json_file, indent=4)
with open(f'{train_folder}/dataset-1.json', 'w') as json_file:
    json.dump(json_train_datalist2, json_file, indent=4)
with open(f'{test_folder}/dataset-1.json', 'w') as json_file:
    json.dump(json_test_datalist2, json_file, indent=4)
# with open(f'{train_folder}/dataset-1.json', 'w') as json_file:
#     json.dump(json_data_list_train, json_file, indent=4)
# with open(f'{test_folder}/dataset-1.json', 'w') as json_file:
#     json.dump(json_data_list_test, json_file, indent=4)