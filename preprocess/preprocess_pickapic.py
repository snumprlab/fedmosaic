from datasets import load_dataset
import random
import os
import json
from PIL import Image

random.seed(42)

# export HF_HOME=/disk1/thkim/datasets/

ds = load_dataset("yuvalkirstain/pickapic_v1")

print(len(ds))
print(ds.keys())
print(len(ds['train']))
print(len(ds['test']))

# breakpoint()

task_instruction = [
    "Presented with two images, A and B, and a caption describing an image, your task is to determine which image better matches the caption. You must choose your answer from the Choice List.",
    "Given images A and B along with a caption, your responsibility is to judge which image more accurately corresponds to the given caption. You must choose your answer from the Choice List.",
    "Upon receiving images A and B and a caption describing an image, your job is to decide which image best aligns with the given description. You must choose your answer from the Choice List.",
    "Using images A and B and a caption that describes an image, your task is to evaluate which of the two images better fits the given caption. You must choose your answer from the Choice List.",
    "With images A and B and a caption describing one of them, your job is to assess which image is best represented by the caption. You must choose your answer from the Choice List.",
    "Provided with images A and B and a caption that describes an image, your role is to determine which image corresponds better to the given caption. You must choose your answer from the Choice List.",
    "Given images A and B along with a descriptive caption, your task is to judge which image is the better match for the provided description. You must choose your answer from the Choice List.",
    "With images A and B and a caption that describes an image, your job is to decide which image correctly aligns with the provided description. You must choose your answer from the Choice List.",
    "Presented with images A and B and a caption specifying an image, your duty is to assess which image best fits the given description. You must choose your answer from the Choice List.",
]

output_folder = 'dataset/PickaPic'

train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')
image_folder = os.path.join(output_folder, 'images')

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
    
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

def save_datalist(datalist, output_folder, image_folder,max_len):
    json_datalist = []
    turn_A = True
    for i, item in enumerate(datalist):
        if not item['has_label']:
            continue
        else:
            if item['best_image_uid'] == item['image_0_uid'] and turn_A:
                answer = 'Image A'
                turn_A = False
            elif item['best_image_uid'] == item['image_1_uid'] and not turn_A:
                answer = 'Image B'
                turn_A = True
            else:
                continue
        image_A = item['jpg_0']
        image_B = item['jpg_1']
        image_A_path = os.path.join(image_folder,item['image_0_uid']) + '.jpg'
        image_B_path = os.path.join(image_folder,item['image_1_uid']) + '.jpg'
        with open(image_A_path, "wb") as file:
            file.write(image_A)
        with open(image_B_path, "wb") as file:
            file.write(image_B)
        caption = item['caption']
        
        question = f'\nImage A:<image> Image B:<image>\nCaption:{caption}\nChoice list:[Image A, Image B].'
        
        
        inst_idx = i % len(task_instruction)
        images = [image_A_path, image_B_path]
        json_data = {
            "id": f"{item['user_id']}_{item['ranking_id']}",
            "image": images,
            "conversations": [
                {
                    "from": "human",
                    "value": task_instruction[inst_idx] + question
                },
                { 
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        json_datalist.append(json_data)

        if len(json_datalist) == max_len:
            break
    with open(f'{output_folder}/dataset-0.json', 'w') as json_file:
        json.dump(json_datalist, json_file, indent=4)


save_datalist(ds['train'], train_folder, image_folder,max_len=10000)
save_datalist(ds['test'], test_folder, image_folder, max_len=1000)
