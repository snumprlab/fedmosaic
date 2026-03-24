from datasets import load_dataset
import random
import os
import json
from PIL import Image

random.seed(42)

ds = load_dataset("osunlp/MagicBrush")
print(len(ds))
print(ds.keys())
print(len(ds['train']))
print(len(ds['dev']))

breakpoint()

task_instruction = [
    "Presented with two images, A and B, and a sentence that describes an editing request, your task is to determine whether the instruction applies to changing A to B or B to A. You must choose your answer from the Choice List.",
    "Given images A and B along with an edit instruction, your responsibility is to judge whether the sentence describes transforming A to B or B to A. You must choose your answer from the Choice List.",
    "Upon receiving images A and B, along with a statement that illustrates an editing process, your job is to decide whether the instruction is for changing A to B or B to A. You must choose your answer from the Choice List.",
    "Using images A and B and a sentence that depicts the change between them, your task is to evaluate whether the instruction describes transforming A to B or B to A. You must choose your answer from the Choice List.",
    "With images A and B and an edit instruction describing the shift between them, your job is to assess whether the sentence refers to changing A to B or B to A. You must choose your answer from the Choice List.",
    "Provided with images A and B and a statement that illustrates a transition, your role is to determine if the instruction is for transforming A to B or B to A. You must choose your answer from the Choice List.",
    "Given images A and B along with a sentence explaining a modification, your task is to judge if the instruction refers to changing A to B or B to A. You must choose your answer from the Choice List.",
    "With images A and B and a sentence expressing an alteration, your job is to decide whether the instruction correctly interprets the transformation from A to B or B to A. You must choose your answer from the Choice List.",
    "Presented with images A and B and a sentence specifying a change, your duty is to assess whether the instruction describes transforming A to B or B to A. You must choose your answer from the Choice List.",
]

output_folder = 'dataset/MagicBrush'

train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')
image_folder = os.path.join(output_folder, 'images')

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
    
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

def save_datalist(datalist, output_folder, image_folder):
    json_datalist = []
    for i, item in enumerate(datalist):
        img_id = item['img_id']
        turn_idx = item['turn_index']
        
        src_img_path = os.path.join(image_folder, f'{img_id}_{turn_idx}_src.png')
        tgt_img_path = os.path.join(image_folder, f'{img_id}_{turn_idx}_tgt.png')
        
        if i % 2 == 0:
            images = [src_img_path, tgt_img_path]
            answer = 'A to B'
        else:
            images = [tgt_img_path, src_img_path]
            answer = 'B to A'
        
        instruction = item['instruction']
        
        question = f'\nImage A:<image> Image B:<image> Instruction:{instruction}\nChoice list:[A to B, B to A].'
        
        
        inst_idx = i % len(task_instruction)
        
        json_data = {
            "id": f'{img_id}_{turn_idx}',
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
    
    with open(f'{output_folder}/dataset-1.json', 'w') as json_file:
        json.dump(json_datalist, json_file, indent=4)


save_datalist(ds['train'], train_folder, image_folder)
save_datalist(ds['dev'], test_folder, image_folder)
