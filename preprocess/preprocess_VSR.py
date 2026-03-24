import jsonlines
import os
import random
import json

random.seed(42)

output_folder = 'dataset/VSR'
train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')
image_folder = os.path.join(output_folder, 'images')

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
if not os.path.exists(image_folder):
    os.makedirs(image_folder)


train_datalist = []
test_datalist = []

instruction_ = "For a given image and caption, your task is to determine whether the caption correctly describe the spatial relationship between objects in the image. You must choose your answer from the Choice List."

with jsonlines.open("visual-spatial-reasoning/data/splits/zeroshot/train.jsonl") as f:
    for item in f.iter():
        image_file = item['image']
        caption = item['caption']
        answer = 'True' if item['label'] == 1 else 'False'
        filepath = os.path.join(image_folder, image_file)
        json_data = {
                'id': image_file[:-3],
                'image':[filepath],
                "conversations": [
                    {
                        "from": "human",
                        "value": instruction_ + '\n<image>\nCaption: ' + caption + '\nChoice list:[True, False]. Your answer is:'
                    },
                    { 
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }
        train_datalist.append(json_data)
with jsonlines.open("visual-spatial-reasoning/data/splits/zeroshot/dev.jsonl") as f:
    for item in f.iter():
        image_file = item['image']
        caption = item['caption']
        answer = 'True' if item['label'] == 1 else 'False'
        filepath = os.path.join(image_folder, image_file)
        json_data = {
                'id': image_file[:-3],
                'image':[filepath],
                "conversations": [
                    {
                        "from": "human",
                        "value": instruction_ + '\n<image>\nCaption: ' + caption + '\nChoice list:[True, False]. Your answer is:'
                    },
                    { 
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }
        train_datalist.append(json_data)

with jsonlines.open("visual-spatial-reasoning/data/splits/zeroshot/test.jsonl") as f:
    for item in f.iter():
        image_file = item['image']
        caption = item['caption']
        answer = 'True' if item['label'] == 1 else 'False'
        filepath = os.path.join(image_folder, image_file)
        json_data = {
                'id': image_file[:-3],
                'image':[filepath],
                "conversations": [
                    {
                        "from": "human",
                        "value": instruction_ + '\n<image>\nCaption: ' + caption + '\nChoice list:[True, False]. Your answer is:'
                    },
                    { 
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }
        test_datalist.append(json_data)

print(len(train_datalist))
print(len(test_datalist))

with open(f'{train_folder}/dataset-0.json', 'w') as json_file:
    json.dump(train_datalist, json_file, indent=4)

with open(f'{test_folder}/dataset-0.json', 'w') as json_file:
    json.dump(test_datalist, json_file, indent=4)