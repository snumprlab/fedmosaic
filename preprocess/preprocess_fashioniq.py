import json
import random
import os
from PIL import Image
import numpy as np

np.random.seed(42)
random.seed(42)

# Load datasets
def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

output_folder = 'dataset/FashionIQ'

train_folder = 'dataset/FashionIQ/train'
test_folder = 'dataset/FashionIQ/test'
image_folder = 'dataset/FashionIQ/images'

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
    
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

if not os.path.exists(image_folder):
    os.makedirs(image_folder)

task_instructions = [
    "Presented with a query image, a reference image, and a sentence describing the query image, your task is to determine if the sentence accurately depicts the query image in comparison to the reference image. You must choose your answer from the Choice List.",
    "Given a query image, a reference image, and a sentence outlining the query image, your responsibility is to judge whether the sentence correctly describes the query image relative to the reference image. You must choose your answer from the Choice List.",
    "Upon receiving a query image, a reference image, and a statement that illustrates the query image, your job is to decide whether the sentence correctly characterizes the query image in relation to the reference image. You must choose your answer from the Choice List.",
    "Using a query image, a reference image, and a sentence that narrates the query image, your task is to evaluate whether the sentence accurately portrays the query image compared to the reference image. You must choose your answer from the Choice List.",
    "With a query image, a reference image, and a sentence describing the query image at your disposal, your job is to assess whether the sentence correctly represents the query image relative to the reference image. You must choose your answer from the Choice List.",
    "Provided with a query image, a reference image, and a sentence describing the query image, your role is to determine if the sentence accurately conveys the description of the query image in comparison to the reference image. You must choose your answer from the Choice List.",
    "Given a query image, a reference image, and a sentence explaining the query image, your task is to judge if the sentence correctly communicates the description of the query image in relation to the reference image. You must choose your answer from the Choice List.",
    "Using a query image, a reference image, and a sentence that presents the query image, your responsibility is to evaluate whether the sentence appropriately depicts the query image in comparison to the reference image. You must choose your answer from the Choice List.",
    "With a query image, a reference image, and a sentence expressing the query image at hand, your job is to decide whether the sentence correctly interprets the query image relative to the reference image. You must choose your answer from the Choice List.",
]
types = ['dress', 'shirt', 'toptee']
subset = ['train', 'val']

json_train_datalist = []
json_test_datalist = []

for index, _types in enumerate(types):
    for _subset in subset:
        
        datalist = load_data(f"{output_folder}/captions/cap.{_types}.{_subset}.json")
        json_datalist = []
        for id, item in enumerate(datalist):
            query_img = f'{image_folder}/{item["target"]}.jpg'
            reference_img = f'{image_folder}/{item["candidate"]}.jpg'
            images = [query_img, reference_img]
            sentence = None
            if id % 2 == 0:
                sentence = "The cloth in the query image " + random.choice(item['captions']).lower()
                answer='True'
            else:
                for item2 in datalist:
                    if item2['candidate'] == item['target']:
                        sentence = "The cloth in the query image " + random.choice(item2['captions']).lower()
                        break
                if sentence is None:
                    item2 = random.choice(datalist)
                    sentence = "The cloth in the query image " + random.choice(item2['captions']).lower()
                answer='False'
            question = f'\nQuery Image:<image> Reference Image:<image> Description:{sentence}'
            
            inst_idx = int(id%len(task_instructions))
            
            json_data = {
                "id": id,
                "image": images,
                "conversations": [
                    {
                        "from": "human",
                        "value": task_instructions[inst_idx] + question + '\nChoice list:[True, False]. Your answer is:'
                    },
                    { 
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }
            json_datalist.append(json_data)
        print(len(json_datalist))
        if _subset == 'train':
            json_train_datalist.append(json_datalist)
            with open(f'{train_folder}/dataset-{index+1}.json', 'w') as json_file:
                json.dump(json_datalist, json_file, indent=4)
        elif _subset == 'val':
            json_test_datalist.append(json_datalist)
            with open(f'{test_folder}/dataset-{index+1}.json', 'w') as json_file:
                json.dump(json_datalist, json_file, indent=4)

json_train_datalist = np.random.choice(json_train_datalist[0], size=3333, replace=False).tolist() + np.random.choice(json_train_datalist[1], size=3333, replace=False).tolist() + np.random.choice(json_train_datalist[2], size=3333, replace=False).tolist()
json_test_datalist = np.random.choice(json_test_datalist[0], size=667, replace=False).tolist() + np.random.choice(json_test_datalist[1], size=667, replace=False).tolist() + np.random.choice(json_test_datalist[2], size=667, replace=False).tolist()

print(len(json_train_datalist))
print(len(json_test_datalist))

with open(f'{train_folder}/dataset-0.json', 'w') as json_file:
    json.dump(json_train_datalist, json_file, indent=4)

with open(f'{test_folder}/dataset-0.json', 'w') as json_file:
    json.dump(json_test_datalist, json_file, indent=4)