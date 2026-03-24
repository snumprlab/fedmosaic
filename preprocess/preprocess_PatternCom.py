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

output_folder = 'dataset/PatternCom'

train_folder = 'dataset/PatternCom/train'
test_folder = 'dataset/PatternCom/test'
image_folder = 'dataset/PatternCom/images'

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
    
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

image_path_pair = {
    'airplane':'airplane',
    'baseballfield': 'baseball_field',
    'basketballcourt': 'basketball_court',
    'beach': 'beach',
    'bridge': 'bridge',
    'cemetery': 'cemetery',
    'chaparral': 'chaparral',
    'christmastreefarm': 'christmas_tree_farm',
    'closedroad': 'closed_road',
    'coastalmansion': 'coastal_mansion',
    'crosswalk': 'crosswalk',
    'denseresidential': 'dense_residential',
    'ferryterminal': 'ferry_terminal',
    'footballfield': 'football_field',
    'forest': 'forest',
    'freeway': 'freeway',
    'golfcourse': 'golf_course',
    'harbor': 'harbor',
    'intersection': 'intersection',
    'mobilehomepark': 'mobile_home_park',
    'nursinghome': 'nursing_home',
    'oilgasfield': 'oil_gas_field',
    'oilwell': 'oil_well',
    'overpass': 'overpass',
    'parkinglot': 'parking_lot',
    'parkingspace': 'parking_space',
    'railway': 'railway',
    'river': 'river',
    'runway': 'runway',
    'runwaymarking': 'runway_marking',
    'shippingyard': 'shipping_yard',
    'solarpanel': 'solar_panel',
    'sparseresidential': 'sparse_residential',
    'storagetank': 'storage_tank',
    'swimmingpool': 'swimming_pool',
    'tenniscourt': 'tennis_court',
    'transformerstation': 'transformer_station',
    'wastewaterplant': 'wastewater_treatment_plant',
}

task_instructions = [
    "Presented with a query image, a query text, and a target image, your task is to determine if the target image shows the same object as the query image with the attribute specified by the query text. You must choose your answer from the Choice List.",
    "Given a query image, a query text, and a target image, your responsibility is to judge whether the target image depicts the same object as the query image, matching the attribute described in the query text. You must choose your answer from the Choice List.",
    "Upon receiving a query image, a query text, and a target image, your job is to decide if the target image represents the same object as the query image, incorporating the attribute outlined in the query text. You must choose your answer from the Choice List.",
    "Using a query image, a query text, and a target image, your task is to evaluate whether the target image portrays the same object as the query image, with the attribute given in the query text. You must choose your answer from the Choice List.",
    "With a query image, a query text, and a target image at your disposal, your job is to assess whether the target image shows the same object as the query image, reflecting the attribute described in the query text. You must choose your answer from the Choice List.",
    "Provided with a query image, a query text, and a target image, your role is to determine if the target image accurately depicts the same object as the query image, with the attribute conveyed by the query text. You must choose your answer from the Choice List.",
    "Given a query image, a query text, and a target image, your task is to judge whether the target image illustrates the same object as the query image, corresponding to the attribute mentioned in the query text. You must choose your answer from the Choice List.",
    "Using a query image, a query text, and a target image, your responsibility is to evaluate whether the target image properly depicts the same object as the query image, as described in the query text. You must choose your answer from the Choice List.",
    "With a query image, a query text, and a target image at hand, your job is to decide whether the target image reflects the same object as the query image, consistent with the attribute in the query text. You must choose your answer from the Choice List.",
]
types = ['color', 'shape', 'density', 'quantity', 'context', 'existence']
datalist_dict = {}
for type_ in types:
    datalist = load_data(f"{output_folder}/metadata/retrieved_{type_}.json")
    datalist_dict[type_] = datalist

total_list = []
total_size = 0
for index, _types in enumerate(types):
    datalist = datalist_dict[_types]
    json_datalist = []
    for id, item in enumerate(datalist):
        query_img = f"{image_folder}/{image_path_pair[item['query_file'][:-7]]}/{item['query_file']}"
        query_prompt = item['prompt']
        
        if id % 2 == 0:
            reference_img = f"{image_folder}/{image_path_pair[item['target_file'][:-7]]}/{item['target_file']}"
            answer='True'
        else:
            if random.random() >= 0.25:
                reference_img = f"{image_folder}/{image_path_pair[item['softNeg_file'][:-7]]}/{item['softNeg_file']}"
            else:
                other_datalist = random.choice([datalist_dict[t] for t in types if t != _types])
                random_item = random.choice(other_datalist)
                reference_img = f"{image_folder}/{image_path_pair[random_item['softNeg_file'][:-7]]}/{random_item['softNeg_file']}"
            answer='False'
        images = [query_img, reference_img]
        question = f'\nQuery Image:<image> Query Text:{query_prompt} => Output Image:<image>'
        
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
    with open(f'{train_folder}/dataset-{index+1}.json', 'w') as json_file:
        json.dump(json_datalist, json_file, indent=4)

    total_list.append(json_datalist)
    total_size += len(json_datalist)

json_train_datalist = []
json_test_datalist = []
for jsonlist in total_list:
    proportion = len(jsonlist) / total_size
    train_sizes = np.floor(proportion * 10000).astype(int)
    test_sizes = np.floor(proportion * 2000).astype(int)

    json_train_datalist += np.random.choice(jsonlist[:int(len(jsonlist)*0.8)], size=min(train_sizes,len(jsonlist[:int(len(jsonlist)*0.8)])), replace=False).tolist()
    json_test_datalist += np.random.choice(jsonlist[int(len(jsonlist)*0.8):], size=min(test_sizes,len(jsonlist[int(len(jsonlist)*0.8):])), replace=False).tolist()

print(len(json_train_datalist))
print(len(json_test_datalist))
with open(f'{train_folder}/dataset-0.json', 'w') as json_file:
    json.dump(json_train_datalist, json_file, indent=4)

with open(f'{test_folder}/dataset-0.json', 'w') as json_file:
    json.dump(json_test_datalist, json_file, indent=4)