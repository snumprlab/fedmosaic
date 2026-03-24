# from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
import os
import json
import uuid
import numpy as np
import random

np.random.seed(42)
random.seed(42)

def process_and_save(dataset, output_folder, subset_name, size, idx):
    # Define image subfolder within output folder
    subset_folder = os.path.join(output_folder, subset_name)
    image_subfolder = os.path.join(output_folder, 'images')


    if not os.path.exists(image_subfolder):
        os.makedirs(image_subfolder)


    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)


    # Initialize list to hold all JSON data
    # dataset-0 : no need external knowledge
    # dataset-1 : need external knowledge
    json_data_list = []
    json_data_list2 = []


    # Process and save images and labels
    for item in dataset:
        # Load image if it's a URL or a file path
        # if isinstance(item['image'], str):
        #     # response = requests.get(item['image'])
        #     image = Image.open(f"./dataset/AQUA/images/{item['image']}")
        #     # image = Image.open(BytesIO(response.content))
        # else:
        #     image = item['image']  # Assuming it's a PIL.Image object


        # Define image path
        image_path = os.path.join(image_subfolder, f"{item['image']}")


        # Save image
        # image.save(image_path)


        # Remove duplicates and format answers
        answers = [item['answer']]
        unique_answers = list(set(answers))
        formatted_answers = ", ".join(unique_answers)


        # Structure for LLaVA JSON
        json_data = {
            "id": item['image'],
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n" + item['question'] + "? Answer as concise as possible." 
                },
                {
                    "from": "gpt",
                    "value": formatted_answers
                }
            ]
        }


        # Append to list
        if item['need_external_knowledge']:
            json_data_list2.append(json_data)
        else:
            json_data_list.append(json_data)

    print(len(json_data_list))
    print(len(json_data_list2))
    # 2777 7259
    # 29568 40244
    # Save the JSON data list to a file
    json_output_path = os.path.join(output_folder, subset_name, f'dataset-{idx}.json')
    json_data_list = np.random.choice(json_data_list, replace=False, size=min(size, len(json_data_list))).tolist()
    print(len(json_data_list))
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)


def save_dataset(dataset_name, output_folder):
    # Load the dataset from Hugging Face
    # dataset = load_dataset(dataset_name, split=subset_name)


    # Filter for images with the specified class in 'question_type'
    # filtered_dataset = [item for item in dataset if item['question_type'] == class_name]


    # Determine the split for training and validation
    # if val_samples is not None and subset_name == 'train':
    #     train_dataset = filtered_dataset[val_samples:]
    #     val_dataset = filtered_dataset[:val_samples]
    # else:
    #     train_dataset = filtered_dataset
    #     val_dataset = []
    with open(f"dataset/{dataset_name}/train.json") as fp:
        train_dataset = json.load(fp)

    with open(f"dataset/{dataset_name}/test.json") as fp:
        test_dataset = json.load(fp)
    with open(f"dataset/{dataset_name}/val.json") as fp:
        val_dataset = json.load(fp)
    test_dataset.extend(val_dataset)
    # Process and save the datasets
    for subset, data, size, idx in [('train', train_dataset, 6000, 10), ('train', train_dataset, 6000, 20)]:
        if data:
            process_and_save(data, output_folder, subset, size, idx)

# Usage example
output_folder = 'dataset/AQUA'
# class_name = 'other'
# val_samples = 300

save_dataset('AQUA', output_folder)

import shutil
shutil.copyfile('dataset/AQUA/test/dataset-0.json', 'dataset/AQUA/test/dataset-10.json')
shutil.copyfile('dataset/AQUA/test/dataset-0.json', 'dataset/AQUA/test/dataset-20.json')
# save_dataset('Multimodal-Fatima/OK-VQA_train', output_folder, class_name, 'train', val_samples)
# save_dataset('Multimodal-Fatima/OK-VQA_test', output_folder, class_name, 'test')
