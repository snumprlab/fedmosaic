# from datasets import load_dataset
from PIL import Image
import os
import json
import uuid
import numpy as np
import random
from glob import glob
from collections import defaultdict

np.random.seed(42)
random.seed(42)

def process_choose_txt(output_folder, subset_name, size):
    subset_folder = os.path.join(output_folder, subset_name)
    image_subfolder = os.path.join(output_folder, 'images')
    
    if not os.path.exists(image_subfolder):
        os.makedirs(image_subfolder)

    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)
    
    json_data_list_1 = []
    json_data_list_2 = []
    
    datafolder_list = sorted(glob(subset_folder + '/choose_txt/*'))
    for folder in datafolder_list:
        with open(f'{folder}/data.json') as fp:
            data = json.load(fp)
        
        folder_name = folder.split('/')[-1]
        
        question = data['question']
        choices = data['choices']
        answer = choices[data['answer']]
        
        query_img_path = f'{folder}/image.png'
        
        # Load image if it's a URL or a file path
        image_paths = []
        try:
            image = Image.open(query_img_path)
            image_path = os.path.join(image_subfolder, f'choose_txt_{folder_name}_image.png')
            image.save(image_path)
            image_paths.append(image_path)
        except:
            print(query_img_path)
            continue
        
        
        
        choice_list = 'Choice list:['
        for i in range(len(choices)):
            choice_list += choices[i] + ', '
        choice_list = choice_list[:-2] + ']. Your answer is:'

        # Structure for LLaVA JSON
        json_data = {
            "id": folder_name,
            "image": image_paths,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n" + question + f"\n{choice_list}"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        
        if data['grade'] == 'prek' or data['grade'] == 'kindergarten':
            json_data_list_1.append(json_data)
        else:
            json_data_list_2.append(json_data)
        # json_data_list_1.append(json_data)
    # grouped_datalist = split_choose_txt(json_data_list_1)
    # for k in grouped_datalist.keys():
    #     print(f"{k}: {len(grouped_datalist[k])}")
    # breakpoint()    
    
    json_output_path = os.path.join(output_folder, subset_name, 'dataset-0.json')
    json_data_list_1 = np.random.choice(json_data_list_1, replace=False, size=min(size, len(json_data_list_1))).tolist()
    print(len(json_data_list_1))
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_1, json_file, indent=4)
    
    json_output_path = os.path.join(output_folder, subset_name, 'dataset-1.json')
    json_data_list_2 = np.random.choice(json_data_list_2, replace=False, size=min(size, len(json_data_list_2))).tolist()
    print(len(json_data_list_2))
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_2, json_file, indent=4)

def process_choose_img(output_folder, subset_name, size):
    answer_list = ["Image A", "Image B", "Image C", "Image D"]
    
    subset_folder = os.path.join(output_folder, subset_name)
    image_subfolder = os.path.join(output_folder, 'images')
    
    if not os.path.exists(image_subfolder):
        os.makedirs(image_subfolder)

    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)
    
    json_data_list_1 = []
    json_data_list_2 = []
    
    datafolder_list = sorted(glob(subset_folder + '/choose_img/*'))
    
    for folder in datafolder_list:
        with open(f'{folder}/data.json') as fp:
            data = json.load(fp)
        
        folder_name = folder.split('/')[-1]
        
        question = data['question']
        choice_imgs = data['choices']
        if len(choice_imgs) >= 5:
            continue
        answer = answer_list[data['answer']]
        
        query_img_path = f'{folder}/image.png'
        
        # Load image if it's a URL or a file path
        image_paths = []
        try:
            image = Image.open(query_img_path)
            image_path = os.path.join(image_subfolder, f'choose_img_{folder_name}_image.png')
            image.save(image_path)
            image_paths.append(image_path)
        except:
            print(query_img_path)
            continue
        
        
        fail = False
        for choice_img in choice_imgs:
            try:
                image = Image.open(f'{folder}/{choice_img}')
                image_path = os.path.join(image_subfolder, f'choose_img_{folder_name}_{choice_img}')
                image.save(image_path)
                image_paths.append(image_path)
            except:
                print(image_path)
                fail = True
                break
        
        if fail:
            continue
        choices = "\nChoices: "
        choice_list = 'Choice list:['
        for i in range(len(choice_imgs)):
            choice_list += answer_list[i] + ', '
            choices += answer_list[i] + ': <image>, '
        choices = choices[:-2] + '. '
        choice_list = choice_list[:-2] + ']. Your answer is:'

        # Structure for LLaVA JSON
        json_data = {
            "id": folder_name,
            "image": image_paths,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n" + question + f"{choices}\n{choice_list}"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        
        if data['grade'] == 'prek' or data['grade'] == 'kindergarten':
            json_data_list_1.append(json_data)
        else:
            json_data_list_2.append(json_data)
    
    json_output_path = os.path.join(output_folder, subset_name, 'dataset-4.json')
    json_data_list_1 = np.random.choice(json_data_list_1, replace=False, size=min(size, len(json_data_list_1))).tolist()
    print(len(json_data_list_1))
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_1, json_file, indent=4)
    
    json_output_path = os.path.join(output_folder, subset_name, 'dataset-5.json')
    json_data_list_2 = np.random.choice(json_data_list_2, replace=False, size=min(size, len(json_data_list_2))).tolist()
    print(len(json_data_list_2))
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_2, json_file, indent=4)

def process_choose_fillblank(output_folder, subset_name, size):
    subset_folder = os.path.join(output_folder, subset_name)
    image_subfolder = os.path.join(output_folder, 'images')
    
    if not os.path.exists(image_subfolder):
        os.makedirs(image_subfolder)

    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)
    
    json_data_list_1 = []
    json_data_list_2 = []
    
    datafolder_list = sorted(glob(subset_folder + '/fill_in_blank/*'))
    for folder in datafolder_list:
        with open(f'{folder}/data.json') as fp:
            data = json.load(fp)
        
        folder_name = folder.split('/')[-1]
        
        question = data['question']
        try:
            answer = str(int(data['answer']))
        except:
            answer = data['answer']
        
        query_img_path = f'{folder}/image.png'
        
        # Load image if it's a URL or a file path
        image_paths = []
        try:
            image = Image.open(query_img_path)
            image_path = os.path.join(image_subfolder, f'fill_in_blank_{folder_name}_image.png')
            image.save(image_path)
            image_paths.append(image_path)
        except:
            print(query_img_path)
            continue

        # Structure for LLaVA JSON
        json_data = {
            "id": folder_name,
            "image": image_paths,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n" +question + " Answer with a single word or a number. "
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        
        if data['grade'] == 'prek' or data['grade'] == 'kindergarten':
            json_data_list_1.append(json_data)
        else:
            json_data_list_2.append(json_data)
        # json_data_list_1.append(json_data)
    # grouped_datalist = split_choose_txt(json_data_list_1)
    # for k in grouped_datalist.keys():
    #     print(f"{k}: {len(grouped_datalist[k])}")
    # breakpoint()    
    
    json_output_path = os.path.join(output_folder, subset_name, 'dataset-2.json')
    json_data_list_1 = np.random.choice(json_data_list_1, replace=False, size=min(size, len(json_data_list_1))).tolist()
    print(len(json_data_list_1))
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_1, json_file, indent=4)
    
    json_output_path = os.path.join(output_folder, subset_name, 'dataset-3.json')
    json_data_list_2 = np.random.choice(json_data_list_2, replace=False, size=min(size, len(json_data_list_2))).tolist()
    print(len(json_data_list_2))
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_2, json_file, indent=4)

# Usage example
output_folder = 'dataset/iconqa'

# subset_names = ['train', 'test']
# subtasks = ['choose_img', 'choose_txt', 'fill_in_blank']
# grades = ['grade2', 'grade1', 'kindergarten', 'prek']

process_choose_img(output_folder, 'test', 2000)
process_choose_txt(output_folder, 'test', 2000)
process_choose_fillblank(output_folder, 'test', 2000)

process_choose_img(output_folder, 'train', 10000)
process_choose_txt(output_folder, 'train', 10000)
process_choose_fillblank(output_folder, 'train', 10000)