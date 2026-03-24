# from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
import os
import json
import uuid
import csv
import glob

def split_data(outputdata, datalist, datatype, subset_name):
    # Process and save images and labels
    for item in datalist:
        # Define image path
        image_path = f"dataset/Mementos/{subset_name}/image_{datatype}/{item['img_path']}"

        json_data = {
            "id": item['img_path'],
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n" + "Write a description for the given image sequence in a single paragraph, what is happening in this episode?"
                },
                { 
                    "from": "gpt",
                    "value": item['description']
                }
            ]
        }
        outputdata.append(json_data)
        
    return outputdata

def save_dataset_train(dataset_name, output_folder):
    #cmc
    cmc = []
    f = open(f'./{output_folder}/train_cmc_description.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        cmc.append({'img_path':line[0], 'description':line[1]})
    f.close()
    cmc = cmc[1:]
    
    # robo
    robo = []
    f = open(f'./{output_folder}/train_robo_description.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        robo.append({'img_path':line[0], 'description':line[1]})
    f.close()
    robo = robo[1:]
    
    # rw
    rw = []
    f = open(f'./{output_folder}/train_dl_description.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        rw.append({'img_path':line[0], 'description':line[1]})
    f.close()
    rw = rw[1:]
    
    train_json_data_list = []
    
    train_json_data_list = split_data(train_json_data_list, cmc, 'cmc', 'train')
    train_json_data_list = split_data(train_json_data_list, robo, 'robo', 'train')
    train_json_data_list = split_data(train_json_data_list, rw, 'rw', 'train')
    print(len(train_json_data_list))
    # Save the JSON data list to a file
    json_output_path = os.path.join(output_folder, 'train', 'dataset-0.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(train_json_data_list, json_file, indent=4)

def save_dataset_test(dataset_name, output_folder):
    #cmc
    cmc = []
    f = open(f'./{output_folder}/cmc_description.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        cmc.append({'img_path':line[0], 'description':line[1]})
    f.close()
    cmc = cmc[1:]
    
    # robo
    robo = []
    f = open(f'./{output_folder}/robo_description.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        robo.append({'img_path':line[0], 'description':line[1]})
    f.close()
    robo = robo[1:]
    
    # rw
    rw = []
    f = open(f'./{output_folder}/dl_description.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        rw.append({'img_path':line[0], 'description':line[1]})
    f.close()
    rw = rw[1:]
    
    test_json_data_list = []
    
    test_json_data_list = split_data(test_json_data_list, cmc, 'cmc', 'test')
    test_json_data_list = split_data(test_json_data_list, robo, 'robo', 'test')
    test_json_data_list = split_data(test_json_data_list, rw, 'rw', 'test')
    
    print(len(test_json_data_list))
    # Save the JSON data list to a file
    json_output_path = os.path.join(output_folder, 'test', 'dataset-0.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(test_json_data_list, json_file, indent=4)


# Usage example
output_folder = 'dataset/Mementos'

save_dataset_train('Mementos', output_folder)

save_dataset_test('Mementos', output_folder)