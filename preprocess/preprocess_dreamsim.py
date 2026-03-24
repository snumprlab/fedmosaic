import pandas as pd
import os
import json
import random
import numpy as np

random.seed(42)
np.random.seed(42)

output_folder = 'dataset/DreamSim'

task_instruction=[
    "Presented with a reference image and two other images (Image A and Image B), your task is to determine which image is more visually similar to the reference image. You must choose your answer from the Choice List.",

    "Given a reference image along with Image A and Image B, your responsibility is to judge which of the two images better resembles the reference image. You must choose your answer from the Choice List.",

    "Upon receiving a reference image and two images (Image A and Image B), your job is to decide which image more accurately reflects the visual characteristics of the reference image. You must choose your answer from the Choice List.",

    "Using a reference image and two other images (Image A and Image B), your task is to evaluate which of the two images is more visually similar to the reference image. You must choose your answer from the Choice List.",

    "With a reference image and two images (Image A and Image B) provided, your job is to assess which image more closely resembles the reference image. You must choose your answer from the Choice List.",

    "Provided with a reference image and two additional images (Image A and Image B), your role is to determine which image better conveys visual similarity to the reference image. You must choose your answer from the Choice List.",

    "Given a reference image and two images (Image A and Image B), your task is to judge which image more effectively communicates visual resemblance to the reference image. You must choose your answer from the Choice List.",

    "Using a reference image and two images (Image A and Image B), your responsibility is to evaluate which of the two images more appropriately depicts similarity to the reference image. You must choose your answer from the Choice List.",

    "With a reference image and two images (Image A and Image B) at hand, your job is to decide which image best interprets the visual similarity to the reference image. You must choose your answer from the Choice List.",

    "Presented with a reference image and two other images (Image A and Image B), your duty is to assess which image more accurately resembles the reference image. You must choose your answer from the Choice List.",
]

df = pd.read_csv(os.path.join(output_folder, 'data.csv'))

json_data_list_train = []
json_data_list_test = []

train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
    
# Track all referenced images
referenced_images = set()

for index, row in df.iterrows():
    images = [os.path.join(output_folder,row['left_path']),
              os.path.join(output_folder,row['ref_path']),
              os.path.join(output_folder,row['right_path'])]
    
    question = f'\nImage A:<image> Ref Image:<image> Image B:<image>'
    answer = 'Image A' if row['left_vote'] else 'Image B'
    inst_idx = row['id'] % len(task_instruction)
    json_data = {
        'id': row['id'],
        'image':images,
        "conversations": [
            {
                "from": "human",
                "value": task_instruction[inst_idx] + question + '\nChoice list:[Image A, Image B]. Your answer is:'
            },
            { 
                "from": "gpt",
                "value": answer
            }
        ]
    }
    
    if row['split'] == 'train':
        json_data_list_train.append(json_data)
    elif row['split'] == 'test':
        json_data_list_test.append(json_data)

print(len(json_data_list_train))
print(len(json_data_list_test))

if len(json_data_list_train) > 10000:
    json_data_list_train = np.random.choice(json_data_list_train, size=10000, replace=False).tolist()
if len(json_data_list_test) > 2000:
    json_data_list_test = np.random.choice(json_data_list_test, size=2000, replace=False).tolist()
print(len(json_data_list_train))
print(len(json_data_list_test))    

for item in json_data_list_train:
    # Add images to referenced set
    referenced_images.update(item['image'])

for item in json_data_list_test:
    referenced_images.update(item['image'])

with open(f'{train_folder}/dataset-0.json', 'w') as json_file:
    json.dump(json_data_list_train, json_file, indent=4)
with open(f'{test_folder}/dataset-0.json', 'w') as json_file:
    json.dump(json_data_list_test, json_file, indent=4)
    
# Remove unreferenced images
all_images_in_folder = set()

# Walk through all subdirectories and collect all image files
for root, dirs, files in os.walk(output_folder):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            all_images_in_folder.add(os.path.join(root, file))

# Find unreferenced images
unreferenced_images = all_images_in_folder - referenced_images

# Remove unreferenced images
for image in unreferenced_images:
    try:
        os.remove(image)
        print(f'Removed unreferenced image: {image}')
    except OSError as e:
        print(f"Error removing {image}: {e}")