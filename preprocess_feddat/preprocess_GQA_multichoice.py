from PIL import Image
import io
import base64
import numpy as np
import json
from collections import defaultdict
import os
import random
import shutil

import openai
from openai import OpenAI
import time
from multiprocessing import Process, Queue, Manager, Pool, cpu_count
import math


NUM_SECONDS_TO_SLEEP = 0.01
NUM_PROCESSES = 10 #cpu_count()  # Use all available CPU cores

# Set up OpenAI client
client = OpenAI(
    api_key="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"
    )

rule = {
    "role": "Assistant",
    "prompt": "Please generate three answer choices for the given question asking about the given image and the given correct answer.\n"
              "The possible answers are the variation of correct and wrong answers. You should use it to infer about the image.\n"
              "You should:\n"
              "1. generate 3 choices.\n"
              "2. NOT INCLUDE correct answer in the generated choices.\n"
              "3. generate choices relevant to the question, but should not be confusing with the correct answer. Specifically, choices and answers should not be synonyms, nor should they have a relationship of inclusion.\n"
              "You must return the choices by strictly following this format:\"[[Choice A | Choice B | Choice C]]\", for example: \"Choice list: [[red | blue | pink]]\"."
}


task_mapping = {
    "verify": {"attr": 0, "rel": 1, "obj": 2, "global": 3},
    "query": {"rel": 4, "attr": 5, "cat": 6, "global": 7},
    "choose": {"attr": 8, "rel": 9, "cat": 10, "global": 11},
}

def resize_image(image_path, max_size):
    with Image.open(image_path) as img:
        img.thumbnail(max_size)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr

def encode_image(image_path, max_size=(256, 256)):
    img_byte_arr = resize_image(image_path, max_size)
    return base64.b64encode(img_byte_arr.read()).decode('utf-8')

def parse_score(review):
    try:
        score = review.split('[[')
        assert len(score) == 2
        score = score[-1].split(']]')[0]
        return score
    except Exception as e:
        print(f"Process {os.getpid()}: Error parsing score: {e}")
        print('error', review)
        return -1


def get_eval(content: str, max_tokens: int):
    user_content = [
                        {"type":"text", "text":content['text']},
                    ]
    for base64_image in content['image']:
        user_content.append(
            {"type": "image_url",
             "image_url":{"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        )

    while True:
        try:
            response = client.chat.completions.create(
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant to generate a choice list.'
                }, {
                    'role': 'user',
                    'content': user_content
                }],
                model="gpt-4o-mini-2024-07-18",
                temperature=0.2,
                max_tokens=max_tokens,
            )
            break
        except openai.APIConnectionError as e:
            print(f"Process {os.getpid()}: The server could not be reached")
            print(e.__cause__)
        except openai.RateLimitError as e:
            print(f"Process {os.getpid()}: A 429 status code was received; backing off")
            time.sleep(5)  # Back off for longer on rate limit errors
        except openai.APIStatusError as e:
            print(f"Process {os.getpid()}: Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
        time.sleep(NUM_SECONDS_TO_SLEEP)
    return response.choices[0].message.content


def count_types_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return count_types(data)

def count_types(data):
    type_counts = defaultdict(lambda: defaultdict(int))
    
    for item in data.values():
        if "types" in item:
            structural_type = item["types"].get("structural", "unknown")
            semantic_type = item["types"].get("semantic", "unknown")
            type_counts[structural_type][semantic_type] += 1
    
    return {key: dict(value) for key, value in type_counts.items()}


def process_data_chunk(chunk_data):
    process_id = os.getpid()
    chunk_id = chunk_data['chunk_id']
    chunk_datalists = chunk_data['datalists']
    data_dir = chunk_data['data_dir']
    type_name = chunk_data['type_name']
    task = chunk_data['task']

    new_datalists = [] 
    for idx, data in enumerate(chunk_datalists):
        new_data = {}
        new_data['id'] = data['id']
        print(chunk_id, idx) 

        if task <= 3:
            candidates = ['yes'] if data['answer'] == 'no' else ['no']
        else:
            base64_imgs = []
            image_paths = data['image']
            if isinstance(image_paths, list):
                for img_path in image_paths:
                    base64_imgs.append(encode_image(img_path))
            else:
                base64_imgs.append(encode_image(image_paths))

            content = {
                "text": (f"[Instruction]\n{rule['prompt']}\n\n"
                        f"[Question]\n{data['question']}\n\n"
                        f"[Correct Answer]\n{data['answer']}\n"),
                "image":
                   base64_imgs
                }

            review = get_eval(content, 512)
            candidates = parse_score(review).split(' | ')
        
        candidates += [data['answer']]
        random.shuffle(candidates)

        new_datalists.append({
            "id": data['id'],
            "image": data['image'],
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n Question: {data['question']}\n Choice lists: {candidates}.\n Your answer is:"
                },
                {
                    "from": "gpt",
                    "value": data["answer"]
                }
            ]
        })

    return {
        'chunk_id': chunk_id,
        'datalist': new_datalists,
    }
 

def preprocess_data(input_file):
    with open(os.path.join(base_dir, input_file), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    task_datasets = defaultdict(list)
    
    for item_id, item in data.items():
        structural_type = item["types"].get("structural", "unknown")
        semantic_type = item["types"].get("semantic", "unknown") 
        print(structural_type, semantic_type)
        task_id = task_mapping.get(structural_type, {}).get(semantic_type, -1)
        
        if task_id == -1:
            pass
            #raise ValueError("Not defined task")
        
        if task_id <= 3: #yes/no task
            task_datasets[task_id].append({
                "id": item_id,
                "question": item['question'],
                "answer": item['answer'],
                "image": f"{base_dir}/images/{item['imageId']}.jpg",
            })
        else:
            task_datasets[task_id].append({
                "id": item_id,
                "question": item['question'],
                "answer": item['answer'],
                "image": f"{base_dir}/images/{item['imageId']}.jpg",
            })

        '''
        if os.path.exists(f"{base_dir}/raw_images/{item['imageId']}.jpg"):
            print("exist", f"{base_dir}/raw_images/{item['imageId']}.jpg")
        else:
            print("pass", f"{base_dir}/raw_images/{item['imageId']}.jpg")
        '''

    new_task_datasets = defaultdict(list)
    for task_id, dataset in task_datasets.items():
        # Subsampling to a maximum of 10000 samples
        sampled_dataset = random.sample(dataset, min(len(dataset), max_num))

        # Save dataset to file
        output_file = f"dataset-{task_id//4}{task_id%4}.json"
        random.shuffle(sampled_dataset)
        
        new_task_datasets[task_id] = sampled_dataset

        print("task_id", task_id, "whole", len(dataset), "sampled", len(sampled_dataset))
        # Copy selected images to target directory
        for item in sampled_dataset:
            source_path = item["image"].replace('images', 'raw_images')
            target_path = item["image"]
            if os.path.exists(source_path):
                shutil.copy(source_path, target_path)
                pass
            else:
                print(source_path)
    
    return new_task_datasets


def combine_results(results, out_json):
    # Initialize combined data lists
    combined_datalist = []

    # Combine results from all processes
    for result in results:
        combined_datalist.extend(result['datalist'])

    print(out_json, len(combined_datalist))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined_datalist, f, indent=4, ensure_ascii=False)

base_dir = 'dataset/GQA'
dir_type = 'train'
max_num = 10000 if dir_type == 'train' else 1000
input_file = f"{dir_type}_balanced_questions.json"
os.makedirs(f'{base_dir}/{dir_type}', exist_ok=True)
os.makedirs(f'{base_dir}/images', exist_ok=True)
datalists = preprocess_data(input_file)

### Preprocessing per task ###
for task_num, task in enumerate(sorted(list(datalists.keys()))):
    if task == -1:
        continue
    num_chunks = min(NUM_PROCESSES, len(datalists[task]))
    sampled_task_datalist = np.random.choice(datalists[task], size=min(max_num, len(datalists[task])), replace=False).tolist()
    chunk_size = math.ceil(len(sampled_task_datalist) / num_chunks)
    print(f"Processing {len(sampled_task_datalist)} items using {num_chunks} processes")

    # Prepare chunks for processing
    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(sampled_task_datalist))
        print(f"chunk {i}, {len(sampled_task_datalist[start_idx:end_idx])}")
        chunk_data = {
            'chunk_id': i,
            'datalists': sampled_task_datalist[start_idx:end_idx],
            'data_dir': base_dir,
            'type_name': dir_type,
            'task': task
        }
        chunks.append(chunk_data)

    # Process chunks in parallel
    with Pool(processes=num_chunks) as pool:
        results = pool.map(process_data_chunk, chunks)
    out_json = os.path.join(base_dir, dir_type, f"dataset-{task//4}{task%4}.json")
    # Combine results from all processes
    combine_results(results, out_json)
