from PIL import Image
import io
import base64
import numpy as np
import json
from collections import defaultdict
from collections import Counter
import re
import os
import random
import shutil

import openai
from openai import OpenAI
import time
from multiprocessing import Process, Queue, Manager, Pool, cpu_count
import math


NUM_SECONDS_TO_SLEEP = 0.01
NUM_PROCESSES = 20 #cpu_count()  # Use all available CPU cores

# Set up OpenAI client
client = OpenAI(
    api_key="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"
    #"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"
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

object_recognition_keywords = [
    "what is this", "which object", "what device is this", "what type of",
    "identify", "which item", "which tool", "which animal",
    "what is shown in the image", "what can be seen", "find the",
    "what is the name of",
    "who ", "which person", "what is the name of the person",
    "which character", "identify the person", "who can be seen in the image",
    "which man", "which woman", "who is standing", "who is sitting",
    "which child"
]        

def most_frequent_element(lst):
    count = Counter(lst)
    max_count = max(count.values())
    most_common_elements = [key for key, val in count.items() if val == max_count]  # 최빈값 찾기
    return most_common_elements[0] if len(most_common_elements) == 1 else most_common_elements

def get_knowledge_task_mapping(question):
    if any(keyword in question for keyword in object_recognition_keywords):
        #subtasks["object_recognition"].append(item)
        return 0
    elif "color" in question or "shape" in question or "material" in question:
        #subtasks["attribute_based"].append(item)
        return 1
    elif "next to" in question or "on the" in question or "behind" in question:
        #subtasks["relational_reasoning"].append(item)
        return 2
    elif "used for" in question or "can store" in question or "can hold" in question:
        #subtasks["functional_knowledge"].append(item)
        return 3
    else:
        return -1

# 질문 유형 분석 함수
def get_object_task_mapping(question):
    question_lower = question.lower()

    if re.search(r"(what color|how big|how many|what shape|what type|what kind|what material|what brand|how heavy|how deep|how wide|what texture|what does this feel like|what flavor|what category)", question_lower):
        return 0 #"attribute"

    elif re.search(r"(is this.* or |which one is |choose|what is this,.* or )", question_lower):
        return 1 #"comparison_choice"

    elif re.search(r"(which place|what place|where.*image|where was.*taken)", question_lower):
        return 2 #"place_identification"

    elif re.search(r"(what.*object|which.*object|what type of.*device|which animal|what kind of.*furniture)", question_lower):
        return 3 #"object_identification"

    return -1 

def get_attribute_task_mapping(question):
        
    # 1. 행동 및 상태 관련 질문
    if any(keyword in question for keyword in ["walking", "doing", "pose", "standing", "running", "jumping", "holding", "carrying", "sitting", "lying", "playing", "moving", "gesture", "picking", "dropping", "throwing", "catching", "drinking", "eating", "writing", "reading", "typing", "holding", "waving", "pointing", "looking", "smiling", "crying", "talking", "climbing", "dancing", "driving", "riding", "pushing", "pulling", "lifting", "hugging", "shaking hands", "swimming", "skating", "skiing", "fighting", "jumping over", "diving", "shooting", "kicking", "hitting", "bending", "stretching"]):
        return 0

    # 2. 위치 및 관계 관련 질문
    elif any(keyword in question for keyword in ["indoors", "outdoors", "location", "where", "position", "place", "scene", "area", "setting", "direction", "near", "far", "beside", "next to", "behind", "in front of", "above", "below", "left", "right", "between", "center"]):
        return 1

    # 3. 색상 및 외형 관련 질문
    elif any(keyword in question for keyword in ["color", "shade", "hue", "tint", "brightness", "contrast", "pattern", "shape", "design", "appearance"]):
        return 2

    # 4. 재질 및 물체 유형 관련 질문
    elif any(keyword in question for keyword in ["made of", "kind of", "material", "fabric", "texture", "surface", "constructed", "built", "composition", "wood", "metal", "plastic", "glass", "stone", "cotton", "leather", "paper", "furniture", "vehicle", "clothing", "food", "tool", "electronics"]):
        return 3

    else:
        return -1

def get_scenetext_task_mapping(entry):
    question_text = entry.get("question", "").lower()

    if any(keyword in question_text for keyword in ["number", "digit", "symbol", "percentage", "code", "amount", "year", "time", "measurement", "units", "value", "count", "sum", "total", "rate", "temperature"]):
        #return "number_symbol_recognition"
        return 0
    elif any(keyword in question_text for keyword in ["left", "right", "above", "below", "beside", "next to", "middle", "top", "bottom", "corner", "between", "side", "near", "aligned", "opposite", "facing", "oriented"]):
        #return "spatial_directional_reasoning"
        return 1
    elif any(keyword in question_text for keyword in ["kind", "type", "category", "brand", "name"]):
        #return "object_category_identification"
        return 2
    else:
        return 3

def get_logical_task_mapping(entry):
    types = entry.get("raw_question_type", {})
    detailed_type = types.get("detailed", "")
    question_text = entry.get("question", "").lower()

    if detailed_type in ["twoSameC", "twoDifferentC"]:
        #return "color_comparison"
        return 0
    elif detailed_type == "twoCommon":
        #return "common_attribute"
        return 1
    elif detailed_type in ["diffAnimals", "compare", "attr"] or any(
        keyword in question_text for keyword in ["type", "kind", "category", "class", "same", "different", "group", "belong"]
    ):
        #return "object_type_comparison"
        return 2
    else:
        #return "general_attribute"
        return 3

def get_scene_task_mapping(question):
    question_lower = question.lower()
    
    # Attributes like (color, size, quantity)
    if re.search(r"(what color|how big|how many|what shape|how tall|how long|what size|what type|what kind|what material|what brand|how heavy|how deep|how wide)", question_lower):
        return 0

    # Specific Object Identification
    elif re.search(r"^(what is (this|that|the|an|a)|which (one|object|animal|device|item) is|who is)", question_lower):
        return 1

    # Spatial Relationships
    elif re.search(r"(where|next to|in front of|behind|on top of|under|between|beside|above|below|near|to the left of|to the right of)", question_lower):
        return 2

    else:
        if re.search(r"\bwhat\b", question_lower):
            if re.search(r"(color|size|type|shape|kind|material|brand|amount)", question_lower):
                return 0
            elif re.search(r"(next to|in front of|behind|on top of|under|between|beside|above|below|near|left|right)", question_lower):
                return 2
            return 3
        else:
            return 3

def top_4_keys_by_items(dictionary):
    filtered_dict = {k: v for k, v in dictionary.items() if k != -1}
    return [k for k, v in sorted(filtered_dict.items(), key=lambda item: len(item[1]), reverse=True)[:4]]

def train_val_sort(directory_list):
    return sorted(directory_list, key=lambda x: ('val' in x, 'train' not in x, x))

def extract_key(path):
    filename = os.path.basename(path)
    name, _ = os.path.splitext(filename)
    parts = name.split('_')
    return '_'.join(parts[:-1]) if len(parts) > 1 else name

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

        if data['answer'] in ['yes', 'no']:
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
 

def preprocess_data(input_file, max_num, task_key=None, top_keys=None, preprocess_require=True):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    task_datasets = defaultdict(list)

    for item_id, item in enumerate(data):

        if 'function' in input_file: 
            if "knowledge" in input_file:
                task_id = get_knowledge_task_mapping(item['question'])
                key_image_id = 'image_id' 
            elif 'logical' in input_file:
                task_id = get_logical_task_mapping(item)
                item = item['gqa_question']
                key_image_id = 'imageId'
            elif 'scenetext' in input_file:
                task_id = get_scenetext_task_mapping(item)
                key_image_id = 'image_id'
            elif 'attribute' in input_file:
                task_id = get_attribute_task_mapping(item['question'])
                key_image_id = 'image_id'
            else:
                item = item['gqa_question']
                structural_type = item["types"].get("structural", "unknown")
                semantic_type = item["types"].get("semantic", "unknown")
                task_id = task_mapping.get(structural_type, {}).get(semantic_type, -1)
                key_image_id = 'imageId'
        else:
            key_image_id = 'image_id'
            task_id = get_scene_task_mapping(item['question'])


        if task_id == -1:
            pass
            #raise ValueError("Not defined task")
        
        answer = most_frequent_element(item['answer']) if type(item['answer']) is list else item['answer']
        task_datasets[task_id].append({
            "id": item_id,
            "question": item['question'],
            "answer": item['answer'],
            "image": f"{base_dir}/images/{item[key_image_id]}.jpg",
        })

    new_task_datasets = defaultdict(list)

    if task_key is None:
        task_key = extract_key(input_file)
        top_keys = top_4_keys_by_items(task_datasets)
    
    if preprocess_require:

        for task_id, dataset in task_datasets.items():
            if task_id not in top_keys:
                continue

            # Subsampling to a maximum of 10000 samples
            sampled_dataset = random.sample(dataset, min(len(dataset), max_num))

            # Save dataset to file
            # output_file = f"dataset-{task_id//4}{task_id%4}.json"
            random.shuffle(sampled_dataset)
            new_task_datasets[task_id] = sampled_dataset

            print("task_id", task_id, "whole", len(dataset), "sampled", len(sampled_dataset))
            # Copy selected images to target directory
            for item in sampled_dataset:
                source_path = item["image"].replace('images', 'raw_images')
                if not os.path.exists(source_path):
                    source_path = source_path.replace('raw_images', 'raw_images2')
                target_path = item["image"]
                if os.path.exists(source_path):
                    shutil.copy(source_path, target_path)
                    pass
                else:
                    print(source_path)    
    return new_task_datasets, task_key, top_keys


def combine_results(results, out_json):
    # Initialize combined data lists
    combined_datalist = []

    # Combine results from all processes
    for result in results:
        combined_datalist.extend(result['datalist'])

    print(out_json, len(combined_datalist))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined_datalist, f, indent=4, ensure_ascii=False)

base_dir = 'dataset/CLVQA'
#dir_type = 'train'

function_dir = os.listdir(os.path.join(base_dir, 'function'))
scene_dir = os.listdir(os.path.join(base_dir, 'scene'))
json_dirs = [os.path.join(base_dir, 'function', f_dir) for f_dir in function_dir] + [os.path.join(base_dir, 'scene', s_dir) for s_dir in scene_dir]
json_dirs = train_val_sort(json_dirs)
max_num_dic = {'train':5000, 'test':500}
os.makedirs(f'{base_dir}/train', exist_ok=True)
os.makedirs(f'{base_dir}/test', exist_ok=True)
os.makedirs(f'{base_dir}/images', exist_ok=True)

task_subtask = defaultdict()
preprocessed_datalists = defaultdict()

for task_idx, input_file in enumerate(json_dirs):
    if 'train' in input_file:
        dir_type = 'train'
    else:
        dir_type = 'test'

    max_num = max_num_dic[dir_type]
    
    print(input_file)

    preprocess_require = True
    if os.path.exists(os.path.join(base_dir, dir_type, f"dataset-{task_idx}{3}.json")):
        preprocess_require = False

    if dir_type == 'train':
        datalists, task_key, top_keys = preprocess_data(input_file, max_num, preprocess_require=preprocess_require)
        task_subtask[task_key] = top_keys
    else:
        task_key = extract_key(input_file)
        top_keys = task_subtask[task_key]
        datalists, _, _ = preprocess_data(input_file, max_num, task_key, top_keys, preprocess_require=preprocess_require)

    if not preprocess_require:
        print("file exists", input_file)
        continue

    ### Preprocessing per task ###
    for task_num, task in enumerate(sorted(list(datalists.keys()))):
        num_chunks = min(NUM_PROCESSES, len(datalists[task]))
        #sampled_task_datalist = np.random.choice(datalists[task], size=min(max_num, len(datalists[task])), replace=False).tolist()
        chunk_size = math.ceil(len(datalists[task]) / num_chunks)
        print(f"Processing {len(datalists[task])} items using {num_chunks} processes")

        # Prepare chunks for processing
        chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(datalists[task]))
            print(f"chunk {i}, {len(datalists[task][start_idx:end_idx])}")
            chunk_data = {
                'chunk_id': i,
                'datalists': datalists[task][start_idx:end_idx],
                'data_dir': base_dir,
                'type_name': dir_type,
                'task': task
            }
            chunks.append(chunk_data)

        # Process chunks in parallel
        with Pool(processes=num_chunks) as pool:
            results = pool.map(process_data_chunk, chunks)
        out_json = os.path.join(base_dir, dir_type, f"dataset-{task_idx}{task_num}.json")
        print(f"dataset-{task_idx}{task_num}.json stored")
        # Combine results from all processes
        combine_results(results, out_json)
