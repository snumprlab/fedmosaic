import shutil
import json 
import os
import numpy as np
import random
from collections import defaultdict
import openai
from openai import OpenAI
import time
from multiprocessing import Process, Queue, Manager, Pool, cpu_count
import math


ATTRIBUTE_KEYWORDS = ["color", "size", "shape", "type", "kind", "pattern", "style", "hue", "shade", 
    "tint", "big", "small", "large", "height", "width", "length", "depth", 
    "form", "appearance", "contour", "outline", "texture", "material", "fabric", 
    "surface", "design", "decoration", "print", "motif", "fashion", "aesthetic", 
    "theme", "art style", "brightness", "contrast", "density", "opacity"]

data_type = 'train'
question_file_path = f"MultipleChoice_abstract_v002_{data_type}2015_questions.json"
answer_file_path = f"abstract_v002_{data_type}2015_annotations.json"
data_path = "dataset/Abstract"
max_num = 10000 if data_type == 'train' else 1000


NUM_SECONDS_TO_SLEEP = 0.01
NUM_PROCESSES = 10 #cpu_count()  # Use all available CPU cores

# Set up OpenAI client
client = OpenAI(
    api_key="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"
    )

rule = {
    "role": "Assistant",
    "prompt": "Please generate three answer choices for the given question asking about a image referring the given correct answer and the given candidates.\n"
              "The possible answers are the variation of correct and wrong answers. You should use it to infer about the image.\n"
              "You should:\n"
              "1. Select hard negative choices (i.e., Not entirely incorrect, but something worth reconsidering against the correct answer) from the given candidates. If fewer than three valid choices exist, generate additional choices to make a total of three.\n"
              "2. NOT INCLUDE correct answer in the generated choices.\n"
              "3. generate choices relevant to the question, but should not be confusing with the correct answer. Specifically, choices and answers should not be synonyms, nor should they have a relationship of inclusion.\n"
              "You must return the choices by strictly following this format:\"[[Choice A | Choice B | Choice C]]\", for example: \"Choice list: [[red | blue | pink]]\"."
}

def get_eval(content: str, max_tokens: int):
    user_content = [
                        {"type":"text", "text":content['text']},
                    ]
    
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

'''
def make_json_files(sampled_data):
    json_data = []
    for idx, data in enumerate(sampled_data):
        new_data = {}
        new_data['id'] = idx
        new_data['image'] = data['image']
        shutil.copy(os.path.join(data_path, f"{data_type}_images", f"abstract_v002_{data_type}2015_{str(data['image_id']).zfill(12)}.png"), os.path.join(data_path, "images", f"abstract_v002_{data_type}2015_{str(data['image_id']).zfill(12)}.png"))

        new_data['conversations'] = [
            {
                "from": "human",
                "value": f"Please respond accurately to the following query. You must choose your answer from the Choice List.\n<image>Question: {data['question']}\nChoice list:[{data['candidates']}].\n Your answer is:"
            },
            {
                "from": "gpt",
                "value": data['answer']
            }
        ]
        json_data.append(new_data)
    return json_data
'''

def process_data_chunk(chunk_data):
    process_id = os.getpid()
    chunk_id = chunk_data['chunk_id']
    datalists = chunk_data['datalists']
    data_dir = chunk_data['data_dir']
    type_name = chunk_data['type_name']
    task = chunk_data['task']

    new_datalists = []
    for idx, data in enumerate(datalists):
        print(chunk_id, f"{idx}/{len(datalists)}")
        new_data = {}
        new_data['question'] = data['question']
        new_data['answer'] = data['answer']
        new_data['image_id'] = data['image_id']
        new_data['image'] = data['image'] 
        content = {
            "text": (f"[Instruction]\n{rule['prompt']}\n\n"
                    f"[Question]\n{data['question']}\n\n"
                    f"[Candidates]\n{data['candidates']}\n\n"
                    f"[Correct Answer]\n{data['answer']}\n")
            }
        
        if task == 'yes/no':
            sampled_candidates = ['yes'] if data['answer'] == 'no' else ['no']
        else:
            review = get_eval(content, 512)
            sampled_candidates = parse_score(review).split(' | ')
        sampled_candidates += [new_data['answer']]
        random.shuffle(sampled_candidates)

        shutil.copy(os.path.join(data_path, f"{data_type}_images", f"abstract_v002_{data_type}2015_{str(data['image_id']).zfill(12)}.png"), os.path.join(data_path, "images", f"abstract_v002_{data_type}2015_{str(data['image_id']).zfill(12)}.png"))

        new_data['conversations'] = [
            {
                "from": "human",
                "value": f"Please respond accurately to the following query. You must choose your answer from the Choice List.\n<image>Question: {data['question']}\nChoice list:[{sampled_candidates}].\n Your answer is:"
            },
            {
                "from": "gpt",
                "value": data['answer']
            }
        ]


        # sampled_candidates = [new_data['answer']] + random.sample(candidates, min(3, len(candidates)))
        # random.shuffle(sampled_candidates)
        new_datalists.append(new_data)

    return {
        'chunk_id': chunk_id,
        'datalist': new_datalists
    }


'''
os.makedirs(os.path.join(data_path, data_type), exist_ok=True)
for task_num, task_type in enumerate(list(datalists.keys())):
    json_data = []
    sampled_data = np.random.choice(datalists[task_type], size=min(max_num, len(datalists[task_type])), replace=False).tolist()
    for idx, data in enumerate(sampled_data):
        new_data = {}
        new_data['id'] = idx
        new_data['image'] = data['image']
        shutil.copy(os.path.join(data_path, f"{data_type}_images", f"abstract_v002_{data_type}2015_{str(data['image_id']).zfill(12)}.png"), os.path.join(data_path, "images", f"abstract_v002_{data_type}2015_{str(data['image_id']).zfill(12)}.png"))

        new_data['conversations'] = [
            {
                "from": "human",
                "value": f"Please respond accurately to the following query. You must choose your answer from the Choice List.<image>Question: {data['question']}\nChoice list:[{data['candidates']}]. Your answer is:"
            },
            {
                "from": "gpt",
                "value": data['answer']
            }
        ]
        json_data.append(new_data)
    print(f"{task_type} sampled: {len(json_data)}, original: {len(datalists[task_type])}")
    with open(os.path.join(data_path, data_type, f"dataset-{task_num}.json"), 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
'''


def combine_results(results, out_json):
    # Initialize combined data lists
    combined_datalist = []

    # Combine results from all processes
    for result in results:
        combined_datalist.extend(result['datalist'])

    print(out_json, len(combined_datalist))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined_datalist, f, indent=4, ensure_ascii=False)


os.makedirs(os.path.join(data_path, data_type), exist_ok=True)
os.makedirs(os.path.join(data_path, "images"), exist_ok=True)

with open(f"{data_path}/{question_file_path}", "r", encoding="utf-8") as file:
    question_datas = json.load(file)

with open(f"{data_path}/{answer_file_path}", "r", encoding="utf-8") as file:
    answer_datas = json.load(file)

print(len(question_datas['questions']))
print(len(answer_datas['annotations']))


### Task split ###
datalists = defaultdict(list)
for question_data, answer_data in zip(question_datas['questions'], answer_datas['annotations']):
    new_data = {}
    if str(question_data['question_id']) != str(answer_data['question_id']):
        raise ValueError("question id different!")
    new_data['question'] = question_data['question']
    new_data['answer'] = answer_data['multiple_choice_answer']
    question_data['multiple_choices'].remove(new_data['answer'])
    new_data['candidates'] = question_data['multiple_choices']
    new_data['image_id'] = question_data['image_id']
    new_data['image'] = os.path.join(data_path, "images", f"abstract_v002_{data_type}2015_{str(question_data['image_id']).zfill(12)}.png")
    
    '''
    content = {
        "text": (f"[Instruction]\n{rule['prompt']}\n\n"
                f"[Question]\n{new_data['question']}\n\n"
                f"[Candidates]\n{question_data['multiple_choices']}\n\n"
                f"[Correct Answer]\n{new_data['answer']}\n")
        }
    review = get_eval(content, 512)
    sampled_candidates = parse_score(review).split(' | ')
    sampled_candidates += [answer]
    random.shuffle(sampled_candidates)
    '''
    if answer_data['answer_type'] == 'number':
        answer_type = 'number'
        # candidates = [x for x in question_data['multiple_choices'] if x.isdigit()]
    elif answer_data['answer_type'] == 'yes/no':
        # candidates = ['yes'] if new_data['answer']=='no' else ['no']
        answer_type = 'yes/no'
    else:
        answer_type = 'remaining'
        for keyword in ATTRIBUTE_KEYWORDS:
            if keyword in question_data['question']:
                answer_type = 'attribute'
    
    # sampled_candidates = [new_data['answer']] + random.sample(candidates, min(3, len(candidates)))
    # random.shuffle(sampled_candidates)
    datalists[answer_type].append(new_data)


### Preprocessing per task ###
for task_num, task in enumerate(sorted(list(datalists.keys()))):

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
            'data_dir': data_path,
            'type_name': data_type,
            'task': task
        }
        chunks.append(chunk_data)

    # Process chunks in parallel
    with Pool(processes=num_chunks) as pool:
        results = pool.map(process_data_chunk, chunks)
    out_json = os.path.join(data_path, data_type, f"dataset-{task_num}.json")
    # Combine results from all processes
    combine_results(results, out_json)

