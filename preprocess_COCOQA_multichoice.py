import base64
import random
import numpy as np
import json
import os
import shutil
import openai
from openai import OpenAI
import time
from multiprocessing import Process, Queue, Manager, Pool, cpu_count
import math

data_dir = 'dataset/COCOQA'
type_name = 'test'
max_num = 10000 if type_name == 'train' else 1000
os.makedirs(os.path.join(data_dir, type_name), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'images'), exist_ok=True)

NUM_SECONDS_TO_SLEEP = 0.01
NUM_PROCESSES = 6 #cpu_count()  # Use all available CPU cores

# Set up OpenAI client
client = OpenAI(
    api_key="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"
)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_eval(content: str, max_tokens: int):
    user_content = [
                        {"type":"text", "text":content['text']},
                    ]
    '''
    for base64_image in content['image']:
        user_content.append(
                            {"type": "image_url",
                             "image_url":{"url": f"data:image/jpeg;base64,{base64_image}"}}
                            )
    '''
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

rule = {
    "role": "Assistant",
    "prompt": "Please generate three answer choices for the given question asking about the given image and the given correct answer.\n"
              "The possible answers are the variation of correct and wrong answers. You should use it to infer about the image.\n"
              "You should:\n"
              "1. generate 3 choices.\n"
              "2. NOT INCLUDE correct answer in the generated choices.\n"
              "3. generate choices relevant to the question, but not too similar to the correct answer.\n"
              "You must return the choices by strictly following this format:\"[[Choice A | Choice B | Choice C]]\", for example: \"Choice list: [[red | blue | pink]]\"."
}


def process_data_chunk(chunk_data):
    process_id = os.getpid()
    chunk_id = chunk_data['chunk_id']
    question_chunk = chunk_data['question_chunk']
    img_id_chunk = chunk_data['img_id_chunk']
    answer_chunk = chunk_data['answer_chunk']
    data_dir = chunk_data['data_dir']
    type_name = chunk_data['type_name']
    
    print(f"Process {process_id} starting on chunk {chunk_id} with {len(question_chunk)} items")

    datalist = []
    review_file_path = f'{data_dir}/vizwiz_choice_list_{type_name}_process_{chunk_id}.jsonl'
    
    with open(review_file_path, 'w') as review_file:
        for idx, (img_id, question, answer) in enumerate(zip(img_ids, questions, answers)):

            try:
                shutil.copy(f"dataset/coco_images/train/{img_id.zfill(12)}.jpg", f"{data_dir}/images/{img_id.zfill(12)}.jpg")
            except:
                shutil.copy(f"dataset/coco_images/val/{img_id.zfill(12)}.jpg", f"{data_dir}/images/{img_id.zfill(12)}.jpg")

            base64_imgs = []
            image_paths = f"{data_dir}/images/{img_id.zfill(12)}.jpg"
            if isinstance(image_paths, list):
                for img_path in image_paths:
                    base64_imgs.append(encode_image(img_path))
            else:
                base64_imgs.append(encode_image(image_paths))

            content = {
                "text": (f"[Instruction]\n{rule['prompt']}\n\n"
                        f"[Question]\n{question}\n\n"
                        f"[Correct Answer]\n{answer}\n"),
                "image":
                   base64_imgs
                }

            review = get_eval(content, 512)
            answer_candidate_list = parse_score(review).split(' | ')
            answer_candidate_list += [answer]
            random.shuffle(answer_candidate_list)
            review_file.write(json.dumps({"question": question, "gpt_response": review, "answer": answer}) + "\n")

            entry = {
                "id": idx,
                "image": f"{data_dir}/images/{img_id.zfill(12)}.jpg",
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n Question: {question} \n Choice list:{answer_candidate_list} \n Your answer is:"
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }

            datalist.append(entry)
            img_ids.append(img_id)

    return {
        'chunk_id': chunk_id,
        'datalist': datalist,
    }


def combine_results(results, task_num, out_json):
    target_folder = train_folder
    
    # Initialize combined data lists
    combined_datalist = []
    
    # Combine results from all processes
    for result in results:
        combined_datalist.extend(result['datalist'])
    
    print(task_num, len(datalist))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(datalist, f, indent=4, ensure_ascii=False)
    

for task_num in range(4):
    txt_questions = f"{data_dir}/{type_name}_annotations/type_{task_num}/questions.txt"
    txt_img_ids = f"{data_dir}/{type_name}_annotations/type_{task_num}/img_ids.txt"
    txt_answers = f"{data_dir}/{type_name}_annotations/type_{task_num}/answers.txt"
    out_json = f"{data_dir}/{type_name}/dataset-{task_num}.json"

    # 파일 읽기
    with open(txt_questions, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f.readlines()]

    with open(txt_img_ids, "r", encoding="utf-8") as f:
        img_ids = [line.strip() for line in f.readlines()]

    with open(txt_answers, "r", encoding="utf-8") as f:
        answers = [line.strip() for line in f.readlines()]

    # Determine number of chunks based on available CPUs
    num_chunks = min(NUM_PROCESSES, len(questions))
    chunk_size = math.ceil(len(questions) / num_chunks)
    
    print(f"Processing {len(questions)} items using {num_chunks} processes")
    
    # Prepare chunks for processing
    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(questions))
        chunk_data = {
            'chunk_id': i,
            'question_chunk': questions[start_idx:end_idx],
            'img_id_chunk': img_ids[start_idx:end_idx],
            'answer_chunk': answers[start_idx:end_idx],
            'data_dir': data_dir,
            'type_name': type_name
        }
        chunks.append(chunk_data)
    
    # Process chunks in parallel
    with Pool(processes=num_chunks) as pool:
        results = pool.map(process_data_chunk, chunks)
    
    # Combine results from all processes
    combine_results(results, task_num, out_json)
