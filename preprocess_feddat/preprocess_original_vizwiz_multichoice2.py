import json
import os
import random
import openai
from openai import OpenAI
import time
import jsonlines
from multiprocessing import Process, Queue, Manager, Pool, cpu_count
import math

NUM_SECONDS_TO_SLEEP = 0.01
NUM_PROCESSES = cpu_count()  # Use all available CPU cores

# Set up OpenAI client
client = OpenAI(
    api_key="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"
)

random.seed(42)

output_folder = 'dataset/VizWiz-Original'
train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')

# Create output directories if they don't exist
for folder in [train_folder, test_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_eval(content: str, max_tokens: int):
    user_content = [
        {"type": "text", "text": content['text']},
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

rule = {
    "role": "Assistant",
    "prompt": "Please generate two or three answer choices for the given question asking about an image and the given correct answer.\n"
              "The possible answers are the variation of correct and wrong answers. You should use it to infer about the image.\n"
              "You should:\n"
              "1. generate 3 choices if the correct answer is \"unanswerable\".\n"
              "2. generate 2 choices not in Possible Answers if the correct answer is not \"unanswerable\".\n"
              "3. NOT INCLUDE unanswerable or correct answer in the generated choices.\n"
              "4. generate choices relevant to the question, but should not be confusing with the correct answer. Specifically, choices and answers should not be synonyms, nor should they have a relationship of inclusion.\n"
              "You must return the choices by strictly following this format:\"[[Choice A | Choice B]]\", for example: \"Choice list: [[red | blue | pink]]\"."
}

def process_data_chunk(chunk_data):
    process_id = os.getpid()
    chunk_id = chunk_data['chunk_id']
    data_chunk = chunk_data['data_chunk']
    is_train = chunk_data['is_train']
    
    print(f"Process {process_id} starting on chunk {chunk_id} with {len(data_chunk)} items")
    
    output_datalist1 = []
    output_datalist2 = []
    output_datalist3 = []
    output_datalist4 = []
    
    # Create a process-specific review file
    if is_train:
        review_file_path = f'vizwiz_choice_list_train_process_{chunk_id}.jsonl'
    else:
        review_file_path = f'vizwiz_choice_list_test_process_{chunk_id}.jsonl'
    
    with open(review_file_path, 'w') as review_file:
        for idx, item in enumerate(data_chunk):
            if idx % 10 == 0:
                print(f"Process {process_id}, Chunk {chunk_id}: Processing item {idx}/{len(data_chunk)}")
            
            image_file = item['image']
            question = item['question']
            answers = {}
            for ans in item['answers']:
                if ans['answer_confidence'] in ["yes", "maybe", "no"]:
                    if ans['answer'] not in answers.keys():
                        answers[ans['answer']] = [1, ans['answer_confidence']]
                    else:
                        answers[ans['answer']][0] += 1
                        if (answers[ans['answer']][1] == 'no' and (ans['answer_confidence'] == 'maybe' or ans['answer_confidence'] == 'yes')) or \
                           (answers[ans['answer']][1] == 'maybe' and ans['answer_confidence'] == 'yes'):
                            answers[ans['answer']][1] = ans['answer_confidence']
            
            # for k1 in answers.keys():
            #     for k2 in answers.keys():
            #         if k1 == k2:
            #             continue
            #         if k2 in k1:
            #             answers[k2][0] += 1
            
            question_type = item['question_type']
            if item['answerable'] == 0:
                answer = 'unanswerable'
            else:
                max_cnt = 0
                answer = ""
                for candidate, cnt in answers.items():
                    if candidate == 'unanswerable': continue
                    if (cnt[1] == 'yes' or cnt[1] == 'maybe') and cnt[0] > max_cnt:
                        answer = candidate
                        max_cnt = cnt[0]
            if answer == "":
                continue

            if answer == "yes" or answer == "no":
                answer_list = ["yes", "no", "unanswerable"]
            else:
                answer_keys = list(answers.keys())
                if answer != "unanswerable":
                    if "unanswerable" in answer_keys:
                        index = answer_keys.index("unanswerable")
                        del answer_keys[index]
                
                content = {
                    "text": f"[Instruction]\n{rule['prompt']}\n\n"
                            f"[Question]\n{question}\n\n"
                            f"[Correct Answer]\n{answer}\n\n"
                            f"[Possible Answers]\n[{', '.join(answer_keys)}]\n"
                }
                review = get_eval(content, 512)
                answer_list = parse_score(review).split(' | ')
                
                review_file.write(json.dumps({"question": question, "gpt_response": review, "answer": answer}) + "\n")

                answer_list.append(answer)
                if 'unanswerable' not in answer_list:
                    answer_list.append('unanswerable')
            random.shuffle(answer_list)
                
            instruction = "\nWhen the provided information is insufficient, respond with 'unanswerable'.\nAnswer the question using the choices from the choice list."
            json_data = {
                "id": image_file.split('.jpg')[0],
                "image": [os.path.join(output_folder, 'images', 'train', image_file)] if is_train else [os.path.join(output_folder, 'images', 'val', image_file)],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{question}{instruction}\nChoice list:[{', '.join(answer_list)}]. Your answer is: "
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }
            
            if question_type == 1:
                output_datalist1.append(json_data)
            elif question_type == 3 or question_type == 4:
                output_datalist2.append(json_data)
            elif question_type == 2 or question_type == 8 or question_type == 10:
                output_datalist3.append(json_data)
            else:
                output_datalist4.append(json_data)
    
    # Determine output folder based on train/test
    target_folder = train_folder if is_train else test_folder
    
    # Save output files with process-specific names
    with open(f'{target_folder}/dataset-0_chunk_{chunk_id}.json', 'w') as json_file:
        json.dump(output_datalist1, json_file, indent=4)
    with open(f'{target_folder}/dataset-1_chunk_{chunk_id}.json', 'w') as json_file:
        json.dump(output_datalist2, json_file, indent=4)
    with open(f'{target_folder}/dataset-2_chunk_{chunk_id}.json', 'w') as json_file:
        json.dump(output_datalist3, json_file, indent=4)
    with open(f'{target_folder}/dataset-3_chunk_{chunk_id}.json', 'w') as json_file:
        json.dump(output_datalist4, json_file, indent=4)
    
    print(f"Process {process_id} finished processing chunk {chunk_id}")
    
    return {
        'chunk_id': chunk_id,
        'datalist1': output_datalist1,
        'datalist2': output_datalist2,
        'datalist3': output_datalist3,
        'datalist4': output_datalist4
    }

def combine_results(results, is_train):
    target_folder = train_folder if is_train else test_folder
    
    # Initialize combined data lists
    combined_datalist1 = []
    combined_datalist2 = []
    combined_datalist3 = []
    combined_datalist4 = []
    
    # Combine results from all processes
    for result in results:
        combined_datalist1.extend(result['datalist1'])
        combined_datalist2.extend(result['datalist2'])
        combined_datalist3.extend(result['datalist3'])
        combined_datalist4.extend(result['datalist4'])
    
    # Save combined results
    with open(f'{target_folder}/dataset-0_L2.json', 'w') as json_file:
        json.dump(combined_datalist1, json_file, indent=4)
    with open(f'{target_folder}/dataset-1_2.json', 'w') as json_file:
        json.dump(combined_datalist2, json_file, indent=4)
    with open(f'{target_folder}/dataset-2_2.json', 'w') as json_file:
        json.dump(combined_datalist3, json_file, indent=4)
    with open(f'{target_folder}/dataset-3_2.json', 'w') as json_file:
        json.dump(combined_datalist4, json_file, indent=4)
    
    # Combine the review files
    combined_review_path = f'vizwiz_choice_list_{"train" if is_train else "test"}.jsonl'
    with open(combined_review_path, 'w') as combined_file:
        for chunk_id in range(len(results)):
            chunk_review_path = f'vizwiz_choice_list_{"train" if is_train else "test"}_process_{chunk_id}.jsonl'
            if os.path.exists(chunk_review_path):
                with open(chunk_review_path, 'r') as chunk_file:
                    combined_file.write(chunk_file.read())
    
    # Remove individual chunk files
    for chunk_id in range(len(results)):
        chunk_review_path = f'vizwiz_choice_list_{"train" if is_train else "test"}_process_{chunk_id}.jsonl'
        if os.path.exists(chunk_review_path):
            os.remove(chunk_review_path)
    
    print(f"Combined all results into final dataset files")

def preprocess_vizwiz_original(original_json, is_train=True):
    # Load the original data
    print(f"Loading data from {original_json}")
    original_datalist = json.load(open(original_json, 'r'))
    
    # Determine number of chunks based on available CPUs
    num_chunks = min(NUM_PROCESSES, len(original_datalist))
    chunk_size = math.ceil(len(original_datalist) / num_chunks)
    
    print(f"Processing {len(original_datalist)} items using {num_chunks} processes")
    
    # Prepare chunks for processing
    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(original_datalist))
        chunk_data = {
            'chunk_id': i,
            'data_chunk': original_datalist[start_idx:end_idx],
            'is_train': is_train
        }
        chunks.append(chunk_data)
    
    # Process chunks in parallel
    with Pool(processes=num_chunks) as pool:
        results = pool.map(process_data_chunk, chunks)
    
    # Combine results from all processes
    combine_results(results, is_train)

if __name__ == "__main__":
    # Process train and test data
    for json_path in [('train_split.json', True)]:  # Add ('val_split.json', False) when needed ('train_split.json', True)
        json_path, is_train = json_path
        preprocess_vizwiz_original(os.path.join(output_folder, json_path), is_train)