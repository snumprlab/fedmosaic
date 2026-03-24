import json
import os
import random
import openai
from openai import OpenAI
import time
import jsonlines
from multiprocessing import Process, Pool, cpu_count
import math

NUM_SECONDS_TO_SLEEP = 0.01
NUM_PROCESSES = cpu_count()  # Use all available CPU cores

client = OpenAI(
    api_key=""
)

random.seed(42)

output_folder = 'dataset/VizWiz-Original'
train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')

# Create output directories if they don't exist
for folder in [train_folder, test_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_eval(content: str, max_tokens: int, process_id=None):
    user_content = [
        {"type": "text", "text": content['text']},
    ]

    while True:
        try:
            response = client.chat.completions.create(
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant to categorize given question.'
                }, {
                    'role': 'user',
                    'content': user_content
                }],
                model="gpt-4o-mini-2024-07-18",
                temperature=0.0,
                max_tokens=max_tokens,
            )
            break
        except openai.APIConnectionError as e:
            print(f"Process {process_id}: The server could not be reached")
            print(e.__cause__)
        except openai.RateLimitError as e:
            print(f"Process {process_id}: A 429 status code was received; backing off")
            time.sleep(5)  # Back off longer on rate limit errors
        except openai.APIStatusError as e:
            print(f"Process {process_id}: Another non-200-range status code was received")
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
        print(e)
        print('error', review)
        return -1

rule = {
    "role": "Assistant", 
    "prompt": 
        "Please act as a classifier that classifies the given question asking about an image into one of the 4 question types displayed below. The 4 question types are:\n"
        "1. Object Identification Questions: Questions asking to identify an object, such as \"What is this?\", \"What is this item?\", or \"What is this product?\"\n"
        "2. Color Identification Questions: Questions specifically asking about the color of an object, like \"What color is this?\", \"What color are my shoes?\", or \"What color is this shirt?\"\n"
        "3. Food and Drink Identification: Questions related to identifying food or beverages, such as \"What is in this can?\", \"What flavor is this?\", or \"What type of drink is this?\"\n"
        "4. Brand and Label Identification: Questions asking to identify the brand or text on a label, such as \"What brand is this?\", \"What does this label say?\", or \"What is written on this bottle?\"\n"
        "5. Size or Quantity Questions: Questions asking for quantities, measurements, or sizes, such as \"What size is this shirt?\", \"How much is this?\", or \"How many calories are in this drink?\"\n"
        "6. Instruction or Direction Questions: Questions asking for instructions, such as \"What are the instructions for this product?\", or \"What does this sign say?\"\n"
        "7. Location or Setting Questions: Questions asking for details about a location, setting, or environment, such as \"Where is this?\", \"What room is this?\", or \"Is this the backyard?\"\n"
        "8. Product Type or Category: Questions asking to categorize or identify the type of product, like \"What kind of soup is this?\", \"What type of cereal is this?\", or \"What type of coffee is this?\"\n"
        "9. Time or Date Related Questions: Questions asking about time, such as \"What is the expiration date?\", \"When does this expire?\", or \"What time is it?\"\n"
        "10. Device or Equipment Identification: Questions related to identifying devices or equipment, such as \"What kind of TV is this?\", \"What is this computer screen showing?\", or \"What is this exercise machine?\"\n"
        "Begin your evaluation by providing a short explanation. Be as objective as possible."
        " After providing your explanation, you must return the type of the question among 1 to 10 in a single integer by strictly following this format:\"[[type]]\", for example: \"Question type: [[2]]\"."
}

def process_data_chunk(chunk_data):
    process_id = os.getpid()
    chunk_id = chunk_data['chunk_id']
    data_chunk = chunk_data['data_chunk']
    split_name = chunk_data['split_name']
    
    print(f"Process {process_id} starting on chunk {chunk_id} with {len(data_chunk)} items")
    
    new_datalist = []
    
    # Create a process-specific review file
    review_file_path = f'vizwiz_type_explanation2_process_{chunk_id}.jsonl'
    
    with open(review_file_path, 'w') as review_file:
        for idx, item in enumerate(data_chunk):
            if idx % 10 == 0:
                print(f"Process {process_id}, Chunk {chunk_id}: Processing item {idx}/{len(data_chunk)}")
            
            question = item['question']
            
            content = {
                "text": f"[Instruction]\n{rule['prompt']}\n\n"
                        f"[Question]\n{question}\n"
            }
            
            review = get_eval(content, 512, process_id)
            question_type = parse_score(review)
            if int(question_type) == -1:
                continue
            else:
                item['question_type'] = int(question_type)
            
            review_file.write(json.dumps({"question": question, "gpt_response": review}) + "\n")
            new_datalist.append(item)
    
    print(f"Process {process_id} finished processing chunk {chunk_id}")
    
    return {
        'chunk_id': chunk_id,
        'new_datalist': new_datalist
    }

def combine_results(results, split_name):
    # Initialize combined data list
    combined_datalist = []
    
    # Combine results from all processes
    for result in results:
        combined_datalist.extend(result['new_datalist'])
    
    # Save combined results
    output_path = os.path.join(output_folder, f'{split_name}_split.json')
    with open(output_path, 'w') as json_file:
        json.dump(combined_datalist, json_file, indent=4)
    
    # Combine the review files
    combined_review_path = 'vizwiz_type_explanation2.jsonl'
    
    # Check if the file exists - if not, create it; if yes, append to it
    mode = 'a' if os.path.exists(combined_review_path) else 'w'
    
    with open(combined_review_path, mode) as combined_file:
        for chunk_id in range(len(results)):
            chunk_review_path = f'vizwiz_type_explanation2_process_{chunk_id}.jsonl'
            if os.path.exists(chunk_review_path):
                with open(chunk_review_path, 'r') as chunk_file:
                    combined_file.write(chunk_file.read())
    
    # Remove individual chunk files
    for chunk_id in range(len(results)):
        chunk_review_path = f'vizwiz_type_explanation2_process_{chunk_id}.jsonl'
        if os.path.exists(chunk_review_path):
            os.remove(chunk_review_path)
    
    print(f"Combined all results for {split_name} into final dataset file")
    return len(combined_datalist)

def categorize_question_type_parallel(split_name):
    # Load the original data
    input_path = os.path.join(output_folder, f'{split_name}.json')
    print(f"Loading data from {input_path}")
    
    original_datalist = json.load(open(input_path, 'r'))
    
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
            'split_name': split_name
        }
        chunks.append(chunk_data)
    
    # Process chunks in parallel
    with Pool(processes=num_chunks) as pool:
        results = pool.map(process_data_chunk, chunks)
    
    # Combine results from all processes
    total_processed = combine_results(results, split_name)
    
    print(f"Processed {total_processed} items from {split_name}")
    return total_processed

if __name__ == "__main__":
    # Process validation and training data
    # For validation data
    # val_count = categorize_question_type_parallel('val')
    # print(f"Processed {val_count} validation items")
    
    # For training data
    train_count = categorize_question_type_parallel('train')
    print(f"Processed {train_count} training items")
    
    print("Question type categorization complete!")