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

output_folder = 'dataset/Fed-aya2'
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
                    'content': 'You are a helpful and precise assistant to determine the category of question-response pair'
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
        # Find the last occurrence of [[ and ]]
        start_idx = review.rfind('[[')
        if start_idx == -1:
            raise ValueError("No [[ found in text")
        
        end_idx = review.find(']]', start_idx)
        if end_idx == -1:
            raise ValueError("No matching ]] found after the last [[")
        
        # Extract the content between the last [[ and its matching ]]
        score = review[start_idx + 2:end_idx]
        return score
    except Exception as e:
        print(f"Error: {e}")
        print(f"Problem text: {review}")
        return -1

rule = {
    "role": "Assistant", 
    "prompt": 
        "Please act as a classifier that classifies the given question-response pair written in various langauges with super-category and sub-category.\n"
        "Super-category is the primary domain, the overarching field the question and answer are related to, for example: Science, History, Arts.\n"
        "You must choose super-category from the given list of super-categories.\n"
        "Sub-category involves specific fields or topics within the domain, for example: Machine Learning, Movie.\n"
        "Even though the question-answer pair is not written in English, please specify the categories in English.\n"
        "Begin your evaluation by providing a short explanation. Be as objective as possible."
        " After providing your explanation, you must return the super-category and sub-category by strictly following this format:\"[[super-category | sub-category]]\", for example: \"Category: [[Eduation | Machine Learning]]\"."
}

super_categories = [
    'Science',
    'Mathematics',
    'Technology & Engineering',
    'Philosophy',
    'Geography',
    'History',
    'Psychology',
    'Politics',
    'Economics & Finance',
    'Government & Law',
    'Health & Medicine',
    'Social Sciences',
    'Literature',
    'Linguistics',
    'Arts',
    'Education & Knowledge',
    'Food',
    'Entertainment',
    'Sports & Exercise',
    'Fashion & Beauty',
    'Hobbies & Crafts',
    'Career & Business',
    'Environment & Nature',
    'Culture',
    'Travel',
    'News',
    'Agriculture',
    'Safety & Public Safety',
    'Lifestyle & Home',
    'Religion & Spirituality',
    'Personal Development',
    'Military & Defense',
    'Parenting & Children',
]

def process_data_chunk(chunk_data):
    process_id = os.getpid()
    chunk_id = chunk_data['chunk_id']
    data_chunk = chunk_data['data_chunk']
    split_name = chunk_data['split_name']
    
    print(f"Process {process_id} starting on chunk {chunk_id} with {len(data_chunk)} items")
    
    new_datalist = []
    
    # Create a process-specific review file
    review_file_path = f'fed_aya_topic_explanation_process_{chunk_id}.jsonl'
    
    with open(review_file_path, 'w') as review_file:
        for idx, item in enumerate(data_chunk):
            if idx % 10 == 0:
                print(f"Process {process_id}, Chunk {chunk_id}: Processing item {idx}/{len(data_chunk)}")
            
            question = item['conversations'][0]['value']
            response = item['conversations'][1]['value']
            
            content = {
                "text": f"[Instruction]\n{rule['prompt']}\n\n"
                        f"[Super-Categories]\n{', '.join(super_categories)}\n\n"
                        f"[Question]\n{question}\n\n"
                        f"[Response]\n{response}\n"
            }
            
            review = get_eval(content, 512, process_id)
            categories = parse_score(review)
            if categories == -1:
                super_category = 'Other'
                sub_category = 'Other'
            else:
                # super_category, sub_category = categories.split(' | ')
                categories = categories.split(' | ')
                if len(categories) == 2:
                    super_category, sub_category = categories
                else:
                    super_category = categories[0]
                    sub_category = "Other"
                if super_category not in super_categories:
                    for default_sc in super_categories:
                        if super_category in default_sc:
                            super_category = default_sc
                            break
            item['super_category'] = super_category
            item['sub_category'] = sub_category
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
    output_path = os.path.join(train_folder, f'dataset-{split_name}_category.json')
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(combined_datalist, json_file, indent=4, ensure_ascii=False)
    
    # Combine the review files
    combined_review_path = 'fed_aya_topic_explanation.jsonl'
    
    # Check if the file exists - if not, create it; if yes, append to it
    mode = 'a' if os.path.exists(combined_review_path) else 'w'
    
    with open(combined_review_path, mode) as combined_file:
        for chunk_id in range(len(results)):
            chunk_review_path = f'fed_aya_topic_explanation_process_{chunk_id}.jsonl'
            if os.path.exists(chunk_review_path):
                with open(chunk_review_path, 'r') as chunk_file:
                    combined_file.write(chunk_file.read())
    
    # Remove individual chunk files
    for chunk_id in range(len(results)):
        chunk_review_path = f'fed_aya_topic_explanation_process_{chunk_id}.jsonl'
        if os.path.exists(chunk_review_path):
            os.remove(chunk_review_path)
    
    print(f"Combined all results for {split_name} into final dataset file")
    return len(combined_datalist)

def categorize_question_type_parallel(split_name):
    # Load the original data
    input_path = os.path.join(train_folder, f'dataset-{split_name}.json')
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
    train_count = categorize_question_type_parallel(2)
    print(f"Processed {train_count} items")
    
    print("Question type categorization complete!")