import numpy as np
import random
import json
import os
import multiprocessing
from googletrans import Translator

# ----------------------------------------------------------------------
# Global data load and setup (only run once in main process)
# ----------------------------------------------------------------------
def load_and_prune_data(path):
    """Load the all_clients.json, build `languages`, but limit items to 10 per language."""
    with open(path, 'r', encoding='utf-8') as f:
        datalist = json.load(f)

    languages = {}
    for k, v in datalist.items():
        for item in v:
            lang = item['language']
            if lang not in languages:
                languages[lang] = []
            languages[lang].append(item)
    return languages


def ensure_dir(directory):
    """Create directory if not exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


# ----------------------------------------------------------------------
# The function that will run in each process
# ----------------------------------------------------------------------
def process_data_chunk(args):
    """
    Process a chunk of data for a specific client
    """
    (chunk_id, data_chunk, client_key, 
     language_dict, 
     length_threshold1, 
     length_threshold2, 
     output_folder) = args

    # Set the same seeds in each process for reproducibility
    random.seed(42 + chunk_id)  # Unique seed per chunk
    np.random.seed(42 + chunk_id)

    # Create a local Translator for this process
    translator = Translator()

    processed_data = []
    max_instruction = 0
    max_response = 0

    # -----------------------------------------------------------
    # Helper: handle one item
    # -----------------------------------------------------------
    def handle_item(idx, item, key):
        nonlocal max_instruction, max_response

        # Skip if text is too long
        if len(item['instruction']) > length_threshold1 or len(item['response']) > length_threshold2:
            return None

        # Language detection on the instruction
        result1 = translator.detect(item['instruction'])
        if result1.lang not in language_dict[key]:
            # print('mismatch (instruction):', key, result1.lang)
            return None

        # Language detection on the response
        result2 = translator.detect(item['response'])
        if result2.lang not in language_dict[key]:
            # print('mismatch (response):', key, result2.lang)
            return None

        # Update max lengths
        if len(item['instruction']) > max_instruction:
            max_instruction = len(item['instruction'])
        if len(item['response']) > max_response:
            max_response = len(item['response'])

        # Build JSON data
        json_data = {
            'id': f"{chunk_id}_{idx}",  # Unique ID with chunk prefix
            'language': language_dict[key],
            'conversations': [
                {'from': 'human', 'value': item['instruction']},
                {'from': 'gpt',   'value': item['response']}
            ]
        }

        return json_data

    # -----------------------------------------------------------
    # Process the chunk based on client_key type
    # -----------------------------------------------------------
    if isinstance(client_key, list):
        # A list of languages
        for key in client_key:
            for idx, item in enumerate(data_chunk.get(key, [])):
                json_data = handle_item(idx, item, key)
                if json_data:
                    processed_data.append(json_data)
    else:
        # Single language string
        key = client_key
        for idx, item in enumerate(data_chunk.get(key, [])):
            json_data = handle_item(idx, item, key)
            if json_data:
                processed_data.append(json_data)

    # -----------------------------------------------------------
    # Write intermediate results
    # -----------------------------------------------------------
    print(f'Chunk {chunk_id} processed {len(processed_data)} items')
    print(f'Max length: instruction={max_instruction}, response={max_response}')

    # Write chunk results to temporary file
    chunk_path = os.path.join(output_folder, f'chunk_{chunk_id}.json')
    with open(chunk_path, 'w', encoding='utf-8') as f_out:
        json.dump(processed_data, f_out, indent=4, ensure_ascii=False)
    
    return len(processed_data)


# ----------------------------------------------------------------------
# Function to split data into chunks
# ----------------------------------------------------------------------
def split_data_for_client(client_key, languages, num_chunks):
    """Split data for a specific client into chunks for parallel processing"""
    chunked_data = []
    
    if isinstance(client_key, list):
        # A list of languages
        all_items = {}
        for key in client_key:
            if key in languages:
                all_items[key] = languages[key]
                random.shuffle(all_items[key])
        
        # Split items by language
        for key in all_items:
            chunk_size = max(1, len(all_items[key]) // num_chunks)
            chunks = [all_items[key][i:i + chunk_size] for i in range(0, len(all_items[key]), chunk_size)]
            
            # Distribute chunks across our return list
            for i, chunk in enumerate(chunks):
                if i >= len(chunked_data):
                    chunked_data.append({})
                if key not in chunked_data[i]:
                    chunked_data[i][key] = []
                chunked_data[i][key].extend(chunk)
    else:
        # Single language
        if client_key in languages:
            items = languages[client_key]
            random.shuffle(items)
            chunk_size = max(1, len(items) // num_chunks)
            chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
            
            for chunk in chunks:
                chunked_data.append({client_key: chunk})
    
    # Ensure we have exactly num_chunks (even if some are empty)
    while len(chunked_data) < num_chunks:
        chunked_data.append({})
    
    return chunked_data[:num_chunks]


# ----------------------------------------------------------------------
# Main: run one specific task in parallel
# ----------------------------------------------------------------------
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    
    # Parameters for selecting which task to run
    task_i = 10  # Specify which i value to use
    task_j = 1  # Specify which j value to use
    num_processes = 72  # Number of processes to split the task into

    # 1. Prepare data
    data_path = './dataset/Fed-aya/all_clients.json'
    languages = load_and_prune_data(data_path)

    print(f"Running task i={task_i}, j={task_j} with {num_processes} processes")
    print("Languages in dataset:", list(languages.keys()))

    # Create output folders
    output_folder = 'dataset/Fed-aya2'
    train_folder = os.path.join(output_folder, 'train')
    temp_folder = os.path.join(output_folder, 'temp')
    ensure_dir(train_folder)
    ensure_dir(temp_folder)

    client_configs = [
        ['Standard Arabic', 'Hausa', 'Somali'],
        ['Standard Malay', 'Plateau Malagasy', ['Filipino', 'Cebuano', 'Indonesian', 'Javanese']],
        ['Malayalam', 'Telugu'],
        [['Russian','Ukrainian']],
        ['Dutch'],
        [['Hindi','Urdu'], 'Marathi'],
        ['Panjabi', 'Nepali (individual language)'],
        ['French', 'Spanish', 'Portuguese'],
        ['Yoruba'],
        ['Simplified Chinese', 'Vietnamese'],
        ['Kyrgyz', 'Turkish']
    ]

    language_dict = {
        "Standard Arabic": ["ar"],
        "Hausa": ["ha"],
        "Somali": ["so"],
        "Standard Malay": ["ms"],
        "Plateau Malagasy": ["mg"],
        "Filipino": ["tl"],
        "Cebuano": ["ceb", "tl"],
        "Indonesian": ["id"],
        "Javanese": ["jw", "id"],
        "Malayalam": ["ml"],
        "Telugu": ["te"],
        "Tamil": ["ta"],
        "Polish": ["pl"],
        "Russian": ["ru","uk"],
        "Ukrainian": ["uk","ru"],
        "Lithuanian": ["lt"],
        "English": ["en"],
        "Dutch": ["nl"],
        "Swedish": ["sv"],
        "Hindi": ["hi", "ur"],
        "Urdu": ["ur", "hi"],
        "Bengali": ["bn"],
        "Marathi": ["mr"],
        "Panjabi": ["pa", "en"],
        "Nepali (individual language)": ["ne", "hi"],
        "Gujarati": ["gu"],
        "French": ["fr"],
        "Spanish": ["es"],
        "Portuguese": ["pt", "pt-PT"],
        "Yoruba": ["yo"],
        "Zulu": ["zu"],
        "Igbo": ["ig"],
        "Simplified Chinese": ["zh-CN"],
        "Japanese": ["ja"],
        "Vietnamese": ["vi"],
        "Kyrgyz":["kir", "ky"],
        "Turkish":["tr", "tur"],
    }

    length_threshold1 = 4000
    length_threshold2 = 4000

    # 2. Get the specific client key for the task
    try:
        client_key = client_configs[task_i][task_j]
    except IndexError:
        print(f"Error: task_i={task_i}, task_j={task_j} is out of range")
        exit(1)

    print(f"Processing client: {client_key}")

    # 3. Split the data for the specific client into chunks
    data_chunks = split_data_for_client(client_key, languages, num_processes)

    # 4. Process each chunk in parallel
    tasks = []
    for chunk_id, data_chunk in enumerate(data_chunks):
        task_args = (
            chunk_id,
            data_chunk,
            client_key,
            language_dict,
            length_threshold1,
            length_threshold2,
            temp_folder
        )
        tasks.append(task_args)

    # Run tasks in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_data_chunk, tasks)

    total_processed = sum(results)
    print(f"Total processed items: {total_processed}")

    # 5. Combine results from all chunks
    all_data = []
    for chunk_id in range(len(data_chunks)):
        chunk_path = os.path.join(temp_folder, f'chunk_{chunk_id}.json')
        if os.path.exists(chunk_path):
            with open(chunk_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                all_data.extend(chunk_data)
            # Optional: remove temporary file
            # os.remove(chunk_path)

    # 6. Write final combined output
    if task_i == 0:
        output_path = os.path.join(train_folder, f'dataset-{task_j}.json')
    else:
        output_path = os.path.join(train_folder, f'dataset-{task_i}{task_j}.json')
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(all_data, f_out, indent=4, ensure_ascii=False)

    print(f"Combined data written to {output_path}")