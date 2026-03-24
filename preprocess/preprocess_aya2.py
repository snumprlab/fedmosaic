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
            # Limit to 10 items per language
            # if len(languages[lang]) < 10:
            languages[lang].append(item)
    return languages


def ensure_dir(directory):
    """Create directory if not exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


# ----------------------------------------------------------------------
# The function that will run in each process
# ----------------------------------------------------------------------
def process_one_client(args):
    """
    This function processes one client configuration (i, j, client_key),
    including all data filtering, train/test splits, language detection,
    and writing to JSON files.
    """
    (i, j, client_key,
     languages,
     language_dict,
     length_threshold1,
     length_threshold2,
     train_folder,
     test_folder) = args

    # -----------------------------------------------------------
    # Set the same seeds in each process so that random calls
    # produce reproducible results per process
    # -----------------------------------------------------------
    random.seed(42)
    np.random.seed(42)

    # Create a local Translator for this process
    translator = Translator()

    train_datalist = []
    test_datalist = []
    max_instruction = 0
    max_response = 0
    max_test_num = 100
    if i == 0 or i == 1 or i == 9:
        max_train_num = 3200
    elif i == 2 or i == 5 or i == 8:
        max_train_num = 1600
    elif i == 3 or i == 4:
        max_train_num = 1200
    elif i == 6:
        max_train_num = 4000
    elif i == 7:
        max_train_num = 1200
    else:
        return

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
            print('mismatch (instruction):', key, result1.lang)
            return None

        # Language detection on the response
        result2 = translator.detect(item['response'])
        if result2.lang not in language_dict[key]:
            print('mismatch (response):', key, result2.lang)
            return None

        # Update max lengths
        if len(item['instruction']) > max_instruction:
            max_instruction = len(item['instruction'])
        if len(item['response']) > max_response:
            max_response = len(item['response'])

        # Build JSON data
        json_data = {
            'id': idx,
            'conversations': [
                {'from': 'human', 'value': item['instruction']},
                {'from': 'gpt',   'value': item['response']}
            ]
        }

        # Return which split it goes to
        return json_data

    # -----------------------------------------------------------
    # Main logic for single or multiple languages in client_key
    # -----------------------------------------------------------
    if isinstance(client_key, list):
        # A list of languages
        test_per_lang = max_test_num // len(client_key)
        for key in client_key:
            if key not in languages:
                continue
            items = languages[key]

            # Shuffle
            random.shuffle(items)
            test_cnt = 0
            for idx, item in enumerate(items):
                json_data = handle_item(idx, item, key)
                if not json_data:
                    continue
                if test_cnt < test_per_lang:
                    test_datalist.append(json_data)
                    test_cnt += 1
                else:
                    train_datalist.append(json_data)
                
                if len(train_datalist) >= max_train_num:
                    train_datalist = train_datalist[:max_train_num]
                    test_datalist = test_datalist[:max_test_num]
                    break
    else:
        # Single language string
        key = client_key
        if key not in languages:
            # No data at all for this key
            pass
        else:
            items = languages[key]
            random.shuffle(items)

            for idx, item in enumerate(items):
                json_data = handle_item(idx, item, key)
                if not json_data:
                    continue
                if len(test_datalist) < max_test_num:
                    test_datalist.append(json_data)
                else:
                    train_datalist.append(json_data)

                if len(train_datalist) >= max_train_num:
                    train_datalist = train_datalist[:max_train_num]
                    test_datalist = test_datalist[:max_test_num]
                    break

    # -----------------------------------------------------------
    # Printing Info (optional)
    # -----------------------------------------------------------
    print(f'Client {i} - task {j}')
    print('train size:', len(train_datalist), 'test size:', len(test_datalist))
    print(f'max length: {max_instruction} {max_response}')

    # -----------------------------------------------------------
    # Write JSON output
    # -----------------------------------------------------------
    if i == 0:
        train_path = os.path.join(train_folder, f'dataset-{j}.json')
        test_path  = os.path.join(test_folder,  f'dataset-{j}.json')
    else:
        train_path = os.path.join(train_folder, f'dataset-{i}{j}.json')
        test_path  = os.path.join(test_folder,  f'dataset-{i}{j}.json')

    with open(train_path, 'w', encoding='utf-8') as f_out:
        json.dump(train_datalist, f_out, indent=4, ensure_ascii=False)

    with open(test_path, 'w', encoding='utf-8') as f_out:
        json.dump(test_datalist, f_out, indent=4, ensure_ascii=False)


# ----------------------------------------------------------------------
# Main: build tasks and run with multiprocessing
# ----------------------------------------------------------------------
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    # 1. Prepare data
    data_path = './dataset/Fed-aya/all_clients.json'
    languages = load_and_prune_data(data_path)

    print("Languages in dataset:", languages.keys())

    # Create output folders
    output_folder = 'dataset/Fed-aya'
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    ensure_dir(train_folder)
    ensure_dir(test_folder)

    client_configs = [
        ['Standard Arabic', 'Hausa', 'Somali'],
        ['Standard Malay', 'Plateau Malagasy', ['Filipino', 'Cebuano', 'Indonesian', 'Javanese']],
        ['Malayalam', 'Telugu', 'Tamil'],
        ['Polish', ['Russian','Ukrainian'], 'Lithuanian'],
        ['English', 'Dutch', 'Swedish'],
        [['Hindi','Urdu'], 'Bengali', 'Marathi'],
        ['Panjabi', 'Nepali (individual language)', 'Gujarati'],
        ['French', 'Spanish', 'Portuguese'],
        ['Yoruba', 'Zulu', 'Igbo'],
        ['Simplified Chinese', 'Japanese', 'Vietnamese']
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
    }

    length_threshold1 = 6000
    length_threshold2 = 6000

    # 2. Build (i, j, client_key) tasks
    tasks = []
    for i, client_names in enumerate(client_configs):
        for j, client_key in enumerate(client_names):
            if i == 0 and os.path.exists(os.path.join(test_folder, f'dataset-{j}.json')): continue
            if os.path.exists(os.path.join(test_folder, f'dataset-{i}{j}.json')): continue
            task_args = (
                i,
                j,
                client_key,
                languages,
                language_dict,
                length_threshold1,
                length_threshold2,
                train_folder,
                test_folder
            )
            tasks.append(task_args)

    # 3. Use multiprocessing to process all tasks in parallel
    #    Adjust 'processes' to the number of CPU cores you want to use.
    with multiprocessing.Pool(processes=min(30, len(tasks))) as pool:
        pool.map(process_one_client, tasks)

    print("All tasks completed.")

