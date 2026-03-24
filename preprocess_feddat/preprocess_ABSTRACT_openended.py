import shutil
import json 
import os
import numpy as np
import random
from collections import defaultdict

data_type = 'test'
question_file_path = f"OpenEnded_abstract_v002_{data_type}2015_questions.json"
answer_file_path = f"abstract_v002_{data_type}2015_annotations.json"
data_path = "dataset/Abstract"
max_num = 10000 if data_type == 'train' else 1000

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
                "value": f"Please respond accurately to the following query. <image>Question: {data['question']} Your answer is:"
            },
            {
                "from": "gpt",
                "value": data['answer']
            }
        ]
        json_data.append(new_data)
    return json_data

'''
dict_keys(['other', 'yes/no', 'number'])

Question
{'image_id': 11779, 'question': 'Who looks happier?', 'multiple_choices': ['alive', '1', 'woman', 'purple', '2', 'yes', 'white', 'boy', 'she loves him', 'mountain', '3', 'no', 'baby', 'man', 'yellow', 'red', '4', 'blue'], 'question_id': 117792}

answer
{'question_type': 'who', 'multiple_choice_answer': 'man', 'answers': [{'answer': 'old person', 'answer_confidence': 'maybe', 'answer_id': 1}, {'answer': 'man', 'answer_confidence': 'maybe', 'answer_id': 2}, {'answer': 'man', 'answer_confidence': 'yes', 'answer_id': 3}, {'answer': 'man', 'answer_confidence': 'yes', 'answer_id': 4}, {'answer': 'old man', 'answer_confidence': 'yes', 'answer_id': 5}, {'answer': 'man', 'answer_confidence': 'yes', 'answer_id': 6}, {'answer': 'man', 'answer_confidence': 'yes', 'answer_id': 7}, {'answer': 'man', 'answer_confidence': 'yes', 'answer_id': 8}, {'answer': 'man', 'answer_confidence': 'yes', 'answer_id': 9}, {'answer': 'grandpa', 'answer_confidence': 'yes', 'answer_id': 10}], 'image_id': 11779, 'answer_type': 'other', 'question_id': 117792}
'''

def remove_multiple_elements(lst, elements_to_remove):
    return [x for x in lst if x not in elements_to_remove]

with open(f"{data_path}/{question_file_path}", "r", encoding="utf-8") as file:
    question_datas = json.load(file)

with open(f"{data_path}/{answer_file_path}", "r", encoding="utf-8") as file:
    answer_datas = json.load(file)

print(len(question_datas['questions']))
print(len(answer_datas['annotations']))

datalists = defaultdict(list)
#question_type_count = defaultdict(defaultdict(int))
question_type_count = defaultdict(lambda: defaultdict(int))
os.makedirs(os.path.join(data_path, "images"), exist_ok=True)
os.makedirs(os.path.join(data_path, data_type), exist_ok=True)
for question_data, answer_data in zip(question_datas['questions'], answer_datas['annotations']):
    new_data = {}
    if str(question_data['question_id']) != str(answer_data['question_id']):
        raise ValueError("question id different!")
    new_data['question'] = question_data['question']
    new_data['answer'] = answer_data['multiple_choice_answer']
    new_data['image_id'] = question_data['image_id']
    new_data['image'] = os.path.join(data_path, "images", f"abstract_v002_{data_type}2015_{str(question_data['image_id']).zfill(12)}.png")
    datalists[answer_data['answer_type']].append(new_data)
    question_type_count[answer_data['answer_type']][answer_data['question_type']] += 1
    #question_type_count[f"{answer_data['answer_type']} / {answer_data['question_type']}"] += 1


filtered_sorted_items = sorted(
    [(k, v) for k, v in question_type_count['other'].items() if 'color' in k],
    key=lambda x: x[1],
    reverse=True
)

os.makedirs(os.path.join(data_path, data_type), exist_ok=True)

task_index = 0
for task_num, task_type in enumerate(list(datalists.keys())):
    if task_type == 'other':
        data_color = [data for data in datalists[task_type] if 'color' in data['question']]
        data_remaining = [data for data in datalists[task_type] if 'color' not in data['question']]
        sampled_data_color = np.random.choice(data_color, size=min(max_num, len(data_color)), replace=False).tolist()
        sampled_data_remaining = np.random.choice(data_remaining, size=min(max_num, len(data_remaining)), replace=False).tolist()
        sampled_json_data_color = make_json_files(sampled_data_color)
        sampled_json_data_remaining = make_json_files(sampled_data_remaining)
        
        with open(os.path.join(data_path, data_type, f"dataset-{task_index}.json"), 'w') as json_file:
            json.dump(sampled_json_data_color, json_file, indent=4)
        
        task_index += 1

        with open(os.path.join(data_path, data_type, f"dataset-{task_index}.json"), 'w') as json_file:
            json.dump(sampled_json_data_color, json_file, indent=4)

        task_index += 1

        print(f"{task_type} color sampled: {len(sampled_json_data_color)}, original: {len(data_color)}")
        print(f"{task_type} remaining sampled: {len(sampled_json_data_remaining)}, original: {len(data_remaining)}")
    else:
        json_data = []
        sampled_data = np.random.choice(datalists[task_type], size=min(max_num, len(datalists[task_type])), replace=False).tolist()
        json_data = make_json_files(sampled_data)

        print(f"{task_type} sampled: {len(json_data)}, original: {len(datalists[task_type])}")
        with open(os.path.join(data_path, data_type, f"dataset-{task_index}.json"), 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        task_index += 1
