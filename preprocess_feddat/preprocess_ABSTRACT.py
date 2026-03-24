import shutil
import json 
import os
import numpy as np
import random
from collections import defaultdict

data_type = 'test'
question_file_path = f"MultipleChoice_abstract_v002_{data_type}2015_questions.json"
answer_file_path = f"abstract_v002_{data_type}2015_annotations.json"
data_path = "dataset/Abstract"
max_num = 10000 if data_type == 'train' else 1000

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
os.makedirs(os.path.join(data_path, "images"), exist_ok=True)
os.makedirs(os.path.join(data_path, data_type), exist_ok=True)
for question_data, answer_data in zip(question_datas['questions'], answer_datas['annotations']):
    new_data = {}
    if str(question_data['question_id']) != str(answer_data['question_id']):
        raise ValueError("question id different!")
    new_data['question'] = question_data['question']
    new_data['answer'] = answer_data['multiple_choice_answer']
    question_data['multiple_choices'].remove(new_data['answer'])
    new_data['image_id'] = question_data['image_id']
    new_data['image'] = os.path.join(data_path, "images", f"abstract_v002_{data_type}2015_{str(question_data['image_id']).zfill(12)}.png")
    if answer_data['answer_type'] == 'number':
        candidates = [x for x in question_data['multiple_choices'] if x.isdigit()]  
    elif answer_data['answer_type'] == 'yes/no':
        candidates = ['yes'] if new_data['answer']=='no' else ['no']
    else:
        candidates = [x for x in question_data['multiple_choices'] if not x.isdigit()]
        if 'yes' in candidates:
            candidates.remove('yes')
        if 'no' in candidates:
            candidates.remove('no')

    sampled_candidates = [new_data['answer']] + random.sample(candidates, min(3, len(candidates)))
    random.shuffle(sampled_candidates)
    new_data['candidates'] = sampled_candidates
    datalists[answer_data['answer_type']].append(new_data)

print("Tasks")
print(datalists.keys())

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
                "value": f"Please respond accurately to the following query. You must choose your answer from the Choice List.\n<image>\nQuestion: {data['question']}\nChoice list:[{data['candidates']}].\n Your answer is:"
            },
            {
                "from": "gpt",
                "value": data['answer']
            }
        ]
        json_data.append(new_data)
    print(f"{task_type} sampled: {len(json_data)}, original: {len(datalists[task_type])}")
    random.shuffle(json_data)
    with open(os.path.join(data_path, data_type, f"dataset-{task_num//4}{task_num%4}.json"), 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

