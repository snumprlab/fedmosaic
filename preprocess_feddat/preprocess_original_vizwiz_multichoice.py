import json
import os
import random
import openai
from openai import OpenAI
import time
import jsonlines

NUM_SECONDS_TO_SLEEP = 0.01

client = OpenAI(
    # This is the default and can be omitted
    # api_key=os.environ.get("OPENAI_API_KEY"),
    api_key=""
)

random.seed(42)

output_folder = 'dataset/VizWiz-Original'
train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

def get_eval(content: str, max_tokens: int):
    user_content = [
                        {"type":"text", "text":content['text']},
                    ]

    while True:
        try:
            '''
            
            '''
            response = client.chat.completions.create(
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant to generate a choice list.'
                },{
                    'role': 'user',
                    'content': user_content
                }],
                model="gpt-4o-mini-2024-07-18", # TODO: Choose gpt version
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
        time.sleep(NUM_SECONDS_TO_SLEEP)
    return response.choices[0].message.content

def parse_score(review):
    try:
        score = review.split('[[')
        assert len(score) == 2
        score = score[-1].split(']]')[0]
        # return int(score)
        return score

    except Exception as e:
        print(e)
        print('error', review)
        return -1

###########################################################################################

rule = {"role": "Assistant", 
        "prompt": 
            "Please generate two or three answer choices for the given question asking about an image and the given correct answer.\n"
            "The possible answers are the variation of correct and wrong answers. You should use it to infer about the image.\n"
            "You should:\n"
            "1. generate 3 choices if the correct answer is \"unanswerable\".\n"
            "2. generate 2 choices if the correct answer is not \"unanswerable\".\n"
            "3. NOT INCLUDE unanswerable or correct answer in the generated choices.\n"
            "4. generate choices relevant to the question, but not too similar to the correct answer.\n"
            "5. generate only 2 choices if the question is asking between the two things, e.g., yes or no.\n"
            "You must return the choices by strictly following this format:\"[[Choice A | Choice B]]\", for example: \"Choice list: [[red | blue | pink]]\"."
        }
    
def preprocess_vizwiz_original(original_json, is_train=True):
    original_datalist = json.load(open(original_json, 'r'))
    output_datalist1 = []
    output_datalist2 = []
    output_datalist3 = []
    output_datalist4 = []
    
    if is_train:
        review_file = open(f'vizwiz_choice_list_train.jsonl', 'a')
    else:
        review_file = open(f'vizwiz_choice_list_test.jsonl', 'a')
    
    for item in original_datalist:
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
        
        for k1 in answers.keys():
            for k2 in answers.keys():
                if k1 == k2:
                    continue
                if k2 in k1:
                    answers[k2][0] += 1
        
        question_type = item['question_type']
        if item['answer_type'] == 0:
            answer = 'unanswerable'
        else:
            max_cnt = 0
            answer = ""
            for candidate, cnt in answers.items():
                if candidate == 'unanswerable': continue
                if (cnt[1] == 'yes' or cnt[1] == 'maybe') and cnt[0] > max_cnt:
                    answer = candidate
                    max_cnt = cnt[0]
            
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
        
        review_file.write(json.dumps({"question": question, "gpt_response":review, "answer":answer}) + "\n")

        answer_list.append(answer)
        if 'unanswerable' not in answer_list:
            answer_list.append('unanswerable')
        random.shuffle(answer_list)
            
        instruction = "\nWhen the provided information is insufficient, respond with 'unanswerable'.\nAnswer the question using the choices from the choice list."
        json_data = {
            "id": image_file.split('.jpg')[0],
            "image": [os.path.join(output_folder, 'images','train',image_file)] if is_train else [os.path.join(output_folder, 'images','val',image_file)],
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
    
    if is_train:
        target_folder = train_folder
    else:
        target_folder = test_folder
    
    with open(f'{target_folder}/dataset-0.json', 'w') as json_file:
        json.dump(output_datalist1, json_file, indent=4)
    with open(f'{target_folder}/dataset-1.json', 'w') as json_file:
        json.dump(output_datalist2, json_file, indent=4)
    with open(f'{target_folder}/dataset-2.json', 'w') as json_file:
        json.dump(output_datalist3, json_file, indent=4)
    with open(f'{target_folder}/dataset-3.json', 'w') as json_file:
        json.dump(output_datalist4, json_file, indent=4)
    

# ('train_split.json', True),('val_split.json', False)
for json_path in [('train_split.json', True)]:
    json_path, is_train = json_path
    preprocess_vizwiz_original(os.path.join(output_folder, json_path), is_train)