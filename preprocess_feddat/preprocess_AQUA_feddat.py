import json
import os
import numpy as np
import random
import openai
from openai import OpenAI
import time
import jsonlines

NUM_SECONDS_TO_SLEEP = 0.1

client = OpenAI(
    # This is the default and can be omitted
    # api_key=os.environ.get("OPENAI_API_KEY"),
    api_key="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"
)
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
                    'content': 'You are a helpful and precise assistant to generate choice list.'
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


np.random.seed(42)
random.seed(42)

def categorize_question(question_text):
    """
    Very rough rule-based classifier for demonstration.
    You can refine these rules or add new ones.
    """
    q_lower = question_text.lower().strip()
    
    # 1) Determine WH-type
    if q_lower.startswith("who"):
        wh_type = "who"
    elif q_lower.startswith("what"):
        wh_type = "what"
    elif q_lower.startswith("where"):
        wh_type = "where"
    elif q_lower.startswith("which"):
        wh_type = "which"
    elif q_lower.startswith("how"):
        wh_type = "how"
    else:
        wh_type = "other"
    
    # 2) Subject - naive approach looking for keywords
    if "group of people" in q_lower or "people" in q_lower:
        subject = "group_of_people"
    elif "women" in q_lower or "woman" in q_lower:
        subject = "woman"
    elif "men" in q_lower or "man" in q_lower:
        subject = "man"
    elif "dog" in q_lower:
        subject = "dog"
    elif "cat" in q_lower:
        subject = "cat"
    elif "animal" in q_lower:
        subject = "animal"
    else:
        subject = "other"
    
    # 3) Action/relationship - more naive keyword matching
    if "stand" in q_lower:
        action = "stand"
    elif "sit" in q_lower:
        action = "sit"
    elif "wear" in q_lower or "wearing" in q_lower:
        action = "wear"
    elif "hold" in q_lower or "holding" in q_lower:
        action = "hold"
    elif "walk" in q_lower or "walking" in q_lower:
        action = "walk"
    elif "talk" in q_lower or "talking" in q_lower:
        action = "talk"
    elif "look" in q_lower or "looking" in q_lower:
        action = "look"
    elif "sell" in q_lower or "selling" in q_lower:
        action = "sell"
    elif "on the" in q_lower:
        action = "on the"
    else:
        action = "other"
    
    return wh_type, subject, action

def annotate_dataset(input_json_path, output_json_path=None):
    # 1. Load the original data
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    new_data = []
    # 2. Annotate each item with the chosen categories
    for item in data:
        question_text = item["question"]
        
        # We also have 'need_external_knowledge' from the original
        needs_knowledge = item.get("need_external_knowledge", False)
        if needs_knowledge:
            continue
        # Our categorization
        wh_type, subject, action = categorize_question(question_text)
        
        # Attach annotations to the item
        item["wh_type"] = wh_type
        item["subject"] = subject
        item["action"] = action
        
        new_data.append(item)
    
    # 3. Optionally, write annotated data to a new JSON
    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as fout:
            json.dump(new_data, fout, indent=2, ensure_ascii=False)
    else:
        # If no output path specified, just print or return them
        print(json.dumps(new_data, indent=2, ensure_ascii=False))

output_folder = "dataset/AQUA"
train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')
image_folder = os.path.join(output_folder, 'images')

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

input_file = f"{output_folder}/val.json"          # Path to your original data
output_file = f"{output_folder}/val_annotated.json"  # Where to save the annotated data
annotate_dataset(input_file, output_file)

input_file = f"{output_folder}/train.json"          # Path to your original data
output_file = f"{output_folder}/train_annotated.json"  # Where to save the annotated data
annotate_dataset(input_file, output_file)

input_file = f"{output_folder}/test.json"          # Path to your original data
output_file = f"{output_folder}/test_annotated.json"  # Where to save the annotated data
annotate_dataset(input_file, output_file)

answer_lists = ['person','man', 'woman', 'couple', 'girl', 'boy','baby','child',
                'people','men', 'women','children','girls','boys',
                 
                'animal','human','horse','bat','elephant','tiger','donkey','dog','cow','eagle','goat','duck','deer','fish','bear','lion','cat','cattle','sheep','bull','bird','peacock','chicken','antelope','parrot',
                'animals','horses','dogs', 'cows','elephants','birds',
                
                'fruit','cherry','raspberry', 'orange','apple','pineapple', 'apples','vegetable','grapes','produce','peach',
                'plant', 'flower','cabbage','lily','maple', 'pine', 'mushroom','vegetables', 'palm tree','tree','trees', 'oak','rose','flowers',
                
                'forest','mountain','waterfall','cliff','river', 'pool','pond','beach','hillside','lake','ground', 'garden','ocean','floor','field', 'grass','bush','road','hill',
                'building','church','castle','tower','bridge','street', 'city','courtyard', 'clock tower','museum', 'fountain','hydrant','village', 'circus','wall','statue','statues',
                
                'downtown','dock','farm','market','barn','court','house','park','restaurant','pier','cafe','harbor','stadium','tent',

                'blonde','ivory','dark','gold',
                'furniture','bed','closet','chair','bench', 'dining room','room', 'shelf','rug','table','door','staircase', 'window','fireplace','living room', 'piano','couch','cabinet','curtain','drawer','bedroom',
                
                'clothing','coat','jacket','suit', 'shirt','hat', 'sun hat','tie''necklace','helmet','sweater','bikini','shorts', 'dress','shoe','scarf','sunglasses','pants',
                    
                'fog','sun', 'sky','dirt', 'rain','water','snow', 'rock','rocks','leaf','smoke','air','fire', 'wood', 'woods','tropical','cumulus',
                
                'reading', 'railing','sitting','painting','photography','standing',
                
                'food','dessert', 'donuts','cake','seafood','bread','beer','pork',
                
                'transportation', 'train','boat','carriage','ship','boats','motorcycle','sailboat','canoe','bicycle',
                
                'face', 'porcelain', 'hair',   'vase', 
                'pottery', 
                'poster',    
                
                'glass',  'arch', 'dome','picture', 'mirror','tile','cross', 'clock',   'beard', 
                
                'military', 'costume', 'guitar', 'brick', 'book', 'globe',
                'arms', 'paper', 'land', 'stage', 'hand', 
                'robe', 'wine',    'umbrella', 'bowl', 
                  'advertisement', 'basket', 'hands', 'vases',  'cane',
                    'sign',  'wheel',
                 'sidewalk',    'toy', 'blanket', 'lamp', 'fence', 
                'background', 'structure', 'cup', 'glasses',  'wine glass', 
                'camera',  'roof',   'pedestrian', 'pot', 'pillow',  'books',  
                'phone', 'plate', 'doorway', 'box', 'scissors',  'arm',  'ice', 'tattoo', 
                'hood', 'wagon', 'screen',   'knife', 'bag', 
                   'graffiti',  'headband',  'electronics', 'lace',  
                   'branch',   'stick', 'frisbee',
                   'towel',   'cap', 'drink', 'fur','theater', 'crown', 'money', 'logo', 'doll', 'purse', 'store',  'sink', 'sill',  'newspaper', 'umbrellas',  'lap',    'board', 'fork', 'game', 'paddle', 'gun','wedding','play', 'balloon', 'dinner', 'chandelier', 'plaid',  'flag', 
                'arrow']

rule = {"role": "Assistant", 
        "prompt": 
            "Please generate a choice list for the given question asking about an image and the given correct answer.\n"
            "The choice list should:\n"
            "1. include the correct answer.\n"
            "2. include 3 distractors that are somewhat relevant to the given question.\n"
            "3. be chosen from the given Possible Candidates, but should generate distractor if no other relevant candidate exists in the given list.\n"
            "Please make sure NOT to include the choices with the same meaning, i.e., man and person, child and boy, woman and human, women and girls"
            "You must return the choices by strictly following this format:\"[[choice A | choice B | choice C | ...]]\", for example: \"Choices: [[man | woman | dog | people]]\"."
        }

def process_and_save(dataset, output_folder, subset_name, size):
    # Define image subfolder within output folder

    # Initialize list to hold all JSON data
    # dataset-0 : action - stand
    # dataset-1 : action - sit/on the
    # dataset-2 : action - wear/hold/walk/talk/look/sell
    # dataset-3 : action - other
    review_file = open(f'aqua_choices_gpt_train.jsonl', 'a')
    json_data_list1 = []
    json_data_list2 = []
    json_data_list3 = []
    json_data_list4 = []

    # Process and save images and labels
    for item in dataset:
        # Define image path
        image_path = os.path.join(image_folder, f"{item['image']}")
        if item['answer'] in ['crowd','other',]:
            item['answer'] = 'people'
        elif item['answer'] == "kids":
            item['answer'] = "children"
        elif item['answer'] == 'luggage':
            item['answer'] = 'bag'
        elif item['answer'] in ['are', 'it', 'object', 'area', 'something', "place", "items"]:
            continue
        elif item['answer'] in ['pitbull', 'beagle', 'collie']:
            answer_list = ['pitbull', 'beagle', 'collie', 'cat']
        elif item['answer'] == 'female':
            answer_list = ['male', 'female', 'animal']
        elif item['answer'] in ['navy','army']:
            answer_list = ['navy','army','air force','marine corps']
        elif item['answer'] in ['fisheye', 'landscape']:
            if 'what type of photo':
                answer_list = ['fisheye', 'landscape']
            else:
                continue
        elif item['answer'] in ['sunset',  'night']:
            answer_list = ['morning', 'noon', 'sunset', 'night']
        
        elif item['answer'] in ['spring', 'summer', 'fall', 'autumn', 'winter']:
            answer_list = ['spring', 'summer', 'fall', 'winter']
        else:
            content = {
                    "text": f"[Instruction]\n{rule['prompt']}\n\n"
                            f"[Question]\n{item['question']}\n\n"
                            f"[Answer]\n{item['answer']}\n\n"
                            f"[Possible Candidates]\n[{', '.join(answer_lists)}]\n"
                            
                }
            
            review = get_eval(content, 512)
            review_file.write(json.dumps({"question": item['question'], "gpt_response":review}) + "\n")
            answer_list = parse_score(review).split(' | ')
        random.shuffle(answer_list)
        # Structure for LLaVA JSON
        json_data = {
            "id": item['image'],
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n" + item['question'] + f"?\nAnswer the question using the choices from the choice list.\nChoice list:{answer_list}. Your answer is: " 
                },
                {
                    "from": "gpt",
                    "value": item['answer']
                }
            ]
        }
        
        
        if item['action'] == 'stand':
            json_data_list1.append(json_data)
        elif item['action'] in ['sit', 'on the']:
            json_data_list2.append(json_data)
        elif item['action'] in ['wear', 'hold', 'walk', 'talk', 'look', 'sell']:
            json_data_list3.append(json_data)
        else:
            json_data_list4.append(json_data)

    print(len(json_data_list1))
    print(len(json_data_list2))
    print(len(json_data_list3))
    print(len(json_data_list4))
    # 2777 7259
    # 29568 40244
    # Save the JSON data list to a file
    json_output_path = os.path.join(output_folder, subset_name, 'dataset-100.json')
    json_data_list1 = np.random.choice(json_data_list1, replace=False, size=min(size, len(json_data_list1))).tolist()
    print(len(json_data_list1))
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list1, json_file, indent=4)
    
    json_output_path = os.path.join(output_folder, subset_name, 'dataset-101.json')
    json_data_list2 = np.random.choice(json_data_list2, replace=False, size=min(size, len(json_data_list2))).tolist()
    print(len(json_data_list2))
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list2, json_file, indent=4)
        
    json_output_path = os.path.join(output_folder, subset_name, 'dataset-102.json')
    json_data_list3 = np.random.choice(json_data_list3, replace=False, size=min(size, len(json_data_list3))).tolist()
    print(len(json_data_list3))
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list3, json_file, indent=4)
    
    json_output_path = os.path.join(output_folder, subset_name, 'dataset-103.json')
    json_data_list4 = np.random.choice(json_data_list4, replace=False, size=min(size, len(json_data_list4))).tolist()
    print(len(json_data_list4))
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list4, json_file, indent=4)

with open(f"{output_folder}/train_annotated.json") as fp:
    train_dataset = json.load(fp)

with open(f"{output_folder}/test_annotated.json") as fp:
    test_dataset = json.load(fp)
with open(f"{output_folder}/val_annotated.json") as fp:
    val_dataset = json.load(fp)
test_dataset.extend(val_dataset)
# Process and save the datasets ,  ('test', test_dataset, 2000) ('train', train_dataset, 10000)
for subset, data, size in [('train', train_dataset, 10000) , ]:
    if data:
        process_and_save(data, output_folder, subset, size)


##############
import json
for i in range(4):
    datalist = json.load(open(f'dataset/AQUA/test/dataset-10{i}.json','r'))
    for item in datalist:
        question = item['conversations'][0]['value']
        question = question.replace('[[', '[')
        question = question.replace(']]', ']')
        item['conversations'][0]['value'] = question
    
    with open(f'dataset/AQUA/test/dataset-10{i}_fixed.json','w') as fp:
        json.dump(datalist, fp, indent=4)