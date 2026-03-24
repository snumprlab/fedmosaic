from datasets import load_dataset
import random
import os
import requests
import json

idioms = load_dataset("lampent/IRFL", "idioms-dataset", cache_dir="/disk1/thkim/")['dataset']

metaphors = load_dataset("lampent/IRFL", "metaphors-dataset", cache_dir="/disk1/thkim/")['dataset']

similes = load_dataset("lampent/IRFL", "similes-dataset", cache_dir="/disk1/thkim/")['dataset']

# categorize phrase into train/test/both
# train: 0.8 both: 0.1 test: 0.1

random.seed(42)

idioms_phrases = sorted(list(set([data['phrase'] for data in idioms])))
idioms_definitions = {}
for data in idioms:
    idioms_definitions[data['phrase']] = data['definition']
        
metaphors_phrases = sorted(list(set([data['phrase'] for data in metaphors])))
similes_phrases = sorted(list(set([data['phrase'] for data in similes])))

random.shuffle(idioms_phrases)
random.shuffle(metaphors_phrases)
random.shuffle(similes_phrases)

idioms_phrases_train = idioms_phrases[:int(len(idioms_phrases)*0.8)]
idioms_phrases_both = idioms_phrases[int(len(idioms_phrases)*0.8):int(len(idioms_phrases)*0.9)]
idioms_phrases_test = idioms_phrases[int(len(idioms_phrases)*0.9):]

metaphors_phrases_train = metaphors_phrases[:int(len(metaphors_phrases)*0.8)]
metaphors_phrases_both = metaphors_phrases[int(len(metaphors_phrases)*0.8):int(len(metaphors_phrases)*0.9)]
metaphors_phrases_test = metaphors_phrases[int(len(metaphors_phrases)*0.9):]

similes_phrases_train = similes_phrases[:int(len(similes_phrases)*0.8)]
similes_phrases_both = similes_phrases[int(len(similes_phrases)*0.8):int(len(similes_phrases)*0.9)]
similes_phrases_test = similes_phrases[int(len(similes_phrases)*0.9):]

output_folder = 'dataset/IRFL'
train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')
image_folder = os.path.join(output_folder, 'images')
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# # task1: True False
instruction_ = "For a given image and phrase, your task is to determine whether the image correctly illusrates the figurative meaning of the phrase. You must choose your answer from the Choice List."

train_true_datalist = []
train_false_datalist = []
test_true_datalist = []
test_false_datalist = []

for data in idioms:
    # source: image url
    # img file name = uuid
    filename = data['uuid'] + '.jpeg'
    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        print(f'file {filepath} not exist')
        continue
    phrase = data['phrase']
    question =f"\n<image>\nThe image represents the idiom \"{phrase}\", which has figurative meaning of {data['definition']}."
    # label: category == Figurative or Figurative+Literal
    answer = (data['category'] == 'Figurative') or (data['category'] == 'Figurative+Literal')
    json_data = {
        'id': data['uuid'],
        'image':[filepath],
        "conversations": [
            {
                "from": "human",
                "value": instruction_ + question + '\nChoice list:[True, False]. Your answer is:'
            },
            { 
                "from": "gpt",
                "value": "True" if answer else "False"
            }
        ]
    }
    
    if phrase in idioms_phrases_train:
        if answer:
            train_true_datalist.append(json_data)
        else:
            train_false_datalist.append(json_data)
    elif phrase in idioms_phrases_test:
        if answer:
            test_true_datalist.append(json_data)
        else:
            test_false_datalist.append(json_data)
    else:
        if random.random() > 0.5:
            if answer:
                train_true_datalist.append(json_data)
            else:
                train_false_datalist.append(json_data)
        else:
            if answer:
                test_true_datalist.append(json_data)
            else:
                test_false_datalist.append(json_data)

for data in metaphors:
    # source: image url
    # img file name = uuid
    filename = data['uuid'] + '.jpeg'
    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        print(f'file {filename} not exist')
        continue
    phrase = data['phrase']
    question =f"\n<image>\nThe image represents the metaphor \"{phrase}\"."
    # label: category == Figurative or Figurative+Literal
    answer = (data['category'] == 'Figurative') or (data['category'] == 'Figurative+Literal')

    json_data = {
        'id': data['uuid'],
        'image':[filepath],
        "conversations": [
            {
                "from": "human",
                "value": instruction_ + question + '\nChoice list:[True, False]. Your answer is:'
            },
            { 
                "from": "gpt",
                "value": "True" if answer else "False"
            }
        ]
    }
    
    if phrase in metaphors_phrases_train:
        if answer:
            train_true_datalist.append(json_data)
        else:
            train_false_datalist.append(json_data)
    elif phrase in metaphors_phrases_test:
        if answer:
            test_true_datalist.append(json_data)
        else:
            test_false_datalist.append(json_data)
    else:
        if random.random() > 0.5:
            if answer:
                train_true_datalist.append(json_data)
            else:
                train_false_datalist.append(json_data)
        else:
            if answer:
                test_true_datalist.append(json_data)
            else:
                test_false_datalist.append(json_data)

for data in similes:
    # source: image url
    # img file name = uuid
    filename = data['uuid'] + '.jpeg'
    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        print(f'file {filename} not exist')
        continue
    phrase = data['phrase']
    question =f"\n<image>\nThe image represents the simile \"{phrase}\"."
    # label: category == Figurative or Figurative+Literal
    answer = (data['category'] == 'Figurative') or (data['category'] == 'Figurative+Literal')
    json_data = {
        'id': data['uuid'],
        'image':[filepath],
        "conversations": [
            {
                "from": "human",
                "value": instruction_ + question + '\nChoice list:[True, False]. Your answer is:'
            },
            { 
                "from": "gpt",
                "value": "True" if answer else "False"
            }
        ]
    }
    
    if phrase in similes_phrases_train:
        if answer:
            train_true_datalist.append(json_data)
        else:
            train_false_datalist.append(json_data)
    elif phrase in similes_phrases_test:
        if answer:
            test_true_datalist.append(json_data)
        else:
            test_false_datalist.append(json_data)
    else:
        if random.random() > 0.5:
            if answer:
                train_true_datalist.append(json_data)
            else:
                train_false_datalist.append(json_data)
        else:
            if answer:
                test_true_datalist.append(json_data)
            else:
                test_false_datalist.append(json_data)

random.shuffle(train_false_datalist)
random.shuffle(test_false_datalist)
train_datalist = train_true_datalist + train_false_datalist[:len(train_true_datalist)]
test_datalist = test_true_datalist + test_false_datalist[:len(test_true_datalist)]

print(len(train_datalist))
print(len(test_datalist))

with open(f'{train_folder}/dataset-0.json', 'w') as json_file:
    json.dump(train_datalist, json_file, indent=4)

with open(f'{test_folder}/dataset-0.json', 'w') as json_file:
    json.dump(test_datalist, json_file, indent=4)

# breakpoint()

# task2: multi-choice vqa: image
# figurative or figurative + literal img = answer image
# one distractor: one partial literal or literal
# two random choices - combine all and sample two -> check validity by checking redundancy

# split dataset based on phrases

new_dataset_dict = {}
for idioms_phrase in idioms_phrases:
    new_datalist = [d for d in idioms if d['phrase'] == idioms_phrase]
    new_dataset_dict[idioms_phrase] = new_datalist

total_datalist = []
total_datalist.extend(idioms)

print(len(total_datalist))
print(len(new_dataset_dict))

train_datalist = []
test_datalist = []

instruction = "Given a set of images, identify the one image that correctly reprsents the figurative meaning of the phrase. You must choose your answer from the Choice List."
answers = ['Image A', 'Image B', 'Image C', 'Image D']
for phrase, datalist in new_dataset_dict.items():
    # split datalist into figurative, figurative + literal vs others
    figurative_datalist = [d for d in datalist if (d['category'] == 'Figurative') or (d['category'] == 'Figurative+Literal')]
    other_datalist = [d for d in datalist if (d['category'] != 'Figurative') and (d['category'] != 'Figurative+Literal')]
    print(phrase, len(figurative_datalist), len(other_datalist))
    for answer_item in figurative_datalist:
        ans_filepath = os.path.join(image_folder, answer_item['uuid'] + '.jpeg')
        if len(other_datalist) > 9:
            random.shuffle(other_datalist)
            for i in range(1, len(other_datalist), 2):
                dist_filepath1 = os.path.join(image_folder, other_datalist[i-1]['uuid'] + '.jpeg')
                dist_filepath2 = os.path.join(image_folder, other_datalist[i]['uuid'] + '.jpeg')
                while True:
                    random_items = random.sample(total_datalist, 1)
                    valid = True
                    random_images = []
                    for item in random_items:
                        if item['phrase'] == phrase:
                            valid = False
                            break
                        random_images.append(os.path.join(image_folder, item['uuid'] + '.jpeg'))
                    if not valid:
                        continue
                    else:
                        break
                
                imgfiles = [dist_filepath1, dist_filepath2]
                imgfiles.extend(random_images)
                random.shuffle(imgfiles)
                answer_index = random.randint(0, 3)
                imgfiles.insert(answer_index, ans_filepath)
                
                if answer_item.get('definition', None):
                    full_insturction = f"{instruction}\nQuestion: Which of the following images best represents the phrase \"{phrase}\", which has figurative definition of {answer_item['definition']}?"
                else:
                    full_insturction = f"{instruction}\nQuestion: Which of the following images best represents the phrase \"{phrase}\"?"
                
                # Structure for LLaVA JSON
                json_data = {
                    "id": f"{answer_item['uuid']}-{distractor_item['uuid']}",
                    "image": imgfiles,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"{full_insturction}\nImage A: <image>, Image B: <image>, Image C: <image>, Image D: <image>\nChoice list:[Image A, Image B, Image C, Image D]. Your answer is: "
                        },
                        { 
                            "from": "gpt",
                            "value": answers[answer_index]
                        }
                    ]
                }
                if phrase in similes_phrases_train or phrase in metaphors_phrases_train or phrase in idioms_phrases_train:
                    train_datalist.append(json_data)
                elif phrase in similes_phrases_test or phrase in metaphors_phrases_test or phrase in idioms_phrases_test:
                    test_datalist.append(json_data)
                else:
                    if random.random() > 0.5:
                        train_datalist.append(json_data)
                    else:
                        test_datalist.append(json_data)
        else:
            for distractor_item in other_datalist:
                dist_filepath = os.path.join(image_folder, distractor_item['uuid'] + '.jpeg')
                # two random candidates
                while True:
                    random_items = random.sample(total_datalist, 2)
                    valid = True
                    random_images = []
                    for item in random_items:
                        if item['phrase'] == phrase:
                            valid = False
                            break
                        random_images.append(os.path.join(image_folder, item['uuid'] + '.jpeg'))
                    if not valid:
                        continue
                    else:
                        break
                
                imgfiles = [dist_filepath]
                imgfiles.extend(random_images)
                random.shuffle(imgfiles)
                answer_index = random.randint(0, 3)
                imgfiles.insert(answer_index, ans_filepath)
                
                if answer_item.get('definition', None):
                    full_insturction = f"{instruction}\nQuestion: Which of the following images best represents the phrase \"{phrase}\", which has figurative definition of {answer_item['definition']}?"
                else:
                    full_insturction = f"{instruction}\nQuestion: Which of the following images best represents the phrase \"{phrase}\"?"
                
                # Structure for LLaVA JSON
                json_data = {
                    "id": f"{answer_item['uuid']}-{distractor_item['uuid']}",
                    "image": imgfiles,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"{full_insturction}\nImage A: <image>, Image B: <image>, Image C: <image>, Image D: <image>\nChoice list:[Image A, Image B, Image C, Image D]. Your answer is: "
                        },
                        { 
                            "from": "gpt",
                            "value": answers[answer_index]
                        }
                    ]
                }
                if phrase in similes_phrases_train or phrase in metaphors_phrases_train or phrase in idioms_phrases_train:
                    train_datalist.append(json_data)
                elif phrase in similes_phrases_test or phrase in metaphors_phrases_test or phrase in idioms_phrases_test:
                    test_datalist.append(json_data)
                else:
                    if random.random() > 0.5:
                        train_datalist.append(json_data)
                    else:
                        test_datalist.append(json_data)

print(len(train_datalist))
print(len(test_datalist))

# Shuffle the final list to mix types
random.shuffle(train_datalist)
random.shuffle(test_datalist)
# Trim to max_samples if necessary
train_datalist = train_datalist[:10000]
test_datalist = test_datalist[:2000]

with open(f'{train_folder}/dataset-1.json', 'w') as json_file:
    json.dump(train_datalist, json_file, indent=4)

with open(f'{test_folder}/dataset-1.json', 'w') as json_file:
    json.dump(test_datalist, json_file, indent=4)

new_dataset_dict = {}
for metaphors_phrase in metaphors_phrases:
    new_datalist = [d for d in metaphors if d['phrase'] == metaphors_phrase]
    new_dataset_dict[metaphors_phrase] = new_datalist
for similes_phrase in similes_phrases:
    new_datalist = [d for d in similes if d['phrase'] == similes_phrase]
    new_dataset_dict[similes_phrase] = new_datalist
total_datalist = []
total_datalist.extend(metaphors)
total_datalist.extend(similes)

answers = ['Image A', 'Image B', 'Image C', 'Image D']
for phrase, datalist in new_dataset_dict.items():
    # split datalist into figurative, figurative + literal vs others
    figurative_datalist = [d for d in datalist if (d['category'] == 'Figurative') or (d['category'] == 'Figurative+Literal')]
    other_datalist = [d for d in datalist if (d['category'] != 'Figurative') and (d['category'] != 'Figurative+Literal')]
    print(phrase, len(figurative_datalist), len(other_datalist))
    for answer_item in figurative_datalist:
        ans_filepath = os.path.join(image_folder, answer_item['uuid'] + '.jpeg')
        if len(other_datalist) > 9:
            random.shuffle(other_datalist)
            for i in range(1, len(other_datalist), 2):
                dist_filepath1 = os.path.join(image_folder, other_datalist[i-1]['uuid'] + '.jpeg')
                dist_filepath2 = os.path.join(image_folder, other_datalist[i]['uuid'] + '.jpeg')
                while True:
                    random_items = random.sample(total_datalist, 1)
                    valid = True
                    random_images = []
                    for item in random_items:
                        if item['phrase'] == phrase:
                            valid = False
                            break
                        random_images.append(os.path.join(image_folder, item['uuid'] + '.jpeg'))
                    if not valid:
                        continue
                    else:
                        break
                
                imgfiles = [dist_filepath1, dist_filepath2]
                imgfiles.extend(random_images)
                random.shuffle(imgfiles)
                answer_index = random.randint(0, 3)
                imgfiles.insert(answer_index, ans_filepath)
                
                if answer_item.get('definition', None):
                    full_insturction = f"{instruction}\nQuestion: Which of the following images best represents the phrase \"{phrase}\", which has figurative definition of {answer_item['definition']}?"
                else:
                    full_insturction = f"{instruction}\nQuestion: Which of the following images best represents the phrase \"{phrase}\"?"
                
                # Structure for LLaVA JSON
                json_data = {
                    "id": f"{answer_item['uuid']}-{distractor_item['uuid']}",
                    "image": imgfiles,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"{full_insturction}\nImage A: <image>, Image B: <image>, Image C: <image>, Image D: <image>\nChoice list:[Image A, Image B, Image C, Image D]. Your answer is: "
                        },
                        { 
                            "from": "gpt",
                            "value": answers[answer_index]
                        }
                    ]
                }
                if phrase in similes_phrases_train or phrase in metaphors_phrases_train or phrase in idioms_phrases_train:
                    train_datalist.append(json_data)
                elif phrase in similes_phrases_test or phrase in metaphors_phrases_test or phrase in idioms_phrases_test:
                    test_datalist.append(json_data)
                else:
                    if random.random() > 0.5:
                        train_datalist.append(json_data)
                    else:
                        test_datalist.append(json_data)
        else:
            for distractor_item in other_datalist:
                dist_filepath = os.path.join(image_folder, distractor_item['uuid'] + '.jpeg')
                # two random candidates
                while True:
                    random_items = random.sample(total_datalist, 2)
                    valid = True
                    random_images = []
                    for item in random_items:
                        if item['phrase'] == phrase:
                            valid = False
                            break
                        random_images.append(os.path.join(image_folder, item['uuid'] + '.jpeg'))
                    if not valid:
                        continue
                    else:
                        break
                
                imgfiles = [dist_filepath]
                imgfiles.extend(random_images)
                random.shuffle(imgfiles)
                answer_index = random.randint(0, 3)
                imgfiles.insert(answer_index, ans_filepath)
                
                if answer_item.get('definition', None):
                    full_insturction = f"{instruction}\nQuestion: Which of the following images best represents the phrase \"{phrase}\", which has figurative definition of {answer_item['definition']}?"
                else:
                    full_insturction = f"{instruction}\nQuestion: Which of the following images best represents the phrase \"{phrase}\"?"
                
                # Structure for LLaVA JSON
                json_data = {
                    "id": f"{answer_item['uuid']}-{distractor_item['uuid']}",
                    "image": imgfiles,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"{full_insturction}\nImage A: <image>, Image B: <image>, Image C: <image>, Image D: <image>\nChoice list:[Image A, Image B, Image C, Image D]. Your answer is: "
                        },
                        { 
                            "from": "gpt",
                            "value": answers[answer_index]
                        }
                    ]
                }
                if phrase in similes_phrases_train or phrase in metaphors_phrases_train or phrase in idioms_phrases_train:
                    train_datalist.append(json_data)
                elif phrase in similes_phrases_test or phrase in metaphors_phrases_test or phrase in idioms_phrases_test:
                    test_datalist.append(json_data)
                else:
                    if random.random() > 0.5:
                        train_datalist.append(json_data)
                    else:
                        test_datalist.append(json_data)

print(len(train_datalist))
print(len(test_datalist))

# Shuffle the final list to mix types
random.shuffle(train_datalist)
random.shuffle(test_datalist)
# Trim to max_samples if necessary
train_datalist = train_datalist[:10000]
test_datalist = test_datalist[:2000]

with open(f'{train_folder}/dataset-2.json', 'w') as json_file:
    json.dump(train_datalist, json_file, indent=4)

with open(f'{test_folder}/dataset-2.json', 'w') as json_file:
    json.dump(test_datalist, json_file, indent=4)

# breakpoint()
# task3: multi-choice vqa: text
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
import torch
clip_encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device="cuda", dtype=torch.bfloat16)
clipprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
def get_top_k_similar_features(query_feature, feature_list, K=10):
    
    # Normalize the feature list and the query feature for cosine similarity
    feature_list_normalized = F.normalize(torch.stack(feature_list), dim=1)  # Normalize each feature
    query_normalized = F.normalize(query_feature, dim=0)  # Normalize the query feature

    # Compute cosine similarities between the query and all features in the list
    cosine_similarities = torch.matmul(feature_list_normalized, query_normalized)
    # Get the top-K most similar features
    top_k_values, top_k_indices = torch.topk(cosine_similarities, K+1)

    return top_k_indices

phrases = idioms_phrases + metaphors_phrases + similes_phrases
features = []
for phrase in phrases:
    text_ids = clipprocessor(text=[phrase], return_tensors="pt", padding=True)
    text_ids['input_ids'] = text_ids['input_ids'].cuda()
    text_ids['attention_mask'] = text_ids['attention_mask'].cuda()
    # text_feat = self.text_encoder(**text_ids)[1][0].to(torch.bfloat16)
    text_feat = clip_encoder.get_text_features(**text_ids)[0]
    features.append(text_feat)

train_datalist = []
test_datalist = []

total_datalist = []
total_datalist.extend(idioms)
total_datalist.extend(metaphors)
total_datalist.extend(similes)

instruction = "For a given image, determine which phrase in the choices that the image reprsents. You must choose your answer from the Choice List."
for item in total_datalist:
    if (item['category'] != 'Figurative') and (item['category'] != 'Figurative+Literal'):
        continue
    # source: image url
    # img file name = uuid
    filename = item['uuid'] + '.jpeg'
    filepath = os.path.join(image_folder, filename)
    if not os.path.exists(filepath):
        print(f'file {filename} not exist')
        continue
    phrase = item['phrase']
    question =f"\n<image>\nWhat does the image represent?"
    # label: category == Figurative or Figurative+Literal
    choice_list = [phrase]
    query_feature = features[phrases.index(phrase)]
    top_k_indices = get_top_k_similar_features(query_feature, features,15)[1:].tolist()
    top_k_indices = random.sample(top_k_indices, 3)
    for choice_idx in top_k_indices:
        other_phrase = phrases[choice_idx]
        choice_list.append(other_phrase)
    random.shuffle(choice_list)
    
    for choice in choice_list:
        if choice in idioms_definitions.keys():
            question += f"\nThe definition of \"{choice}\" is {idioms_definitions[choice]}"
    
    json_data = {
        'id': item['uuid'],
        'image':[filepath],
        "conversations": [
            {
                "from": "human",
                "value": instruction + question + f'\nChoice list:[{",".join(choice_list)}]. Your answer is:'
            },
            { 
                "from": "gpt",
                "value": phrase
            }
        ]
    }
    
    if phrase in similes_phrases_train or phrase in metaphors_phrases_train or phrase in idioms_phrases_train:
        train_datalist.append(json_data)
    elif phrase in similes_phrases_test or phrase in metaphors_phrases_test or phrase in idioms_phrases_test:
        test_datalist.append(json_data)
    else:
        if random.random() > 0.5:
            train_datalist.append(json_data)
        else:
            test_datalist.append(json_data)

print(len(train_datalist))
print(len(test_datalist))

# Shuffle the final list to mix types
random.shuffle(train_datalist)
random.shuffle(test_datalist)
# Trim to max_samples if necessary
train_datalist = train_datalist[:10000]
test_datalist = test_datalist[:2000]

with open(f'{train_folder}/dataset-3.json', 'w') as json_file:
    json.dump(train_datalist, json_file, indent=4)

with open(f'{test_folder}/dataset-3.json', 'w') as json_file:
    json.dump(test_datalist, json_file, indent=4)