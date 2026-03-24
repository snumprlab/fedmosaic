# from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
import os
import json
import glob
import random
import numpy as np
import itertools
import jsonlines
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn.functional as F

random.seed(123)
np.random.seed(123)

num_per_set = 3
num_combination_per_sample = 14
prompts = f'''Given {num_per_set} "positive" images and {num_per_set} "negative" images, where "positive" images can be summarized as 1 "common" sentence and "negative" images cannot, the "common" sentence describes a set of concepts that are common to "positive" images. Your job is to find the "common" concept within the "positive" images. You must choose your answer from the Choice List.'''

clip_encoder = CLIPModel.from_pretrained("./clip-vit-large-patch14-336").to(device="cuda", dtype=torch.bfloat16)
clipprocessor = CLIPProcessor.from_pretrained("./models/clip_models/clipprocessor/")



def get_top_k_similar_features(query_feature, feature_list, K=10):
    
    # Normalize the feature list and the query feature for cosine similarity
    feature_list_normalized = F.normalize(torch.stack(feature_list), dim=1)  # Normalize each feature
    query_normalized = F.normalize(query_feature, dim=0)  # Normalize the query feature

    # Compute cosine similarities between the query and all features in the list
    cosine_similarities = torch.matmul(feature_list_normalized, query_normalized)
    # Get the top-K most similar features
    top_k_values, top_k_indices = torch.topk(cosine_similarities, K+1)

    return top_k_indices


def save_dataset(dataset_name, output_folder, subset_name, answers, features):
    if subset_name == 'train':
        sample_size = 10000
    elif subset_name == 'test':
        sample_size = 2000
        
    subset_folder = os.path.join(output_folder, subset_name)
    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)
        
    with open(f"{output_folder}/{subset_name}.json") as fp:
        datalist = json.load(fp)
    json_data_list = []
    
    for item in datalist:
        
        answer = item['caption']
        positive_imgfiles = item['imageFiles'][:7]
        negative_imgfiles = item['imageFiles'][7:]
        positive_imgfiles = ["dataset/Bongard-OpenWorld/"+path for path in positive_imgfiles]
        negative_imgfiles = ["dataset/Bongard-OpenWorld/"+path for path in negative_imgfiles]
        
        arr = np.arange(len(positive_imgfiles))
        nCr = list(itertools.combinations(arr, num_per_set))
        random.shuffle(nCr)
        
        for idx, index in enumerate(nCr[:num_combination_per_sample]):
            index = np.array(list(index))
            random.shuffle(index)
            imgs = [positive_imgfiles[i] for i in index] + [negative_imgfiles[i] for i in index]
        
            choice_list = [answer]
            
            query_feature = features[answers.index(answer)]
            top_k_indices = get_top_k_similar_features(query_feature, features)[1:].tolist()
            top_k_indices = random.sample(top_k_indices, 3)
            for choice_idx in top_k_indices:
                choice_list.append(answers[choice_idx])
            choice_list = sorted(choice_list)
        
            # Structure for LLaVA JSON
            json_data = {
                "id": item['uid'] + "-" + str(idx),
                "image": imgs,#" |sep| ".join(imgs),
                "commonSense":item["commonSense"],
                "conversations": [
                    {
                        "from": "human",
                        "value": "Positive: " +  "<image>"*num_per_set + "\nNegative: " + "<image>"*num_per_set + "\n" + prompts + "\n" + "Choice List: [" + ', '.join(choice_list) + "]"
                        # "value": "<image>"*len(imgs) + "\n" + prompts
                    },
                    { 
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }
            json_data_list.append(json_data)

    # Save the JSON data list to a file
    json_output_path = os.path.join(output_folder, subset_name, f'dataset-3.json')
    print(len(json_data_list))
    if len(json_data_list) > sample_size:
        json_data_list = np.random.choice(json_data_list, size=sample_size, replace=False).tolist()
        print(len(json_data_list))
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)

# Usage example
output_folder = 'dataset/Bongard-OpenWorld'

# preprocess jsonl to json
# train (combine train and val)
train_data = []
with jsonlines.open(f"{output_folder}/train.jsonl") as f:
    for line in f.iter():
        train_data.append(line)

with jsonlines.open(f"{output_folder}/val.jsonl") as f:
    for line in f.iter():
        train_data.append(line)

with open(f"{output_folder}/train.json", 'w') as json_file:
    json.dump(train_data, json_file, indent=4)  



# test
test_data = []
with jsonlines.open(f"{output_folder}/test.jsonl") as f:
    for line in f.iter():
        test_data.append(line)

with open(f"{output_folder}/test.json", 'w') as json_file:
    json.dump(test_data, json_file, indent=4)  


answers = []
features = []
with open(f"{output_folder}/train.json") as fp:
    datalist = json.load(fp)
for item in datalist:
    if item['caption'] not in answers:
        answers.append(item['caption'])
        text_ids = clipprocessor(text=[item['caption']], return_tensors="pt", padding=True)
        text_ids['input_ids'] = text_ids['input_ids'].cuda()
        text_ids['attention_mask'] = text_ids['attention_mask'].cuda()
        # text_feat = self.text_encoder(**text_ids)[1][0].to(torch.bfloat16)
        text_feat = clip_encoder.get_text_features(**text_ids)[0]
        features.append(text_feat)

with open(f"{output_folder}/test.json") as fp:
    datalist = json.load(fp)
for item in datalist:
    if item['caption'] not in answers:
        answers.append(item['caption'])
        text_ids = clipprocessor(text=[item['caption']], return_tensors="pt", padding=True)
        text_ids['input_ids'] = text_ids['input_ids'].cuda()
        text_ids['attention_mask'] = text_ids['attention_mask'].cuda()
        # text_feat = self.text_encoder(**text_ids)[1][0].to(torch.bfloat16)
        text_feat = clip_encoder.get_text_features(**text_ids)[0]
        features.append(text_feat)


save_dataset('Bongard-OpenWorld', output_folder, 'test', answers, features)
save_dataset('Bongard-OpenWorld', output_folder, 'train', answers, features)

