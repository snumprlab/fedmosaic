import random
import numpy as np
import json
import os
import shutil
data_dir = 'dataset/COCOQA'
type_name = 'test'
max_num = 10000 if type_name == 'train' else 1000
os.makedirs(os.path.join(data_dir, type_name), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'images'), exist_ok=True)


for task_num in range(4):
    txt_questions = f"{data_dir}/{type_name}_annotations/type_{task_num}/questions.txt"
    txt_img_ids = f"{data_dir}/{type_name}_annotations/type_{task_num}/img_ids.txt"
    txt_answers = f"{data_dir}/{type_name}_annotations/type_{task_num}/answers.txt"
    out_json = f"{data_dir}/{type_name}/dataset-{task_num}.json"

    # 파일 읽기
    with open(txt_questions, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f.readlines()]

    with open(txt_img_ids, "r", encoding="utf-8") as f:
        img_ids = [line.strip() for line in f.readlines()]

    with open(txt_answers, "r", encoding="utf-8") as f:
        answers = [line.strip() for line in f.readlines()]

    dataset = []
    for idx, (img_id, question, answer) in enumerate(zip(img_ids, questions, answers)):
        entry = {
            "id": idx,
            "image": f"{data_dir}/images/{img_id.zfill(12)}.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": f"Please respond accurately to the following query. <image>Question: {question} Your answer is:"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        dataset.append(entry)
        img_ids.append(img_id)
        try:
            shutil.copy(f"dataset/coco_images/train/{img_id.zfill(12)}.jpg", f"{data_dir}/images/{img_id.zfill(12)}.jpg")
        except:
            shutil.copy(f"dataset/coco_images/val/{img_id.zfill(12)}.jpg", f"{data_dir}/images/{img_id.zfill(12)}.jpg")
    #sampled_dataset, indices = random_sample_with_indices(dataset, min(len(dataset), max_num))
    print(task_num, len(dataset))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    '''
    sampled_img_ids = np.array(img_ids)[indices]
    for img_id in sampled_img_ids:
        if os.path.exists(f"{data_dir}/{type_name}_images/type_{task_num}/{img_id.zfill(12)}.jpg"):
            pass
        else:
            print(f"{data_dir}/{type_name}_images/type_{task_num}/{img_id.zfill(12)}.jpg", f"{data_dir}/images/{img_id.zfill(12)}.jpg")
        #shutil.copy(f"{data_dir}/{type_name}_images/type_{task_num}/{img_id.zfill(12)}.jpg", f"{data_dir}/images/{img_id.zfill(12)}.jpg")
    '''
