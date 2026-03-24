import json
import random
import jsonlines
import os
from PIL import Image
random.seed(42)

output_folder= 'dataset/SEED-Bench-2'
train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)


original_list = json.load(open('./dataset/SEED-Bench-2/SEED-Bench_v2_level1_2_3.json','r'))['questions']

data_per_type = {}
max_img_num = 0
for i, item in enumerate(original_list):
    image_path = item['data_id']
    if item['data_source'] == 'SEED-Bench v2':
        image_folder = os.path.join(output_folder, 'SEED-Bench-2-image')
    else:
        image_folder = os.path.join(output_folder, 'cc3m-image')
    file_path = []
    if isinstance(image_path, list):
        for data in image_path:
            file_path.append(os.path.join(image_folder, data))
    else:
        file_path.append(os.path.join(image_folder, image_path))
#    for img_path in file_path:
#        img = Image.open(img_path)
    if len(file_path) > 8:
        continue
    if len(file_path) > max_img_num:
        max_img_num = len(file_path)

    instruction_ = "Answer the following question.\n"
    if item['question_type_id'] == 23 or item['question_type_id'] == 24:
        instruction_ += f"\n{item['question']}"
        instruction_ += f"\nA. {item['choice_a']}\nB. {item['choice_b']}\nC. {item['choice_c']}\nD. {item['choice_d']}"
        instruction_ += "\nAnswer with the option's letter from the given choices directly."
        instruction_ = instruction_.replace("<img>", "<image>")
    elif item['question_type_id'] == 25:
        instruction_ += f"\n{item['question']}"
        instruction_ += "\nA. <image>\nB. <image>\nC. <image>\nD. <image>"
        instruction_ += "\nAnswer with the option's letter from the given choices directly."
    elif item['question_type_id'] == 26:
        assert len(file_path) == 7
        instruction_ += "<image><image><image>"
        instruction_ += f"\n{item['question']}"
        instruction_ += "\nA. <image>\nB. <image>\nC. <image>\nD. <image>"
        instruction_ += "\nAnswer with the option's letter from the given choices directly."
    elif item['question_type_id'] == 27:
        instruction_ += f"\n{item['question']}"
        instruction_ += f"\nA. <image> {item['choice_a']}\nB. <image> {item['choice_b']}\nC. <image> {item['choice_c']}\nD. <image> {item['choice_d']}"
        instruction_ += "\nAnswer with the option's letter from the given choices directly."
    else:
        instruction_ += "<image>"*len(file_path)
        instruction_ += f"\n{item['question']}"
        instruction_ += f"\nA. {item['choice_a']}\nB. {item['choice_b']}\nC. {item['choice_c']}\nD. {item['choice_d']}"
        instruction_ += "\nAnswer with the option's letter from the given choices directly."
    
    json_data = {
        'id': i,
        'image': file_path,
        'conversations':[
            {'from':'human',
            'value': instruction_},
            {'from':'gpt',
            'value': item['answer']}
            ]
    }

    if item['question_type_id'] in data_per_type.keys():
        data_per_type[item['question_type_id']].append(json_data)
    else:
        data_per_type[item['question_type_id']] = [json_data]
total_length = 0
for k,v in data_per_type.items():
    print(k, len(v))
    total_length += len(v)
print(total_length)
print(max_img_num)

train_datalist1 = []
test_datalist1 = []
train_datalist2 = []
test_datalist2 = []
train_datalist3 = []
test_datalist3 = []
train_datalist4 = []
test_datalist4 = []

for k,v in data_per_type.items():
    random.shuffle(v)
    split_num = int(len(v)*0.85)
    if k in[1,19,20,21,22]:
        train_datalist3.extend(v[:split_num])
        test_datalist3.extend(v[split_num:])
    elif k in [3,4,5,6,7]:
        train_datalist1.extend(v[:split_num])
        test_datalist1.extend(v[split_num:])
    elif k in [23,24,25,26,27]:
        train_datalist4.extend(v[:split_num])
        test_datalist4.extend(v[split_num:])
    else:
        train_datalist2.extend(v[:split_num])
        test_datalist2.extend(v[split_num:])
    #if k>= 17 and k <=22:
    #    train_datalist2.extend(v[:split_num])
    #    test_datalist2.extend(v[split_num:])
    #elif k >=23:
    #    train_datalist3.extend(v[:split_num])
    #    test_datalist3.extend(v[split_num:])
    #else:
    #    train_datalist1.extend(v[:split_num])
    #    test_datalist1.extend(v[split_num:])


print(len(train_datalist1))
print(len(test_datalist1))
print(len(train_datalist2))
print(len(test_datalist2))
print(len(train_datalist3))
print(len(test_datalist3))
print(len(train_datalist4))
print(len(test_datalist4))
with open(os.path.join(train_folder, 'dataset-0.json'),'w') as fp:
    json.dump(train_datalist1, fp, indent=4)

with open(os.path.join(test_folder, 'dataset-0.json'),'w') as fp:
    json.dump(test_datalist1, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-1.json'),'w') as fp:
    json.dump(train_datalist2, fp, indent=4)

with open(os.path.join(test_folder, 'dataset-1.json'),'w') as fp:
    json.dump(test_datalist2, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-2.json'),'w') as fp:
    json.dump(train_datalist3, fp, indent=4)

with open(os.path.join(test_folder, 'dataset-2.json'),'w') as fp:
    json.dump(test_datalist3, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-3.json'),'w') as fp:
    json.dump(train_datalist4, fp, indent=4)

with open(os.path.join(test_folder, 'dataset-3.json'),'w') as fp:
    json.dump(test_datalist4, fp, indent=4)
