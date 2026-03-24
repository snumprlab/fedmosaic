import json
import os
import random
random.seed(42)

dir = 'dataset/FS_LLM_Instruct'

train_folder = os.path.join(dir, 'train')
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
    
test_folder = os.path.join(dir, 'test')
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# gsm: math dataset, no category (total 7473) --> random split
# dolly: instruction tuning dataset (total 15015) --> categorized into question type (total 8)
# rosetta: coding dataset (total 7954)--> categorized into programming language (total 9)

# type1: each client learn each downstream-task (instruction tuning / math / coding) incrementally
# client1: gsm 4
# client2: dolly 4
# client3: dolly 4
# client4: rosetta 4
# client5: rosetta 4

# gsm
datalist = json.load(open(os.path.join(dir, 'gsm.json'),'r'))

random.shuffle(datalist)

balanced_split_num = len(datalist)//4
datalist1 = datalist[0:balanced_split_num]
datalist2 = datalist[balanced_split_num:balanced_split_num*2]
datalist3 = datalist[balanced_split_num*2:balanced_split_num*3]
datalist4 = datalist[balanced_split_num*3:]

with open(os.path.join(train_folder, 'dataset-0.json'),'w') as fp:
    json.dump(datalist1, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-1.json'),'w') as fp:
    json.dump(datalist2, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-2.json'),'w') as fp:
    json.dump(datalist3, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-3.json'),'w') as fp:
    json.dump(datalist4, fp, indent=4)

with open(os.path.join(test_folder, 'dataset-0.json'),'w') as fp:
    json.dump(datalist1[-100:], fp, indent=4)

with open(os.path.join(test_folder, 'dataset-1.json'),'w') as fp:
    json.dump(datalist2[-100:], fp, indent=4)

with open(os.path.join(test_folder, 'dataset-2.json'),'w') as fp:
    json.dump(datalist3[-100:], fp, indent=4)

with open(os.path.join(test_folder, 'dataset-3.json'),'w') as fp:
    json.dump(datalist4[-100:], fp, indent=4)


# dolly
datalist = json.load(open(os.path.join(dir, 'dolly_meta.json'),'r'))
# {'closed_qa': 1823, 'classification': 2136, 'open_qa': 3611, 'information_extraction': 1512, 'brainstorming': 1768, 'general_qa': 2191, 'summarization': 1263, 'creative_writing': 711}

datalist1 = [item for item in datalist if item['category'] == 'classification']
datalist2 = [item for item in datalist if item['category'] == 'closed_qa']
datalist3 = [item for item in datalist if item['category'] == 'general_qa']
datalist4 = [item for item in datalist if item['category'] == 'open_qa']

datalist5 = [item for item in datalist if item['category'] == 'information_extraction']
datalist6 = [item for item in datalist if item['category'] == 'creative_writing']
datalist7 = [item for item in datalist if item['category'] == 'brainstorming']
datalist8 = [item for item in datalist if item['category'] == 'summarization']

with open(os.path.join(train_folder, 'dataset-10.json'),'w') as fp:
    json.dump(datalist1, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-11.json'),'w') as fp:
    json.dump(datalist2, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-12.json'),'w') as fp:
    json.dump(datalist3, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-13.json'),'w') as fp:
    json.dump(datalist4, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-14.json'),'w') as fp:
    json.dump(datalist5, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-15.json'),'w') as fp:
    json.dump(datalist6, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-16.json'),'w') as fp:
    json.dump(datalist7, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-17.json'),'w') as fp:
    json.dump(datalist8, fp, indent=4)

with open(os.path.join(test_folder, 'dataset-10.json'),'w') as fp:
    json.dump(datalist1[-100:], fp, indent=4)

with open(os.path.join(test_folder, 'dataset-11.json'),'w') as fp:
    json.dump(datalist2[-100:], fp, indent=4)

with open(os.path.join(test_folder, 'dataset-12.json'),'w') as fp:
    json.dump(datalist3[-100:], fp, indent=4)

with open(os.path.join(test_folder, 'dataset-13.json'),'w') as fp:
    json.dump(datalist4[-100:], fp, indent=4)

with open(os.path.join(test_folder, 'dataset-14.json'),'w') as fp:
    json.dump(datalist5[-100:], fp, indent=4)

with open(os.path.join(test_folder, 'dataset-15.json'),'w') as fp:
    json.dump(datalist6[-100:], fp, indent=4)

with open(os.path.join(test_folder, 'dataset-16.json'),'w') as fp:
    json.dump(datalist7[-100:], fp, indent=4)

with open(os.path.join(test_folder, 'dataset-17.json'),'w') as fp:
    json.dump(datalist8[-100:], fp, indent=4)

# rosetta (coding)
datalist = json.load(open(os.path.join(dir, 'rosetta_meta.json'),'r'))
# {'Go': 1172, 'Python': 1139, 'C': 1024, 'C#': 832, 'C++': 964, 'Java': 1004, 'Pascal': 547, 'PHP': 439, 'Scala': 833}
datalist1 = [item for item in datalist if item['category'] == 'Java']
datalist2 = [item for item in datalist if item['category'] == 'C' or item['category'] == 'PHP']
datalist3 = [item for item in datalist if item['category'] == 'C#']
datalist4 = [item for item in datalist if item['category'] == 'C++']

datalist5 = [item for item in datalist if item['category'] == 'Python']
datalist6 = [item for item in datalist if item['category'] == 'Go']
datalist7 = [item for item in datalist if item['category'] == 'Pascal']
datalist8 = [item for item in datalist if item['category'] == 'Scala']

with open(os.path.join(train_folder, 'dataset-20.json'),'w') as fp:
    json.dump(datalist1, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-21.json'),'w') as fp:
    json.dump(datalist2, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-22.json'),'w') as fp:
    json.dump(datalist3, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-23.json'),'w') as fp:
    json.dump(datalist4, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-24.json'),'w') as fp:
    json.dump(datalist5, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-25.json'),'w') as fp:
    json.dump(datalist6, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-26.json'),'w') as fp:
    json.dump(datalist7, fp, indent=4)

with open(os.path.join(train_folder, 'dataset-27.json'),'w') as fp:
    json.dump(datalist8, fp, indent=4)

with open(os.path.join(test_folder, 'dataset-20.json'),'w') as fp:
    json.dump(datalist1[-100:], fp, indent=4)

with open(os.path.join(test_folder, 'dataset-21.json'),'w') as fp:
    json.dump(datalist2[-100:], fp, indent=4)

with open(os.path.join(test_folder, 'dataset-22.json'),'w') as fp:
    json.dump(datalist3[-100:], fp, indent=4)

with open(os.path.join(test_folder, 'dataset-23.json'),'w') as fp:
    json.dump(datalist4[-100:], fp, indent=4)

with open(os.path.join(test_folder, 'dataset-24.json'),'w') as fp:
    json.dump(datalist5[-100:], fp, indent=4)

with open(os.path.join(test_folder, 'dataset-25.json'),'w') as fp:
    json.dump(datalist6[-100:], fp, indent=4)

with open(os.path.join(test_folder, 'dataset-26.json'),'w') as fp:
    json.dump(datalist7[-100:], fp, indent=4)

with open(os.path.join(test_folder, 'dataset-27.json'),'w') as fp:
    json.dump(datalist8[-100:], fp, indent=4)


# type2: each client learn three downstream-task incrementally (with different train subset)
gsm_datalist = json.load(open(os.path.join(dir, 'gsm.json'),'r'))

random.shuffle(datalist)

balanced_split_num = len(datalist)//8
gsm_datalist1 = gsm_datalist[0:balanced_split_num]
gsm_datalist2 = gsm_datalist[balanced_split_num:balanced_split_num*2]
gsm_datalist3 = gsm_datalist[balanced_split_num*2:balanced_split_num*3]
gsm_datalist4 = gsm_datalist[balanced_split_num*3:balanced_split_num*4]
gsm_datalist5 = gsm_datalist[balanced_split_num*4:balanced_split_num*5]
gsm_datalist6 = gsm_datalist[balanced_split_num*5:balanced_split_num*6]
gsm_datalist7 = gsm_datalist[balanced_split_num*6:balanced_split_num*7]
gsm_datalist8 = gsm_datalist[balanced_split_num*7:]
gsm_datalists = [gsm_datalist1,gsm_datalist2,gsm_datalist3,gsm_datalist4,gsm_datalist5,gsm_datalist6,gsm_datalist7,gsm_datalist8]

dolly_datalist = json.load(open(os.path.join(dir, 'dolly_meta.json'),'r'))
# {'closed_qa': 1823, 'classification': 2136, 'open_qa': 3611, 'information_extraction': 1512, 'brainstorming': 1768, 'general_qa': 2191, 'summarization': 1263, 'creative_writing': 711}

dolly_datalist1 = [item for item in dolly_datalist if item['category'] == 'classification']
dolly_datalist2 = [item for item in dolly_datalist if item['category'] == 'closed_qa']
dolly_datalist3 = [item for item in dolly_datalist if item['category'] == 'general_qa']
dolly_datalist4 = [item for item in dolly_datalist if item['category'] == 'open_qa']

dolly_datalist5 = [item for item in dolly_datalist if item['category'] == 'information_extraction']
dolly_datalist6 = [item for item in dolly_datalist if item['category'] == 'creative_writing']
dolly_datalist7 = [item for item in dolly_datalist if item['category'] == 'brainstorming']
dolly_datalist8 = [item for item in dolly_datalist if item['category'] == 'summarization']
dolly_datalists = [dolly_datalist1,dolly_datalist2,dolly_datalist3,dolly_datalist4,dolly_datalist5,dolly_datalist6,dolly_datalist7,dolly_datalist8]


# rosetta (coding)
rosetta_datalist = json.load(open(os.path.join(dir, 'rosetta_meta.json'),'r'))
# {'Go': 1172, 'Python': 1139, 'C': 1024, 'C#': 832, 'C++': 964, 'Java': 1004, 'Pascal': 547, 'PHP': 439, 'Scala': 833}
rosetta_datalist1 = [item for item in rosetta_datalist if item['category'] == 'Java']
rosetta_datalist2 = [item for item in rosetta_datalist if item['category'] == 'C' or item['category'] == 'PHP']
rosetta_datalist3 = [item for item in rosetta_datalist if item['category'] == 'C#']
rosetta_datalist4 = [item for item in rosetta_datalist if item['category'] == 'C++']
rosetta_datalist5 = [item for item in rosetta_datalist if item['category'] == 'Python']
rosetta_datalist6 = [item for item in rosetta_datalist if item['category'] == 'Go']
rosetta_datalist7 = [item for item in rosetta_datalist if item['category'] == 'Pascal']
rosetta_datalist8 = [item for item in rosetta_datalist if item['category'] == 'Scala']
rosetta_datalists = [rosetta_datalist1, rosetta_datalist2,rosetta_datalist3,rosetta_datalist4,rosetta_datalist5,rosetta_datalist6,rosetta_datalist7,rosetta_datalist8]

for client_id in range(8):
    gsm = gsm_datalists.pop(random.randrange(len(gsm_datalists)))
    dolly = dolly_datalists.pop(random.randrange(len(dolly_datalists)))
    rosetta = rosetta_datalists.pop(random.randrange(len(rosetta_datalists)))
    
    # Combine the popped elements and shuffle them.
    three_elements = [gsm, dolly, rosetta]
    random.shuffle(three_elements)
    
    for i in range(3):
        with open(os.path.join(train_folder, f'dataset-1{client_id}{i}.json'), 'w') as fp:
            json.dump(three_elements[i], fp, indent=4)
        with open(os.path.join(test_folder, f'dataset-1{client_id}{i}.json'), 'w') as fp:
            json.dump(three_elements[i][-100:], fp, indent=4)