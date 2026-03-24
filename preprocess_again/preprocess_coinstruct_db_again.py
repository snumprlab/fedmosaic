import json
import random
import os

random.seed(42)

def split_q_instruct(datalist, output_folder, train_samples, test_samples):
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    
    json_data_list_1 = []
    json_data_list_2 = []
    json_data_list_3 = []
    json_data_list_4 = []
    
    for item in datalist:
        item['image'] = output_folder + '/' + item['image']
        
        if "what" in item['conversations'][0]['value'].lower() and item['conversations'][1]['value'] in ['A.','B.','C.','D.', 'E.']:
            item['conversations'][1]['value'] = item['conversations'][1]['value'][:-1]
            json_data_list_1.append(item)
        elif "how" in item['conversations'][0]['value'].lower() and item['conversations'][1]['value'] in ['A.','B.','C.','D.', 'E.']:
            item['conversations'][1]['value'] = item['conversations'][1]['value'][:-1]
            json_data_list_2.append(item)
        elif "yes" in item['conversations'][0]['value'].lower() and "no" in item['conversations'][0]['value'].lower() and item['conversations'][1]['value'] in ['A.','B.','C.','D.', 'E.']:
            item['conversations'][1]['value'] = item['conversations'][1]['value'][:-1]
            json_data_list_3.append(item)
        elif len(item['conversations'][1]['value'].split(' ')) > 3:
            json_data_list_4.append(item)
    
    print(len(json_data_list_1), len(json_data_list_2), len(json_data_list_3), len(json_data_list_4))
    # Shuffle the final list to mix types
    random.shuffle(json_data_list_1)
    random.shuffle(json_data_list_2)
    random.shuffle(json_data_list_3)
    random.shuffle(json_data_list_4)
    
    train_samples = 8000
    json_data_list_train = json_data_list_1[:int(train_samples/2)] + json_data_list_2[:int(train_samples/2)]
    json_data_list_test = json_data_list_1[-int(test_samples/2):] + json_data_list_2[-int(test_samples/2):]
    
    # Save the JSON data list to a file
    json_output_path = os.path.join(train_folder, f'dataset-10.json')
    print(f"Total samples: {len(json_data_list_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-10.json')
    print(f"Total samples: {len(json_data_list_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_test, json_file, indent=4)
    
    train_samples = 5000
    json_data_list_train = json_data_list_1[:int(train_samples/2)] + json_data_list_2[:int(train_samples/2)]
    json_data_list_test = json_data_list_1[-int(test_samples/2):] + json_data_list_2[-int(test_samples/2):]
    
    # Save the JSON data list to a file
    json_output_path = os.path.join(train_folder, f'dataset-20.json')
    print(f"Total samples: {len(json_data_list_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-20.json')
    print(f"Total samples: {len(json_data_list_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_test, json_file, indent=4)
    
    train_samples = 7000
    json_data_list_2_train = json_data_list_3[:int(train_samples)]
    json_data_list_2_test = json_data_list_3[-int(test_samples):]
    
    json_output_path = os.path.join(train_folder, f'dataset-21.json')
    print(f"Total samples: {len(json_data_list_2_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_2_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-21.json')
    print(f"Total samples: {len(json_data_list_2_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_2_test, json_file, indent=4)
    
    
    train_samples = 6000
    json_data_list_3_train = json_data_list_4[:int(train_samples)]
    json_data_list_3_test = json_data_list_4[-int(test_samples):]
    
    json_output_path = os.path.join(train_folder, f'dataset-12.json')
    print(f"Total samples: {len(json_data_list_3_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_3_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-12.json')
    print(f"Total samples: {len(json_data_list_3_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_3_test, json_file, indent=4)
    
    train_samples = 8000
    json_data_list_3_train = json_data_list_4[:int(train_samples)]
    json_data_list_3_test = json_data_list_4[-int(test_samples):]
    
    json_output_path = os.path.join(train_folder, f'dataset-22.json')
    print(f"Total samples: {len(json_data_list_3_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_3_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-22.json')
    print(f"Total samples: {len(json_data_list_3_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_3_test, json_file, indent=4)
        
        
def split_multi_q_instruct(datalist, output_folder, train_samples, test_samples):
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    
    json_data_list_1 = []
    json_data_list_2 = []
    json_data_list_3 = []
    
    for item in datalist:
        if isinstance(item['image'], list):
            item['image'] = [output_folder + '/' + img for img in item['image']]
        
        if len(item['image']) == 2:
            json_data_list_1.append(item)
        elif len(item['image']) == 3:
            json_data_list_2.append(item)
        elif len(item['image']) == 4:
            json_data_list_3.append(item)
    
    print(len(json_data_list_1), len(json_data_list_2), len(json_data_list_3))
    # Shuffle the final list to mix types
    random.shuffle(json_data_list_1)
    random.shuffle(json_data_list_2)
    random.shuffle(json_data_list_3)
    
    train_samples = 8000
    json_data_list_train = json_data_list_1[:int(train_samples)]
    json_data_list_test = json_data_list_1[-int(test_samples):]
    
    # Save the JSON data list to a file
    json_output_path = os.path.join(train_folder, f'dataset-23.json')
    print(f"Total samples: {len(json_data_list_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-23.json')
    print(f"Total samples: {len(json_data_list_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_test, json_file, indent=4)
    

yes_no_choice_list = '\nChoice list:[Yes, No]. You must choose your answer from the Choice List. '

two_image_choice_list = '\nChoice list:[First image, Second image, None of the images, Both images]. You must choose your answer from the Choice List. '
three_image_choice_list = '\nChoice list:[First image, Second image, Third image, None of the images, All images]. You must choose your answer from the Choice List. '
four_image_choice_list = '\nChoice list:[First image, Second image, Third image, Fourth image, None of the images, All images]. You must choose your answer from the Choice List. '

# both_image_choice_list = '\nChoice list:[Both first and second images, Both second and third images, Both first and third images, All images]. You must choose your answer from the Choice List. '

two_img_choice_list = '\nChoice list:[First, Second, Neither, Both]. You must choose your answer from the Choice List. '
three_img_choice_list = '\nChoice list:[First, Second, Third, All of them]. You must choose your answer from the Choice List. '
four_img_choice_list = '\nChoice list:[First, Second, Third, Fourth, All of them]. You must choose your answer from the Choice List. '

count_v1 = '\nChoice list:[None, One, Two, Three, Four]. You must choose your answer from the Choice List. '
count_v2 = '\nChoice list:[None, One, Two, Three, Four or more]. You must choose your answer from the Choice List. '

answer_dict = {
    "Image 1": "First image",
    "Image 2": "Second image",
    "Image 3": "Third image",
    "Image 4": "Fourth image",
    "The first image": "First image",
    "The second image": "Second image",
    "The third image": "Third image",
    "The fourth image": "Fourth image",
    # "Both First and Second":"Both first and second images",
    "All of the above": "All of them",
    "Neither images": "None of the images",
    "None": "None of the images"
}


    
with open('./dataset/Co-Instruct-DB/coinstruct_562k_llava_format.json', 'r') as fp:
    datalist = json.load(fp)

original_q_instruct = datalist[:200277]
multi_q_instruct = datalist[200277:302409]
compare_general = datalist[302409:332301]
compare_qna = datalist[332301:]

split_q_instruct(original_q_instruct, 'dataset/Co-Instruct-DB', 10000, 2000)

split_multi_q_instruct(multi_q_instruct, 'dataset/Co-Instruct-DB', 10000, 2000)
