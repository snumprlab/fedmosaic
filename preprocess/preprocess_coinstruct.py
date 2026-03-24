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
    
    
    json_data_list_train = json_data_list_1[:int(train_samples/2)] + json_data_list_2[:int(train_samples/2)]
    json_data_list_test = json_data_list_1[-int(test_samples/2):] + json_data_list_2[-int(test_samples/2):]
    
    json_data_list_2_train = json_data_list_3[:int(train_samples)]
    json_data_list_2_test = json_data_list_3[-int(test_samples):]
    
    json_data_list_3_train = json_data_list_4[:int(train_samples)]
    json_data_list_3_test = json_data_list_4[-int(test_samples):]
    
    # Save the JSON data list to a file
    json_output_path = os.path.join(train_folder, f'dataset-0.json')
    print(f"Total samples: {len(json_data_list_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-0.json')
    print(f"Total samples: {len(json_data_list_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_test, json_file, indent=4)

    json_output_path = os.path.join(train_folder, f'dataset-1.json')
    print(f"Total samples: {len(json_data_list_2_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_2_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-1.json')
    print(f"Total samples: {len(json_data_list_2_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_2_test, json_file, indent=4)
    
    json_output_path = os.path.join(train_folder, f'dataset-2.json')
    print(f"Total samples: {len(json_data_list_3_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_3_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-2.json')
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
    
    
    json_data_list_train = json_data_list_1[:int(train_samples)]
    json_data_list_test = json_data_list_1[-int(test_samples):]
    
    json_data_list_2_train = json_data_list_2[:int(train_samples)]
    json_data_list_2_test = json_data_list_2[-int(test_samples):]
    
    json_data_list_3_train = json_data_list_3[:int(train_samples)]
    json_data_list_3_test = json_data_list_3[-int(test_samples):]
    
    # Save the JSON data list to a file
    json_output_path = os.path.join(train_folder, f'dataset-3.json')
    print(f"Total samples: {len(json_data_list_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-3.json')
    print(f"Total samples: {len(json_data_list_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_test, json_file, indent=4)

    json_output_path = os.path.join(train_folder, f'dataset-4.json')
    print(f"Total samples: {len(json_data_list_2_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_2_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-4.json')
    print(f"Total samples: {len(json_data_list_2_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_2_test, json_file, indent=4)
    
    # json_output_path = os.path.join(train_folder, f'dataset-5.json')
    # print(f"Total samples: {len(json_data_list_3_train)}")
    # with open(json_output_path, 'w') as json_file:
    #     json.dump(json_data_list_3_train, json_file, indent=4)
        
    # json_output_path = os.path.join(test_folder, f'dataset-5.json')
    # print(f"Total samples: {len(json_data_list_3_test)}")
    # with open(json_output_path, 'w') as json_file:
    #     json.dump(json_data_list_3_test, json_file, indent=4)

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
def split_compare_qna(datalist, output_folder, train_samples, test_samples):
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
            item['image'] = [output_folder + '/' + img[2:] for img in item['image']]
        
        if item['conversations'][1]['value'] in ['A.', 'B.', 'C.', 'D.']:
            item['conversations'][1]['value'] = item['conversations'][1]['value'][:-1]
        elif item['conversations'][1]['value'] in ['Yes', 'No']:
            item['conversations'][0]['value'] += yes_no_choice_list
        
        elif item['conversations'][1]['value'] in ['Second image', 'First image', 'Third image', 'Fourth image', 'None of the images', "Both images", "All images"]:
            if len(item['image']) == 2:
                item['conversations'][0]['value'] += two_image_choice_list
            elif len(item['image']) == 3:
                item['conversations'][0]['value'] += three_image_choice_list
            elif len(item['image']) == 4:
                item['conversations'][0]['value'] += four_image_choice_list
        
        elif item['conversations'][1]['value'] in ["Image 1", "Image 2", "Image 3", "Image 4", "The first image", "The second image", "The third image", "The fourth image", 'None'] and 'which' in item['conversations'][0]['value'].lower():
            if len(item['image']) == 2:
                item['conversations'][0]['value'] += two_image_choice_list
            elif len(item['image']) == 3:
                item['conversations'][0]['value'] += three_image_choice_list
            elif len(item['image']) == 4:
                item['conversations'][0]['value'] += four_image_choice_list
            item['conversations'][1]['value'] = answer_dict[item['conversations'][1]['value']]
        
        elif item['conversations'][1]['value'] in ['Second', 'First', 'Third', 'Fourth', 'Neither', "Both", "All of them"] and 'which' in item['conversations'][0]['value'].lower():
            if len(item['image']) == 2:
                item['conversations'][0]['value'] += two_img_choice_list
            elif len(item['image']) == 3:
                item['conversations'][0]['value'] += three_img_choice_list
            elif len(item['image']) == 4:
                item['conversations'][0]['value'] += four_img_choice_list
        
        elif item['conversations'][1]['value'] in ['All of the above'] and 'which' in item['conversations'][0]['value'].lower():
            if len(item['image']) == 2:
                item['conversations'][0]['value'] += two_img_choice_list
            elif len(item['image']) == 3:
                item['conversations'][0]['value'] += three_img_choice_list
            elif len(item['image']) == 4:
                item['conversations'][0]['value'] += four_img_choice_list
            item['conversations'][1]['value'] = answer_dict[item['conversations'][1]['value']]
        
        elif item['conversations'][1]['value'] in ['One', 'Two', 'Three', 'Four', 'Four or more']:
            if item['conversations'][1]['value'] == 'Four or More':
                item['conversations'][0]['value'] += count_v2
            elif item['conversations'][1]['value'] == 'Four':
                item['conversations'][0]['value'] += count_v1
            else:
                if random.random() > 0.5:
                    item['conversations'][0]['value'] += count_v1
                else:
                    item['conversations'][0]['value'] += count_v2
        elif item['conversations'][1]['value'] == 'None' and 'how' in item['conversations'][0]['value'].lower():
            if random.random() > 0.5:
                item['conversations'][0]['value'] += count_v1
            else:
                item['conversations'][0]['value'] += count_v2
        
        # elif item['conversations'][1]['value'] in ['Both first and second images', 'Both second and third images', 'Both first and third images']:
        #     if len(item['image']) == 3:
        #         item['conversations'][0]['value'] += both_image_choice_list
        else:
            continue
        
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
    
    
    json_data_list_train = json_data_list_1[:int(train_samples)]
    json_data_list_test = json_data_list_1[-int(test_samples):]
    
    json_data_list_2_train = json_data_list_2[:int(train_samples)]
    json_data_list_2_test = json_data_list_2[-int(test_samples):]
    
    json_data_list_3_train = json_data_list_3[:int(train_samples)]
    json_data_list_3_test = json_data_list_3[-int(test_samples):]
    
    # Save the JSON data list to a file
    json_output_path = os.path.join(train_folder, f'dataset-5.json')
    print(f"Total samples: {len(json_data_list_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-5.json')
    print(f"Total samples: {len(json_data_list_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_test, json_file, indent=4)

    json_output_path = os.path.join(train_folder, f'dataset-6.json')
    print(f"Total samples: {len(json_data_list_2_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_2_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-6.json')
    print(f"Total samples: {len(json_data_list_2_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_2_test, json_file, indent=4)

    json_output_path = os.path.join(train_folder, f'dataset-7.json')
    print(f"Total samples: {len(json_data_list_3_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_3_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-7.json')
    print(f"Total samples: {len(json_data_list_3_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_3_test, json_file, indent=4)
    
    random.shuffle(json_data_list_1)
    random.shuffle(json_data_list_2)
    random.shuffle(json_data_list_3)
    json_data_list_4_train = json_data_list_1[:3333] + json_data_list_2[:3333] + json_data_list_3[:3333]
    json_data_list_4_test = json_data_list_1[-666:] + json_data_list_2[-666:] + json_data_list_3[-666:]
    
    json_output_path = os.path.join(train_folder, f'dataset-8.json')
    print(f"Total samples: {len(json_data_list_4_train)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_4_train, json_file, indent=4)
        
    json_output_path = os.path.join(test_folder, f'dataset-8.json')
    print(f"Total samples: {len(json_data_list_4_test)}")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list_4_test, json_file, indent=4)
    

    
with open('./dataset/Co-Instruct-DB/coinstruct_562k_llava_format.json', 'r') as fp:
    datalist = json.load(fp)

original_q_instruct = datalist[:200277]
multi_q_instruct = datalist[200277:302409]
compare_general = datalist[302409:332301]
compare_qna = datalist[332301:]

split_q_instruct(original_q_instruct, 'dataset/Co-Instruct-DB', 10000, 2000)

split_multi_q_instruct(multi_q_instruct, 'dataset/Co-Instruct-DB', 10000, 2000)

split_compare_qna(compare_qna, 'dataset/Co-Instruct-DB', 10000, 2000)