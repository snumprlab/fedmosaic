import json
import os
import random
random.seed(42)

from collections import OrderedDict

def topic_analyze(split):

    datalist = json.load(open(f'dataset/Fed-aya_preprocessed/train/dataset-{split}_category.json','r'))
    # datalist = json.load(open(f'dataset/Fed-aya2/train/dataset-{split}.json','r'))
    print(len(datalist))

    category_cnt = OrderedDict()

    for item in datalist:
        if item['super_category'] not in category_cnt.keys():
            category_cnt[item['super_category']] = OrderedDict()
        
        if item['sub_category'] not in category_cnt[item['super_category']].keys():
            category_cnt[item['super_category']][item['sub_category']] = [item]
        else:
            category_cnt[item['super_category']][item['sub_category']].append(item)
        
        
    super_categories = [
        'Science',
        'Mathematics',
        'Technology & Engineering',
        'Philosophy',
        'Geography',
        'History',
        'Psychology',
        'Politics',
        'Economics & Finance',
        'Government & Law',
        'Health & Medicine',
        'Social Sciences',
        'Literature',
        'Linguistics',
        'Arts',
        'Education & Knowledge',
        'Food',
        'Entertainment',
        'Sports & Exercise',
        'Fashion & Beauty',
        'Hobbies & Crafts'
        'Career & Business',
        'Environment & Nature',
        'Culture',
        'Travel',
        'News',
        'Agriculture',
        'Safety & Public Safety',
        'Lifestyle & Home',
        'Religion & Spirituality',
        'Personal Development',
        'Military & Defense',
        'Parenting & Children',
    ]

    # category_groups = {
    #     "Academic & Theoretical Knowledge": [
    #         "Science",
    #         "Mathematics",
    #         "Technology & Engineering",
    #         "Philosophy",
    #         "Psychology",
    #         "Geography",
    #         "History",
    #         "Linguistics"
    #     ],
        
    #     "Society & Governance": [
    #         "Politics",
    #         "Economics & Finance",
    #         "Government & Law",
    #         "Social Sciences",
    #         "News",
    #         "Safety & Public Safety",
    #         "Military & Defense",
    #         "Culture"
    #     ],
        
    #     "Health & Lifestyle": [
    #         "Health & Medicine",
    #         "Sports & Exercise",
    #         "Food",
    #         "Environment & Nature",
    #         "Agriculture",
    #         "Lifestyle & Home",
    #         "Parenting & Children",
    #         "Personal Development"
    #     ],
        
    #     "Arts, Career & Leisure": [
    #         "Literature",
    #         "Arts",
    #         "Education & Knowledge",
    #         "Entertainment",
    #         "Fashion & Beauty",
    #         "Hobbies & Crafts",
    #         "Career & Business",
    #         "Travel",
    #         "Religion & Spirituality"
    #     ],
    #     "Others":[],
    # }
    # category_groups = {
    #     "Science & Mathematics": [
    #         "Science",
    #         "Mathematics",
    #         "Technology & Engineering",
    #         "Environment & Nature",
    #         "Agriculture"
    #     ],
        
    #     "Humanities & Philosophy": [
    #         "Philosophy",
    #         "History",
    #         "Literature",
    #         "Religion & Spirituality",
    #         "Culture"
    #     ],
        
    #     "Social Systems & Governance": [
    #         "Politics",
    #         "Economics & Finance",
    #         "Government & Law",
    #         "Social Sciences",
    #         "News",
    #         "Military & Defense"
    #     ],
        
    #     "Health & Personal Development": [
    #         "Health & Medicine",
    #         "Sports & Exercise",
    #         "Psychology",
    #         "Personal Development",
    #         "Safety & Public Safety"
    #     ],
        
    #     "Arts & Expression": [
    #         "Arts",
    #         "Linguistics",
    #         "Entertainment",
    #         "Fashion & Beauty",
    #         "Hobbies & Crafts"
    #     ],
        
    #     "Practical Life & Career": [
    #         "Education & Knowledge",
    #         "Career & Business",
    #         "Food",
    #         "Lifestyle & Home",
    #         "Travel",
    #         "Geography",
    #         "Parenting & Children"
    #     ],
    #     "Others": []
    # }
    category_groups = {
        "Sciences & Formal Knowledge": [
            "Science",
            "Mathematics",
            "Technology & Engineering",
            "Environment & Nature",
            "Agriculture",
            "Geography"
        ],
        
        "Humanities & Cultural Studies": [
            "Philosophy",
            "History",
            "Literature",
            "Religion & Spirituality",
            "Culture",
            "Arts",
            "Linguistics"
        ],
        
        "Society & Governance": [
            "Politics",
            "Economics & Finance",
            "Government & Law",
            "Social Sciences",
            "News",
            "Military & Defense",
            "Safety & Public Safety"
        ],
        
        "Health & Personal Development": [
            "Health & Medicine",
            "Sports & Exercise",
            "Psychology",
            "Personal Development",
            "Parenting & Children"
        ],
        
        "Lifestyle, Career & Recreation": [
            "Education & Knowledge",
            "Career & Business",
            "Food",
            "Lifestyle & Home",
            "Travel",
            "Entertainment",
            "Fashion & Beauty",
            "Hobbies & Crafts"
        ]
    }
    print(f'dataset-{split}')

    # tasks = {
    #     "Academic & Theoretical Knowledge":0,
    #     "Society & Governance":0,
    #     "Health & Lifestyle":0,
    #     "Arts, Career & Leisure":0,
    #     "Others":0
    # }

    # tasks = {
    #     "Science & Mathematics":0,
    #     "Humanities & Philosophy":0,
    #     "Social Systems & Governance":0,
    #     "Health & Personal Development":0,
    #     "Arts & Expression":0,
    #     "Practical Life & Career":0,
    #     "Others":0
    # }

    tasks = {
        "Sciences & Formal Knowledge": [],
        "Humanities & Cultural Studies": [],
        "Society & Governance": [],
        "Health & Personal Development": [],
        "Lifestyle, Career & Recreation": [],
        "Others": []
    }

    for super_category, sub_categories in sorted(category_cnt.items()):
        # print(super_category)
        total_cnt = []
        for k, v in sub_categories.items():
            # print('\t', k, v)
            total_cnt.extend(v)
        # print('\t', total_cnt)
        
        for key in tasks.keys():
            if key == 'Others':
                tasks[key].extend(total_cnt)
            elif super_category in category_groups[key]:
                tasks[key].extend(total_cnt)
                break

    # print(tasks)
    return tasks

# splits = [0, 1, 2, 10, 11,51, 60, 61, 72, 80, 90, 91]
splits = [0,1,10,11,60,61,71,72,90,91,100,101]

lang_task = {}
language_per_topic = {}
for split in splits:
    tasks = topic_analyze(split)
    lang_task[split] = tasks
    
    for key in tasks.keys():
        if key not in language_per_topic.keys():
            language_per_topic[key] = {}
        if len(tasks[key]) >= 600:
            language_per_topic[key][split] = tasks[key]
print()
for key, value in language_per_topic.items():
    print(key)
    # print(value)
    for k ,v in value.items():
        print(k, len(v))

'''
0 arabic
1 hausa
2 somali
10 malay
11 malagasy
51 marathi
60 panjabi
61 nepali
72 portuguese
80 yoruba
90 chinese
91 vietnamese

71 spanish
100 kyrgyz
101 turkish
'''
random.seed(42)

output_folder = 'dataset/Fed-aya_topic'
train_folder = os.path.join(output_folder, 'train')
test_folder = os.path.join(output_folder, 'test')
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

for key, topic_datasets in language_per_topic.items():
    if key == "Sciences & Formal Knowledge":
        # client 1
        language_codes = [10,91,72,90]
        for i, code in enumerate(language_codes):
            datalist = topic_datasets[code]
            random.shuffle(datalist)
            test_set = datalist[:100]
            train_set = random.sample(datalist[100:],800) if len(datalist[100:]) > 800 else datalist[100:]
            with open(os.path.join(train_folder, f'dataset-{i}.json'),'w', encoding='utf-8') as fp:
                json.dump(train_set, fp, ensure_ascii=False, indent=4)
            with open(os.path.join(test_folder, f'dataset-{i}.json'),'w', encoding='utf-8') as fp:
                json.dump(test_set, fp, ensure_ascii=False, indent=4)
        
        # client 2
        language_codes = [61,0,100,60]
        for i, code in enumerate(language_codes):
            datalist = topic_datasets[code]
            random.shuffle(datalist)
            test_set = datalist[:100]
            train_set = random.sample(datalist[100:],800) if len(datalist[100:]) > 800 else datalist[100:]
            with open(os.path.join(train_folder, f'dataset-1{i}.json'),'w', encoding='utf-8') as fp:
                json.dump(train_set, fp, ensure_ascii=False, indent=4)
            with open(os.path.join(test_folder, f'dataset-1{i}.json'),'w', encoding='utf-8') as fp:
                json.dump(test_set, fp, ensure_ascii=False, indent=4)
    elif key == "Humanities & Cultural Studies":
        # client 3
        language_codes = [1,100,0,101]
        for i, code in enumerate(language_codes):
            datalist = topic_datasets[code]
            random.shuffle(datalist)
            test_set = datalist[:100]
            train_set = random.sample(datalist[100:],1200) if len(datalist[100:]) > 1200 else datalist[100:]
            with open(os.path.join(train_folder, f'dataset-2{i}.json'),'w', encoding='utf-8') as fp:
                json.dump(train_set, fp, ensure_ascii=False, indent=4)
            with open(os.path.join(test_folder, f'dataset-2{i}.json'),'w', encoding='utf-8') as fp:
                json.dump(test_set, fp, ensure_ascii=False, indent=4)
        # client 4
        language_codes = [11,91,10,90]
        for i, code in enumerate(language_codes):
            datalist = topic_datasets[code]
            random.shuffle(datalist)
            test_set = datalist[:100]
            train_set = random.sample(datalist[100:],1500) if len(datalist[100:]) > 1500 else datalist[100:]
            with open(os.path.join(train_folder, f'dataset-3{i}.json'),'w', encoding='utf-8') as fp:
                json.dump(train_set, fp, ensure_ascii=False, indent=4)
            with open(os.path.join(test_folder, f'dataset-3{i}.json'),'w', encoding='utf-8') as fp:
                json.dump(test_set, fp, ensure_ascii=False, indent=4)
        # client 5
        language_codes = [72,61,60,71]
        for i, code in enumerate(language_codes):
            datalist = topic_datasets[code]
            random.shuffle(datalist)
            test_set = datalist[:100]
            train_set = random.sample(datalist[100:],1500) if len(datalist[100:]) > 1500 else datalist[100:]
            with open(os.path.join(train_folder, f'dataset-4{i}.json'),'w', encoding='utf-8') as fp:
                json.dump(train_set, fp, ensure_ascii=False, indent=4)
            with open(os.path.join(test_folder, f'dataset-4{i}.json'),'w', encoding='utf-8') as fp:
                json.dump(test_set, fp, ensure_ascii=False, indent=4)
    
    elif key == "Society & Governance":
        # client 6
        language_codes = [91,60,11,10]
        for i, code in enumerate(language_codes):
            datalist = topic_datasets[code]
            random.shuffle(datalist)
            test_set = datalist[:100]
            train_set = random.sample(datalist[100:],1200) if len(datalist[100:]) > 1200 else datalist[100:]
            with open(os.path.join(train_folder, f'dataset-5{i}.json'),'w', encoding='utf-8') as fp:
                json.dump(train_set, fp, ensure_ascii=False, indent=4)
            with open(os.path.join(test_folder, f'dataset-5{i}.json'),'w', encoding='utf-8') as fp:
                json.dump(test_set, fp, ensure_ascii=False, indent=4)
        
    elif key == "Health & Personal Development":
        # client 7
        language_codes = [10,72,91]
        for i, code in enumerate(language_codes):
            datalist = topic_datasets[code]
            random.shuffle(datalist)
            test_set = datalist[:100]
            train_set = random.sample(datalist[100:],1000) if len(datalist[100:]) > 1000 else datalist[100:]
            with open(os.path.join(train_folder, f'dataset-6{i}.json'),'w', encoding='utf-8') as fp:
                json.dump(train_set, fp, ensure_ascii=False, indent=4)
            with open(os.path.join(test_folder, f'dataset-6{i}.json'),'w', encoding='utf-8') as fp:
                json.dump(test_set, fp, ensure_ascii=False, indent=4)
        
        # client 8
        language_codes = [100,101,11,0]
        for i, code in enumerate(language_codes):
            if code == 11: continue
            datalist = topic_datasets[code]
            random.shuffle(datalist)
            test_set = datalist[:100]
            train_set = random.sample(datalist[100:],800) if len(datalist[100:]) > 800 else datalist[100:]
            with open(os.path.join(train_folder, f'dataset-7{i}.json'),'w', encoding='utf-8') as fp:
                json.dump(train_set, fp, ensure_ascii=False, indent=4)
            with open(os.path.join(test_folder, f'dataset-7{i}.json'),'w', encoding='utf-8') as fp:
                json.dump(test_set, fp, ensure_ascii=False, indent=4)
        
        datalist = topic_datasets[11]
        
        random.shuffle(datalist)
        category_cnt = OrderedDict()

        for item in datalist:
            if item['super_category'] not in category_cnt.keys():
                category_cnt[item['super_category']] = [item]
            else:
                category_cnt[item['super_category']].append(item)
        
        subset_1 = category_cnt["Health & Medicine"] + category_cnt["Sports & Exercise"]
        subset_2 = category_cnt["Personal Development"] + category_cnt["Parenting & Children"] + category_cnt["Psychology"]
        
        random.shuffle(subset_1)
        test_set = subset_1[:100]
        train_set = random.sample(subset_1[100:],1000) if len(subset_1[100:]) > 1000 else subset_1[100:]
        with open(os.path.join(train_folder, f'dataset-63.json'),'w', encoding='utf-8') as fp:
            json.dump(train_set, fp, ensure_ascii=False, indent=4)
        with open(os.path.join(test_folder, f'dataset-63.json'),'w', encoding='utf-8') as fp:
            json.dump(test_set, fp, ensure_ascii=False, indent=4)
        
        random.shuffle(subset_2)
        test_set = subset_2[:100]
        train_set = random.sample(subset_2[100:],800) if len(subset_2[100:]) > 800 else subset_2[100:]
        with open(os.path.join(train_folder, f'dataset-72.json'),'w', encoding='utf-8') as fp:
            json.dump(train_set, fp, ensure_ascii=False, indent=4)
        with open(os.path.join(test_folder, f'dataset-72.json'),'w', encoding='utf-8') as fp:
            json.dump(test_set, fp, ensure_ascii=False, indent=4)