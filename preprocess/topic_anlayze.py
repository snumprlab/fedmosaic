import json
import os
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
            category_cnt[item['super_category']][item['sub_category']] = 1
        else:
            category_cnt[item['super_category']][item['sub_category']] += 1
        
        
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
        "Sciences & Formal Knowledge": 0,
        "Humanities & Cultural Studies": 0,
        "Society & Governance": 0,
        "Health & Personal Development": 0,
        "Lifestyle, Career & Recreation": 0,
        "Others": 0
    }

    for super_category, sub_categories in sorted(category_cnt.items()):
        # print(super_category)
        total_cnt = 0
        for k, v in sub_categories.items():
            # print('\t', k, v)
            total_cnt += v
        # print('\t', total_cnt)
        
        for key in tasks.keys():
            if key == 'Others':
                tasks[key] += total_cnt
            elif super_category in category_groups[key]:
                tasks[key] += total_cnt
                break

    print(tasks)
    return tasks

splits = [0, 1, 10, 11, 60, 61, 71, 72, 90, 91, 100, 101]

lang_task = {}
language_per_topic = {}
for split in splits:
    tasks = topic_analyze(split)
    lang_task[split] = tasks
    
    for key in tasks.keys():
        if key not in language_per_topic.keys():
            language_per_topic[key] = []
        
        if tasks[key] >= 600:
            language_per_topic[key].append({split:tasks[key]})
print()
for key, value in language_per_topic.items():
    print(key)
    print(value)