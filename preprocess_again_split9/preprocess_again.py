import json
import random
import shutil
random.seed(42)

# client2 ######################################################################################
shutil.copyfile('dataset/DocVQA/test/dataset-3.json', 'dataset/DocVQA/test/dataset-53.json')
original_data = json.load(open('dataset/DocVQA/train/dataset-3.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/DocVQA/train/dataset-53.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)


# client 9 #######################################################################################
shutil.copyfile('dataset/VSR/train/dataset-0.json', 'dataset/VSR/train/dataset-50.json')
shutil.copyfile('dataset/VSR/test/dataset-0.json', 'dataset/VSR/test/dataset-50.json')

original_data = json.load(open('dataset/iconqa/train/dataset-3.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/iconqa/train/dataset-53.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/iconqa/test/dataset-3.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/iconqa/test/dataset-53.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/DocVQA/test/dataset-2.json', 'dataset/DocVQA/test/dataset-52.json')
original_data = json.load(open('dataset/DocVQA/train/dataset-2.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/DocVQA/train/dataset-52.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

# client 10 #######################################################################################

shutil.copyfile('dataset/DocVQA/test/dataset-0.json', 'dataset/DocVQA/test/dataset-50.json')
original_data = json.load(open('dataset/DocVQA/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 8000)
with open('dataset/DocVQA/train/dataset-50.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
