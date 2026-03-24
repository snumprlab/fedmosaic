import json
import random
import shutil
random.seed(42)

# client2 ######################################################################################
shutil.copyfile('dataset/SNLI-VE/test/dataset-0.json', 'dataset/SNLI-VE/test/dataset-50.json')
shutil.copyfile('dataset/SNLI-VE/test/dataset-1.json', 'dataset/SNLI-VE/test/dataset-51.json')
shutil.copyfile('dataset/SNLI-VE/test/dataset-2.json', 'dataset/SNLI-VE/test/dataset-52.json')
shutil.copyfile('dataset/SNLI-VE/test/dataset-3.json', 'dataset/SNLI-VE/test/dataset-53.json')

original_data = json.load(open('dataset/SNLI-VE/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/SNLI-VE/train/dataset-50.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/SNLI-VE/train/dataset-3.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/SNLI-VE/train/dataset-53.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/SNLI-VE/train/dataset-2.json','r'))
sampled_data = random.sample(original_data, 7000)
with open('dataset/SNLI-VE/train/dataset-52.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/SNLI-VE/train/dataset-1.json', 'dataset/SNLI-VE/train/dataset-51.json')

original_data = json.load(open('dataset/IRFL/train/dataset-1.json','r'))
sampled_data = random.sample(original_data, 7000)
with open('dataset/IRFL/train/dataset-51.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/IRFL/test/dataset-1.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/IRFL/test/dataset-51.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/IRFL/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 7000)
with open('dataset/IRFL/train/dataset-50.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/IRFL/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/IRFL/test/dataset-50.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/Bongard-OpenWorld/train/dataset-3.json','r'))
sampled_data = random.sample(original_data, 7000)
with open('dataset/Bongard-OpenWorld/train/dataset-53.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/Bongard-OpenWorld/test/dataset-3.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/Bongard-OpenWorld/test/dataset-53.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
