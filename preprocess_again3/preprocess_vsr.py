import json
import random
import shutil
random.seed(42)

shutil.copyfile('dataset/SEED-Bench-2/train/dataset-0.json', 'dataset/SEED-Bench-2/train/dataset-40.json')
# shutil.copyfile('dataset/SEED-Bench-2/test/dataset-0.json', 'dataset/SEED-Bench-2/test/dataset-40.json')
original_data = json.load(open('dataset/SEED-Bench-2/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/SEED-Bench-2/test/dataset-40.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/SEED-Bench-2/train/dataset-1.json', 'dataset/SEED-Bench-2/train/dataset-41.json')
shutil.copyfile('dataset/SEED-Bench-2/test/dataset-1.json', 'dataset/SEED-Bench-2/test/dataset-41.json')

original_data = json.load(open('dataset/SEED-Bench-2/train/dataset-2.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/SEED-Bench-2/train/dataset-42.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/SEED-Bench-2/test/dataset-2.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/SEED-Bench-2/test/dataset-42.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/VSR/train/dataset-0.json', 'dataset/VSR/train/dataset-40.json')
shutil.copyfile('dataset/VSR/test/dataset-0.json', 'dataset/VSR/test/dataset-40.json')