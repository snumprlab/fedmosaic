import json
import random
import shutil
random.seed(42)

# NEW SPLIT 3 (scenario 110~)
# client1 #######################################################################################
shutil.copyfile('dataset/Fashion200K/train/dataset-0.json', 'dataset/Fashion200K/train/dataset-40.json')
shutil.copyfile('dataset/Fashion200K/test/dataset-0.json', 'dataset/Fashion200K/test/dataset-40.json')

original_data = json.load(open('dataset/MagicBrush/train/dataset-1.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/MagicBrush/train/dataset-41.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
shutil.copyfile('dataset/MagicBrush/test/dataset-1.json', 'dataset/MagicBrush/test/dataset-41.json')

shutil.copyfile('dataset/VISION/train/dataset-0.json', 'dataset/VISION/train/dataset-40.json')
shutil.copyfile('dataset/VISION/test/dataset-0.json', 'dataset/VISION/test/dataset-40.json')

# client2 #######################################################################################
shutil.copyfile('dataset/VISION/train/dataset-1.json', 'dataset/VISION/train/dataset-41.json')
shutil.copyfile('dataset/VISION/test/dataset-1.json', 'dataset/VISION/test/dataset-41.json')

shutil.copyfile('dataset/Fashion200K/train/dataset-3.json', 'dataset/Fashion200K/train/dataset-43.json')
shutil.copyfile('dataset/Fashion200K/test/dataset-3.json', 'dataset/Fashion200K/test/dataset-43.json')

original_data = json.load(open('dataset/NLVR2/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/NLVR2/train/dataset-40.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/NLVR2/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/NLVR2/test/dataset-40.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

# client3 #######################################################################################
original_data = json.load(open('dataset/HQ_Edit/train/dataset-0.json','r'))
sample_per_instr = int(len(original_data)/10)
new_data = []
for idx in range(10):
    subset = original_data[idx*sample_per_instr:(idx+1)*sample_per_instr]
    size = len(subset)//2
    sampled_idx = random.sample(range(size), 200)
    for sid in sampled_idx:
        new_data += subset[sid*2:sid*2+2]
with open('dataset/HQ_Edit/train/dataset-40.json','w') as fp:
    json.dump(new_data, fp, indent=4)
original_data = json.load(open('dataset/HQ_Edit/test/dataset-0.json','r'))
sample_per_instr = int(len(original_data)/10)
new_data = []
for idx in range(10):
    subset = original_data[idx*sample_per_instr:(idx+1)*sample_per_instr]
    size = len(subset)//2
    sampled_idx = random.sample(range(size), 50)
    for sid in sampled_idx:
        new_data += subset[sid*2:sid*2+2]
with open('dataset/HQ_Edit/test/dataset-40.json','w') as fp:
    json.dump(new_data, fp, indent=4)

shutil.copyfile('dataset/Fashion200K/train/dataset-2.json', 'dataset/Fashion200K/train/dataset-42.json')
shutil.copyfile('dataset/Fashion200K/test/dataset-2.json', 'dataset/Fashion200K/test/dataset-42.json')

original_data = json.load(open('dataset/NLVR2/train/dataset-1.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/NLVR2/train/dataset-41.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/NLVR2/test/dataset-1.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/NLVR2/test/dataset-41.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/CIRR/train/dataset-1.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/CIRR/train/dataset-41.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/CIRR/test/dataset-1.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/CIRR/test/dataset-41.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

# client 4 #######################################################################################
shutil.copyfile('dataset/VizWiz/train/dataset-0.json', 'dataset/VizWiz/train/dataset-40.json')
shutil.copyfile('dataset/VizWiz/test/dataset-0.json', 'dataset/VizWiz/test/dataset-40.json')

shutil.copyfile('dataset/IRFL/train/dataset-3.json', 'dataset/IRFL/train/dataset-43.json')
shutil.copyfile('dataset/IRFL/test/dataset-3.json', 'dataset/IRFL/test/dataset-43.json')

shutil.copyfile('dataset/MIT-States/train/dataset-0.json', 'dataset/MIT-States/train/dataset-40.json')
shutil.copyfile('dataset/MIT-States/test/dataset-0.json', 'dataset/MIT-States/test/dataset-40.json')

# client 5 ######################################################################################
original_data = json.load(open('dataset/IRFL/train/dataset-1.json','r'))
sampled_data = random.sample(original_data, 5000)
with open('dataset/IRFL/train/dataset-41.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/IRFL/test/dataset-1.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/IRFL/test/dataset-41.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/Bongard-OpenWorld/train/dataset-3.json','r'))
sampled_data = random.sample(original_data, 5000)
with open('dataset/Bongard-OpenWorld/train/dataset-43.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/Bongard-OpenWorld/test/dataset-3.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/Bongard-OpenWorld/test/dataset-43.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/COMICS_Dialogue/train/dataset-0.json', 'dataset/COMICS_Dialogue/train/dataset-40.json')
shutil.copyfile('dataset/COMICS_Dialogue/test/dataset-0.json', 'dataset/COMICS_Dialogue/test/dataset-40.json')

original_data = json.load(open('dataset/IRFL/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 5000)
with open('dataset/IRFL/train/dataset-40.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/IRFL/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/IRFL/test/dataset-40.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

# client 6 #######################################################################################
shutil.copyfile('dataset/Bongard-HOI/train/dataset-2.json', 'dataset/Bongard-HOI/train/dataset-42.json')
original_data = json.load(open('dataset/Bongard-HOI/test/dataset-2.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/Bongard-HOI/test/dataset-42.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/Bongard-OpenWorld/train/dataset-1.json', 'dataset/Bongard-OpenWorld/train/dataset-41.json')
original_data = json.load(open('dataset/Bongard-OpenWorld/test/dataset-1.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/Bongard-OpenWorld/test/dataset-41.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/IRFL/train/dataset-2.json', 'dataset/IRFL/train/dataset-42.json')
original_data = json.load(open('dataset/IRFL/test/dataset-2.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/IRFL/test/dataset-42.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

# client 7 #######################################################################################
original_data = json.load(open('dataset/HRVQA/train/dataset-9.json','r'))
sampled_data = random.sample(original_data, 5000)
with open('dataset/HRVQA/train/dataset-49.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/HRVQA/test/dataset-9.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/HRVQA/test/dataset-49.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/iconqa/train/dataset-2.json', 'dataset/iconqa/train/dataset-42.json')
original_data = json.load(open('dataset/iconqa/test/dataset-2.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/iconqa/test/dataset-42.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

# client 8 #######################################################################################
shutil.copyfile('dataset/dvqa/train/dataset-0.json', 'dataset/dvqa/train/dataset-40.json')
original_data = json.load(open('dataset/dvqa/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/dvqa/test/dataset-40.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
    
shutil.copyfile('dataset/HRVQA/train/dataset-2.json', 'dataset/HRVQA/train/dataset-42.json')
original_data = json.load(open('dataset/HRVQA/test/dataset-2.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/HRVQA/test/dataset-42.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
    
shutil.copyfile('dataset/iconqa/train/dataset-1.json', 'dataset/iconqa/train/dataset-41.json')
original_data = json.load(open('dataset/iconqa/test/dataset-1.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/iconqa/test/dataset-41.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
# client 9 #######################################################################################
shutil.copyfile('dataset/TQA/train/dataset-0.json', 'dataset/TQA/train/dataset-40.json')
original_data = json.load(open('dataset/TQA/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/TQA/test/dataset-40.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
    
original_data = json.load(open('dataset/HRVQA/train/dataset-7.json','r'))
sampled_data = random.sample(original_data, 7000)
with open('dataset/HRVQA/train/dataset-47.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/HRVQA/test/dataset-7.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/HRVQA/test/dataset-47.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/iconqa/train/dataset-3.json', 'dataset/iconqa/train/dataset-43.json')
original_data = json.load(open('dataset/iconqa/test/dataset-3.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/iconqa/test/dataset-43.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

# client 10 #######################################################################################
shutil.copyfile('dataset/iconqa/train/dataset-0.json', 'dataset/iconqa/train/dataset-40.json')
original_data = json.load(open('dataset/iconqa/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/iconqa/test/dataset-40.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/WebQA/train/dataset-0.json', 'dataset/WebQA/train/dataset-40.json')
original_data = json.load(open('dataset/WebQA/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/WebQA/test/dataset-40.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/HRVQA/train/dataset-5.json','r'))
sampled_data = random.sample(original_data, 8000)
with open('dataset/HRVQA/train/dataset-45.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/HRVQA/test/dataset-5.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/HRVQA/test/dataset-45.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
    
    

original_data = json.load(open('dataset/WCVQA/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 5000)
with open('dataset/WCVQA/train/dataset-40.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/WCVQA/train/dataset-1.json','r'))
sampled_data = random.sample(original_data, 7000)
with open('dataset/WCVQA/train/dataset-41.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/WCVQA/train/dataset-2.json','r'))
sampled_data = random.sample(original_data, 8000)
with open('dataset/WCVQA/train/dataset-42.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/WCVQA/test/dataset-0.json', 'dataset/WCVQA/test/dataset-40.json')
shutil.copyfile('dataset/WCVQA/test/dataset-1.json', 'dataset/WCVQA/test/dataset-41.json')
shutil.copyfile('dataset/WCVQA/test/dataset-2.json', 'dataset/WCVQA/test/dataset-42.json')
shutil.copyfile('dataset/WCVQA/train/dataset-3.json', 'dataset/WCVQA/train/dataset-43.json')
shutil.copyfile('dataset/WCVQA/test/dataset-3.json', 'dataset/WCVQA/test/dataset-43.json')