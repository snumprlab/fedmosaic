import json
import random
import shutil
random.seed(42)

#######################################################################################
original_data = json.load(open('dataset/Bongard-OpenWorld/train/dataset-3.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/Bongard-OpenWorld/train/dataset-13.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/Bongard-OpenWorld/test/dataset-3.json', 'dataset/Bongard-OpenWorld/test/dataset-13.json')
#######################################################################################
original_data = json.load(open('dataset/HRVQA/train/dataset-7.json','r'))
sampled_data = random.sample(original_data, 8000)
with open('dataset/HRVQA/train/dataset-17.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/HRVQA/test/dataset-7.json', 'dataset/HRVQA/test/dataset-17.json')
#######################################################################################
original_data = json.load(open('dataset/HRVQA/train/dataset-5.json','r'))
sampled_data = random.sample(original_data, 8000)
with open('dataset/HRVQA/train/dataset-15.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
sampled_data = random.sample(original_data, 5000)
with open('dataset/HRVQA/train/dataset-25.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/HRVQA/test/dataset-5.json', 'dataset/HRVQA/test/dataset-15.json')
shutil.copyfile('dataset/HRVQA/test/dataset-5.json', 'dataset/HRVQA/test/dataset-25.json')
#######################################################################################
original_data = json.load(open('dataset/HRVQA/train/dataset-9.json','r'))
sampled_data = random.sample(original_data, 8000)
with open('dataset/HRVQA/train/dataset-19.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
sampled_data = random.sample(original_data, 7000)
with open('dataset/HRVQA/train/dataset-29.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/HRVQA/test/dataset-9.json', 'dataset/HRVQA/test/dataset-19.json')
shutil.copyfile('dataset/HRVQA/test/dataset-9.json', 'dataset/HRVQA/test/dataset-29.json')
#######################################################################################
original_data = json.load(open('dataset/HRVQA/train/dataset-2.json','r'))
sampled_data = random.sample(original_data, 7000)
with open('dataset/HRVQA/train/dataset-22.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/HRVQA/test/dataset-2.json', 'dataset/HRVQA/test/dataset-22.json')
#######################################################################################
original_data = json.load(open('dataset/PororoSV/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 3000)
with open('dataset/PororoSV/train/dataset-10.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
sampled_data = random.sample(original_data, 4000)
with open('dataset/PororoSV/train/dataset-20.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/PororoSV/test/dataset-0.json', 'dataset/PororoSV/test/dataset-10.json')
shutil.copyfile('dataset/PororoSV/test/dataset-0.json', 'dataset/PororoSV/test/dataset-20.json')
#######################################################################################
original_data = json.load(open('dataset/dvqa/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 8000)
with open('dataset/dvqa/train/dataset-10.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/dvqa/test/dataset-0.json', 'dataset/dvqa/test/dataset-10.json')
#######################################################################################
original_data = json.load(open('dataset/VIST/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/VIST/train/dataset-10.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/VIST/test/dataset-0.json', 'dataset/VIST/test/dataset-10.json')
#######################################################################################
original_data = json.load(open('dataset/AESOP/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/AESOP/train/dataset-10.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/AESOP/test/dataset-0.json', 'dataset/AESOP/test/dataset-10.json')
#######################################################################################
original_data = json.load(open('dataset/NLVR2/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/NLVR2/train/dataset-10.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
sampled_data = random.sample(original_data, 3000)
with open('dataset/NLVR2/train/dataset-20.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/NLVR2/test/dataset-0.json', 'dataset/NLVR2/test/dataset-10.json')
shutil.copyfile('dataset/NLVR2/test/dataset-0.json', 'dataset/NLVR2/test/dataset-20.json')
#######################################################################################
original_data = json.load(open('dataset/FlintstonesSV/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/FlintstonesSV/train/dataset-10.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
sampled_data = random.sample(original_data, 3000)
with open('dataset/FlintstonesSV/train/dataset-20.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/FlintstonesSV/test/dataset-0.json', 'dataset/FlintstonesSV/test/dataset-10.json')
shutil.copyfile('dataset/FlintstonesSV/test/dataset-0.json', 'dataset/FlintstonesSV/test/dataset-20.json')
#######################################################################################
original_data = json.load(open('dataset/Bongard-OpenWorld/train/dataset-1.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/Bongard-OpenWorld/train/dataset-11.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/Bongard-OpenWorld/test/dataset-1.json', 'dataset/Bongard-OpenWorld/test/dataset-11.json')
#######################################################################################
original_data = json.load(open('dataset/iconqa/train/dataset-1.json','r'))
sampled_data = random.sample(original_data, 5000)
with open('dataset/iconqa/train/dataset-21.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/iconqa/test/dataset-1.json', 'dataset/iconqa/test/dataset-21.json')
#######################################################################################
original_data = json.load(open('dataset/CIRR/train/dataset-1.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/CIRR/train/dataset-11.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
sampled_data = random.sample(original_data, 3000)
with open('dataset/CIRR/train/dataset-21.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
shutil.copyfile('dataset/CIRR/test/dataset-1.json', 'dataset/CIRR/test/dataset-11.json')
shutil.copyfile('dataset/CIRR/test/dataset-1.json', 'dataset/CIRR/test/dataset-21.json')

#######################################################################################
original_data = json.load(open('dataset/HQ_Edit/train/dataset-0.json','r'))
sample_per_instr = int(len(original_data)/10)
new_data = []
for idx in range(10):
    subset = original_data[idx*sample_per_instr:(idx+1)*sample_per_instr]
    size = len(subset)//2
    sampled_idx = random.sample(range(size), 150)
    for sid in sampled_idx:
        new_data += subset[sid*2:sid*2+2]
with open('dataset/HQ_Edit/train/dataset-10.json','w') as fp:
    json.dump(new_data, fp, indent=4)

new_data = []
for idx in range(10):
    subset = original_data[idx*sample_per_instr:(idx+1)*sample_per_instr]
    size = len(subset)//2
    sampled_idx = random.sample(range(size), 300)
    for sid in sampled_idx:
        new_data += subset[sid*2:sid*2+2]
with open('dataset/HQ_Edit/train/dataset-20.json','w') as fp:
    json.dump(new_data, fp, indent=4)

shutil.copyfile('dataset/HQ_Edit/test/dataset-0.json', 'dataset/HQ_Edit/test/dataset-10.json')
shutil.copyfile('dataset/HQ_Edit/test/dataset-0.json', 'dataset/HQ_Edit/test/dataset-20.json')

