import json
import random
import shutil
random.seed(42)

# NEW SPLIT 3 (scenario 90~)
# client1 #######################################################################################
shutil.copyfile('dataset/Fashion200K/train/dataset-0.json', 'dataset/Fashion200K/train/dataset-30.json')
shutil.copyfile('dataset/Fashion200K/test/dataset-0.json', 'dataset/Fashion200K/test/dataset-30.json')

original_data = json.load(open('dataset/Bongard-OpenWorld/train/dataset-1.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/Bongard-OpenWorld/train/dataset-31.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/Bongard-OpenWorld/test/dataset-1.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/Bongard-OpenWorld/test/dataset-31.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
    
shutil.copyfile('dataset/VizWiz/train/dataset-0.json', 'dataset/VizWiz/train/dataset-30.json')
shutil.copyfile('dataset/VizWiz/test/dataset-0.json', 'dataset/VizWiz/test/dataset-30.json')

# client2 #######################################################################################
original_data = json.load(open('dataset/HRVQA/train/dataset-2.json','r'))
sampled_data = random.sample(original_data, 5000)
with open('dataset/HRVQA/train/dataset-32.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/HRVQA/test/dataset-2.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/HRVQA/test/dataset-32.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/TQA/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 5000)
with open('dataset/TQA/train/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/TQA/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/TQA/test/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/iconqa/train/dataset-2.json', 'dataset/iconqa/train/dataset-32.json')
original_data = json.load(open('dataset/iconqa/test/dataset-2.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/iconqa/test/dataset-32.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)


# client3 #######################################################################################
shutil.copyfile('dataset/NLVR2/train/dataset-0.json', 'dataset/NLVR2/train/dataset-30.json')
original_data = json.load(open('dataset/NLVR2/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/NLVR2/test/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/VIST/train/dataset-0.json', 'dataset/VIST/train/dataset-30.json')
original_data = json.load(open('dataset/VIST/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/VIST/test/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/Birds-to-Words/train/dataset-0.json', 'dataset/Birds-to-Words/train/dataset-30.json')
original_data = json.load(open('dataset/Birds-to-Words/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/Birds-to-Words/test/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/HQ_Edit/train/dataset-0.json', 'dataset/HQ_Edit/train/dataset-30.json')
original_data = json.load(open('dataset/HQ_Edit/test/dataset-0.json','r'))
sample_per_instr = int(len(original_data)/10)
new_data = []
for idx in range(10):
    subset = original_data[idx*sample_per_instr:(idx+1)*sample_per_instr]
    size = len(subset)//2
    sampled_idx = random.sample(range(size), 50)
    for sid in sampled_idx:
        new_data += subset[sid*2:sid*2+2]
with open('dataset/HQ_Edit/test/dataset-30.json','w') as fp:
    json.dump(new_data, fp, indent=4)

# client 4 #######################################################################################
original_data = json.load(open('dataset/WebQA/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 7000)
with open('dataset/WebQA/train/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/WebQA/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/WebQA/test/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/HRVQA/train/dataset-5.json','r'))
sampled_data = random.sample(original_data, 7000)
with open('dataset/HRVQA/train/dataset-35.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/HRVQA/test/dataset-5.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/HRVQA/test/dataset-35.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/iconqa/train/dataset-3.json', 'dataset/iconqa/train/dataset-33.json')
original_data = json.load(open('dataset/iconqa/test/dataset-3.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/iconqa/test/dataset-33.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

# client 5 ######################################################################################
original_data = json.load(open('dataset/Bongard-OpenWorld/train/dataset-3.json','r'))
sampled_data = random.sample(original_data, 5000)
with open('dataset/Bongard-OpenWorld/train/dataset-33.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/Bongard-OpenWorld/test/dataset-3.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/Bongard-OpenWorld/test/dataset-33.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/Fashion200K/train/dataset-2.json', 'dataset/Fashion200K/train/dataset-32.json')
shutil.copyfile('dataset/Fashion200K/test/dataset-2.json', 'dataset/Fashion200K/test/dataset-32.json')

shutil.copyfile('dataset/VISION/train/dataset-0.json', 'dataset/VISION/train/dataset-30.json')
shutil.copyfile('dataset/VISION/test/dataset-0.json', 'dataset/VISION/test/dataset-30.json')

# client 6 #######################################################################################
shutil.copyfile('dataset/iconqa/train/dataset-0.json', 'dataset/iconqa/train/dataset-30.json')
original_data = json.load(open('dataset/iconqa/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/iconqa/test/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
    
original_data = json.load(open('dataset/HRVQA/train/dataset-7.json','r'))
sampled_data = random.sample(original_data, 8000)
with open('dataset/HRVQA/train/dataset-37.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/HRVQA/test/dataset-7.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/HRVQA/test/dataset-37.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
    
original_data = json.load(open('dataset/dvqa/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 8000)
with open('dataset/dvqa/train/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/dvqa/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/dvqa/test/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

# client 7 #######################################################################################
shutil.copyfile('dataset/iconqa/train/dataset-1.json', 'dataset/iconqa/train/dataset-31.json')
original_data = json.load(open('dataset/iconqa/test/dataset-1.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/iconqa/test/dataset-31.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/HRVQA/train/dataset-9.json', 'dataset/HRVQA/train/dataset-39.json')
original_data = json.load(open('dataset/HRVQA/test/dataset-9.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/HRVQA/test/dataset-39.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

# client 8 #######################################################################################
original_data = json.load(open('dataset/Spot-the-Diff/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/Spot-the-Diff/train/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/Spot-the-Diff/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/Spot-the-Diff/test/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/CIRR/train/dataset-1.json','r'))
# sampled_data = random.sample(original_data, 4000)
sampled_data = random.sample(original_data, 8000)
with open('dataset/CIRR/train/dataset-31.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/CIRR/test/dataset-1.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/CIRR/test/dataset-31.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/FlintstonesSV/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/FlintstonesSV/train/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/FlintstonesSV/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/FlintstonesSV/test/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/IEdit/train/dataset-0.json', 'dataset/IEdit/train/dataset-30.json')
shutil.copyfile('dataset/IEdit/test/dataset-0.json', 'dataset/IEdit/test/dataset-30.json')
# client 9 #######################################################################################
original_data = json.load(open('dataset/PororoSV/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/PororoSV/train/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/PororoSV/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/PororoSV/test/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/CLEVR-Change/train/dataset-0.json', 'dataset/CLEVR-Change/train/dataset-30.json')
shutil.copyfile('dataset/CLEVR-Change/test/dataset-0.json', 'dataset/CLEVR-Change/test/dataset-30.json')

original_data = json.load(open('dataset/AESOP/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/AESOP/train/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/AESOP/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/AESOP/test/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
# client 10 #######################################################################################
shutil.copyfile('dataset/Fashion200K/train/dataset-3.json', 'dataset/Fashion200K/train/dataset-33.json')
shutil.copyfile('dataset/Fashion200K/test/dataset-3.json', 'dataset/Fashion200K/test/dataset-33.json')

shutil.copyfile('dataset/MIT-States/train/dataset-0.json', 'dataset/MIT-States/train/dataset-30.json')
shutil.copyfile('dataset/MIT-States/test/dataset-0.json', 'dataset/MIT-States/test/dataset-30.json')

shutil.copyfile('dataset/VISION/train/dataset-1.json', 'dataset/VISION/train/dataset-31.json')
shutil.copyfile('dataset/VISION/test/dataset-1.json', 'dataset/VISION/test/dataset-31.json')



# sc100

shutil.copyfile('dataset/IRFL/train/dataset-1.json', 'dataset/IRFL/train/dataset-31.json')
original_data = json.load(open('dataset/IRFL/test/dataset-1.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/IRFL/test/dataset-31.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/IRFL/train/dataset-2.json','r'))
sampled_data = random.sample(original_data, 8000)
with open('dataset/IRFL/train/dataset-32.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/IRFL/test/dataset-2.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/IRFL/test/dataset-32.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/Bongard-HOI/train/dataset-2.json', 'dataset/Bongard-HOI/train/dataset-32.json')
original_data = json.load(open('dataset/Bongard-HOI/test/dataset-2.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/Bongard-HOI/test/dataset-32.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

shutil.copyfile('dataset/IRFL/train/dataset-3.json', 'dataset/IRFL/train/dataset-33.json')
shutil.copyfile('dataset/IRFL/test/dataset-3.json', 'dataset/IRFL/test/dataset-33.json')

shutil.copyfile('dataset/IRFL/train/dataset-0.json', 'dataset/IRFL/train/dataset-30.json')
original_data = json.load(open('dataset/IRFL/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/IRFL/test/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/NLVR2/train/dataset-1.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/NLVR2/train/dataset-31.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/NLVR2/test/dataset-1.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/NLVR2/test/dataset-31.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/COMICS_Dialogue/train/dataset-0.json','r'))
sampled_data = random.sample(original_data, 4000)
with open('dataset/COMICS_Dialogue/train/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
original_data = json.load(open('dataset/COMICS_Dialogue/test/dataset-0.json','r'))
sampled_data = random.sample(original_data, 1000)
with open('dataset/COMICS_Dialogue/test/dataset-30.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)

original_data = json.load(open('dataset/MagicBrush/train/dataset-1.json','r'))
sampled_data = random.sample(original_data, 8000)
with open('dataset/MagicBrush/train/dataset-31.json','w') as fp:
    json.dump(sampled_data, fp, indent=4)
shutil.copyfile('dataset/MagicBrush/test/dataset-1.json', 'dataset/MagicBrush/test/dataset-31.json')
