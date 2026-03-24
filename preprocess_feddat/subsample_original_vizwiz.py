import json
import random
random.seed(42)

datalist = json.load(open('dataset/VizWiz-Original/train/dataset-0_L.json','r'))
subsample = random.sample(datalist, 5000)
with open('dataset/VizWiz-Original/train/dataset-0.json','w') as fp:
    json.dump(subsample, fp, indent=4)

datalist = json.load(open('dataset/VizWiz-Original/test/dataset-0_L.json','r'))
subsample = random.sample(datalist, 1000)
with open('dataset/VizWiz-Original/test/dataset-0.json','w') as fp:
    json.dump(subsample, fp, indent=4)