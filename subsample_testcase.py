import json
import random

random.seed(42)

dataset = "CIRR"
subset_id=1

dataset_path = f"./dataset/{dataset}/test/dataset-{subset_id}.json"
output_path = f"./dataset/{dataset}/test/dataset-{subset_id}-small.json"

with open(dataset_path, 'r') as fp:
    datalist = json.load(fp)

print(f'original length: {len(datalist)}')

new_datalist = random.sample(datalist, 500)

print(f'new length: {len(new_datalist)}')

with open(output_path, 'w') as fp:
    json.dump(new_datalist, fp, indent=4)