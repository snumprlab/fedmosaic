import jsonlines
import json
import csv
from collections import defaultdict
import os
import numpy as np
import re

mode = 'feddualMultipqfullfreeze_homoAgg_moe'
Method= 'feddualMultipqfullfreeze_homoAgg_moe_NOCONT_bs4_saveoptim_lr3e-4_sc202_4tasks_5rounds_fixitr29_T0125_decay099'

num_rounds = [5,10,15,20]
is_client = True

scenario_num = 202
with open(f"./scenarios/scenario-{scenario_num}.json") as fp:
    scenario = json.load(fp)
scores = defaultdict(list)
client_scores = defaultdict(lambda: defaultdict(list))
results = []
csv_datas = []

prefix = f'eval_results/{mode}/{Method}/'
for num_round in num_rounds:
    csv_data = [Method, num_round]
    for client_id in range(5):
        # 1. mmlu
        csv_data.append(json.load(open(prefix+f'client{client_id}_round{num_round}_mmlu.json', 'r'))[-1]['accuracy'])
        # 2. math
        csv_data.append(json.load(open(prefix+f'client{client_id}_round{num_round}_math.json', 'r'))[-1]['accuracy'])
        # 3. code
        with open(prefix+f'client{client_id}_round{num_round}_humaneval_answer.jsonl_score.txt', 'r') as file:
            content = file.read()
        # Regular expression to find the pass@1 value
        match = re.search(r"'pass@1': (\d+\.\d+)", content)
        if match:
            pass_at_1 = float(match.group(1))
        csv_data.append(pass_at_1)
    csv_datas.append(csv_data)

# Prepare header
header = ['method', 'round', 'mmlu','math','code','mmlu','math','code','mmlu','math','code','mmlu','math','code','mmlu','math','code',]

# Write to CSV
csv_file = f'FSLLM_{Method}.csv'
file_exists = os.path.isfile(csv_file)

with open(csv_file, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(header)
    for csv_data in csv_datas:
        writer.writerow(csv_data)

print(f"Method: {Method}")
# print(f"AUC Score: {np.mean(avg_scores)}")
# print(f"Last Score: {avg_scores[-1]}")
print(f"Results have been appended to {csv_file}")
