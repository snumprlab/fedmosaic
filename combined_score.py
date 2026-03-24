import jsonlines
import json
import csv
from collections import defaultdict
import os
import numpy as np

client_num = 21
mode = 'fedavg'
Method= 'feddat_fedavg_iter100_round5_homo_1B'
#'feddat_sft_iter50_round5_homo_3B'

num_rounds = [2, 5, 7, 10, 12, 15, 17, 20]
is_client = True

scenario_num = 301
with open(f"./scenarios/scenario-{scenario_num}.json") as fp:
    scenario = json.load(fp)
scores = defaultdict(list)
client_scores = defaultdict(lambda: defaultdict(list))
results = []

for client_data in scenario:

    id = client_data['client_id']
    for data in client_data['datasets']:
        data_name = f"{data['dataset']}-{data['subset_id']}"
        if is_client:
            for num_round in num_rounds:
                with open(f'./eval_results/{mode}/{Method}/client{id}_round{num_round}_{data_name}.json', 'r') as fp:
                    results.append(json.load(fp)[-1])
            
        else:
            for num_round in num_rounds:
                with open(f'./eval_results/{mode}/{Method}/server_round{num_rounds}_{data_name}.json', 'r') as fp:
                    results.append(json.load(fp)[-1])
        
        for r, result in zip(num_rounds, results):
            # print(data_name, data['type'], r, len(results))
            if data['type'] == 'multi-choice':
                score = result['accuracy']
            elif data['type'] == 'open-ended':
                if data['metric'] == 'F1':
                    score = 2*(result['precision']*result['recall']) / (result['precision'] + result['recall'])
                elif data['metric'] == 'RougeL':
                    score = result['ROUGE_L'][0]
        
            scores[data_name].append(score)
            client_scores[r][id].append(score)
        results = []

scores_per_round = np.array(list(scores.values()))
avg_scores = [sum(scores_per_round[:, r]) / len(scores_per_round[:, r]) for r in range(len(num_rounds))]
# avg_score = sum(scores.values()) / len(scores)
client_avg_scores_per_round = [{id: sum(scores) / len(scores) for id, scores in client_scores[r].items()} for r in num_rounds]
# client_avg_scores = {id: sum(scores) / len(scores) for id, scores in client_scores.items()}

# Prepare data for CSV
csv_datas = []
for i, (avg_score, client_avg_scores) in enumerate(zip(avg_scores, client_avg_scores_per_round)):
    csv_data = [Method, avg_score]
    csv_data.extend([client_avg_scores.get(i, '') for i in range(client_num)])  # Assuming client IDs are 1-10
    csv_data.extend(scores_per_round[:, i])
    csv_datas.append(csv_data)

# Prepare header
header = ['method', 'final score']
header.extend([f'client {i}' for i in range(client_num)])
header.extend(scores.keys())

# Write to CSV
csv_file = f'{Method}.csv'
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
