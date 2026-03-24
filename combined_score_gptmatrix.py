import jsonlines
import json
import csv
from collections import defaultdict
import os
import numpy as np

mode = 'feddpa_feddualMultipqfullfreeze_homoAgg'
Method='feddpa_feddualMultipqfullfreeze_homoAgg_bs4_nosaveoptim_COSINE_r16_32_lr3e-4_5e-4_sc262_4tasks_5rounds_fixitr50_T0125_decay099'
# Method='llama3_3b_2'
num_rounds = [20]
is_client = True

scenario_num = 262
with open(f"./scenarios/scenario-{scenario_num}.json") as fp:
    scenario = json.load(fp)

for num_round in num_rounds:
    results = []
    for client_id in range(len(scenario)):
        client_results = []
        for client_data in scenario:
            task_score = 0
            id = client_data['client_id']
            for data in client_data['datasets']:
                total_score = 0
                count = 0
                data_name = f"{data['dataset']}-{data['subset_id']}"
                with jsonlines.open(f'./eval_results_gpt/{mode}/{Method}/client{client_id}_round{num_round}_{data_name}.jsonl') as read_file:
                    for line in read_file.iter():
                        total_score += line['score']
                        count+=1
                score = total_score/count
                task_score += score
            client_results.append(task_score/len(client_data['datasets']))
        results.append(client_results)
    
    # Prepare data for CSV
    csv_datas = []
    for i, (result) in enumerate(results):
        csv_data = [Method, num_round, i]
        csv_data.extend(result[:])
        csv_datas.append(csv_data)

    # Prepare header
    header = ['method', 'num_round', 'client']
    header.extend([f'client {i}' for i in range(10)])

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
    print(f"Results have been appended to {csv_file}")