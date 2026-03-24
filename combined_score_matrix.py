import jsonlines
import json
import csv
from collections import defaultdict
import os
import numpy as np

mode = 'fedsim'
Method='fedsim_NOCONT_bs4_saveoptim_lr2e-5_5e-5_sc315_4tasks_5rounds_fixitr75_T0125_decay099_SEED2'

# num_rounds = [2,5,7,10,12,15,17,20]
num_rounds = [5,10,15,20]
is_client = True

scenario_num = 315
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
                data_name = f"{data['dataset']}-{data['subset_id']}"
                if is_client:
                    # print(f'./eval_results/{mode}/{Method}/client{client_id}_round{num_round}_{data_name}.json')
                    with open(f'./eval_results/{mode}/{Method}/client{client_id}_round{num_round}_{data_name}.json', 'r') as fp:
                        result = json.load(fp)[-1]
                else:
                    with open(f'./eval_results/{mode}/{Method}/server_round{num_round}_{data_name}.json', 'r') as fp:
                        result = json.load(fp)[-1]
                
                if data['type'] == 'multi-choice':
                    score = result['accuracy']
                elif data['type'] == 'open-ended':
                    if data['metric'] == 'F1':
                        score = 2*(result['precision']*result['recall']) / (result['precision'] + result['recall'])
                    elif data['metric'] == 'RougeL':
                        score = result['ROUGE_L'][0]
                
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