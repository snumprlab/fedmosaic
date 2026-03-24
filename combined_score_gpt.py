import jsonlines
import json
import csv
from collections import defaultdict
import os

mode = 'fedours_include'
Method='fedours_include_T1_bs8_saveoptim_lr2e-5_sc215_4tasks_5rounds_fixitr50_T025_decay09'
# Method='llama3_3b_2'
num_rounds = [20]
is_client = True

scenario_num = 215
with open(f"./scenarios/scenario-{scenario_num}.json") as fp:
    scenario = json.load(fp)

scores = {}
client_scores = defaultdict(list)

for num_round in num_rounds:
    for client_data in scenario:
        id = client_data['client_id']
        for data in client_data['datasets']:
            data_name = f"{data['dataset']}-{data['subset_id']}"
            
            if data['type'] == 'open-ended':
                count = 0
                total_score = 0
                if is_client:
                    with jsonlines.open(f'./eval_results_gpt/{mode}/{Method}/client{id}_round{num_round}_{data_name}.jsonl') as read_file:
                        for line in read_file.iter():
                            total_score += line['score']
                            count+=1
                else:
                    with jsonlines.open(f'./eval_results_gpt/{mode}/{Method}/server_round{num_round}_{data_name}.jsonl') as read_file:
                        for line in read_file.iter():
                            total_score += line['score']
                            count+=1
                score = total_score/count
            elif data['type'] == 'multi-choice':
                if is_client:
                    with open(f'./eval_results/{mode}/{Method}/client{id}_round{num_round}_{data_name}.json', 'r') as fp:
                        result = json.load(fp)[-1]
                else:
                    with open(f'./eval_results/{mode}/{Method}/server_round{num_round}_{data_name}.json', 'r') as fp:
                        result = json.load(fp)[-1]
                score = result['accuracy']
            
            scores[data_name] = score
            client_scores[id].append(score)

    avg_score = sum(scores.values()) / len(scores)
    client_avg_scores = {id: sum(scores) / len(scores) for id, scores in client_scores.items()}

    # Prepare data for CSV
    csv_data = [Method, avg_score]
    csv_data.extend([client_avg_scores.get(i, '') for i in range(10)])  # Assuming client IDs are 1-10
    csv_data.extend(scores.values())

    # Prepare header
    header = ['method', 'final score']
    header.extend([f'client {i}' for i in range(10)])
    header.extend(scores.keys())

    # Write to CSV
    csv_file = 'results.csv'
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(csv_data)

    print(f"Method: {Method}")
    print(f"Score: {scores}")
    print(f"Avg Score: {avg_score}")
    print(f"Results have been appended to {csv_file}")
