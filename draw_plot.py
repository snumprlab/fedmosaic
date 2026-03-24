import jsonlines
import json
import csv
from collections import defaultdict
import os
import matplotlib.pyplot as plt

mode_method_dict = {
    # 'SFT(1B)':'sft_bs4_saveoptim_lr2e-5_sc0_4tasks_5rounds_fixitr100_llama3_1B_lora_memonly',
    # 'SFT(3B)':'sft_bs4_saveoptim_lr2e-5_sc5_4tasks_5rounds_fixitr100',
    # 'SFT(3B, rank32)':'sft_bs4_saveoptim_lr2e-5_sc5_4tasks_5rounds_fixitr100_memonly_rank32',
    # 'FedAvg(1B)':'fedavg_bs4_saveoptim_lr2e-5_sc0_4tasks_5rounds_fixitr100',
    # 'FedAvg(3B)':'fedavg_bs4_saveoptim_lr2e-5_sc5_4tasks_5rounds_fixitr100',
    # 'FedAvg(3B, rank32)':'fedavg_bs4_saveoptim_lr2e-5_sc5_4tasks_5rounds_fixitr100_rank32',
    # 'FedOurs(1B)':'fedours_bs4_saveoptim_lr4e-5_sc0_4tasks_5rounds_fixitr100_t0.2_memonly',
    # 'FedOurs(1B, rank32)':'fedours_bs4_saveoptim_lr4e-5_sc0_4tasks_5rounds_fixitr100_t0.2_memonly_rank32',
    # 'FedOurs(3B)':'fedours_bs4_saveoptim_lr4e-5_sc5_4tasks_5rounds_fixitr100_t0.2_memonly',
    # 'FedOurs(3B, rank32)':'fedours_bs4_saveoptim_lr4e-5_sc5_4tasks_5rounds_fixitr100_t0.2_memonly_rank32',
    
    'SFT (20rounds, random)': 'sft_bs4_svaeoptim_lr2e-5_sc5_4tasks_5rounds_fixitr500_random_memonly',
    'SFT (20rounds, T=0.125)': 'sft_bs4_svaeoptim_lr2e-5_sc5_4tasks_5rounds_fixitr500_memonly',
    'SFT (20rounds, T=0.250)': 'sft_bs4_saveoptim_lr2e-5_sc5_4tasks_5rounds_fixitr500_memonly_T0250',
    'FedAvg (20rounds, random)': 'fedavg_bs4_svaeoptim_lr2e-5_sc5_4tasks_5rounds_fixitr500_random_memonly',
    'FedAvg (20rounds, T=0.125)': 'fedavg_bs4_svaeoptim_lr2e-5_sc5_4tasks_5rounds_fixitr500_memonly',
    'FedAvg (100rounds, random)': 'fedavg_bs4_svaeoptim_lr2e-5_sc5_4tasks_25rounds_fixitr100_random_memonly',
    'FedAvg (100rounds, T=0.125)': 'fedavg_bs4_svaeoptim_lr2e-5_sc5_4tasks_25rounds_fixitr100_memonly',
    'FedOurs (20rounds, T=0.125)': 'fedours_bs4_saveoptim_lr4e-5_sc5_4tasks_5rounds_fixitr500_t0.2_memonly',
    # 'FedOurs (20rounds, T=0.250)': 'fedours_bs4_saveoptim_lr4e-5_sc5_4tasks_5rounds_fixitr500_memonly_T0250',
    'FedOurs (100rounds, T=0.125)': 'fedours_bs4_svaeoptim_lr4e-5_sc5_4tasks_25rounds_fixitr100_memonly',
    # 'FedOurs (100rounds, random)': 'fedours_bs4_svaeoptim_lr4e-5_sc5_4tasks_25rounds_fixitr100_random_memonly',
}

# colors = ['#3A1730', '#C18A3D', '#588157', '#E63946', '#BCBD22', '#17BECF', '#457B9D']
colors = ['#457B9D', '#314832', '#D8CFC0', '#E63946', '#3A1730', '#C18A3D', '#588157', '#38322C', '#BCBD22', '#17BECF']

mode_color_dict = {
    'SFT(3B, rank32)': colors[3],
    'FedOurs(1B)': colors[1],
    'FedOurs(3B)': colors[7],
    # 'LLaVA 1B - MemoryOnly': colors[0],
    # 'LLaVA 3B - MemoryOnly': colors[7],
}

scenario_num = 5
with open(f"./scenarios/scenario-{scenario_num}.json", 'r') as fp:
    scenario = json.load(fp)

# iters = [0, 25, 50, 75, 100]
num_rounds = 19

# plot_mode = 'seen_task'
plot_mode = 'all_task'
if plot_mode == 'diagonal':
    data_indices = [
        [0],[0],[0],[0],[0],[0],
        [1],[1],[1],[1],[1],
        [2],[2],[2],[2],[2],
        [3],[3],[3],[3],[3],
    ]
elif plot_mode == 'seen_task':
    data_indices = [
        [0],[0],[0],[0],[0],[0],
        [0,1],[0,1],[0,1],[0,1],[0,1],
        [0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2],
        [0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],
    ]
elif plot_mode == 'all_task':
    data_indices = [
        [0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],
        [0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],
        [0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],
        [0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],
    ]
elif plot_mode == 'unseen_task':
    data_indices = [
        [0,1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],
        [2,3],[2,3],[2,3],[2,3],[2,3],
        [3],[3],[3],[3],[3],
        [3],[3],[3],[3],[3],
    ]
AUC = {}
final_score = {}
for mode in mode_method_dict.keys():
    AUC[mode] = []
    final_score[mode] = []
for client_data in scenario:
    id = client_data['client_id']
    mode_scores = {}
    for name in mode_method_dict.keys():
        Method = mode_method_dict[name]
        client_scores = []
        for num_round in range(num_rounds+1):
            data_index = data_indices[num_round]
            summed_score = 0
            for d_idx in data_index:
                data = client_data['datasets'][d_idx]
                data_name = f"{data['dataset']}-{data['subset_id']}"
                Method_name = Method.split('_')[0]
                if num_round > 0:
                    if '_25rounds_' in Method:
                        filename = f'./eval_results/{Method_name}/{Method}/client{id}_round{num_round*5}_{data_name}.json'
                    else:
                        filename = f'./eval_results/{Method_name}/{Method}/client{id}_round{num_round}_{data_name}.json'
                else:
                    cur_scenario_num = int(Method.split('_')[4][-1])
                    with open(f"./scenarios/scenario-{cur_scenario_num}.json", 'r') as fp:
                        cur_scenario = json.load(fp)
                    if "llama3.2_1B_vl" in cur_scenario[id]['model_id']:
                        zeroshot_folder = "llama3_1b_zeroshot"
                    elif "llama3.2_3B_vl" in cur_scenario[id]['model_id']:
                        zeroshot_folder = "llama3_3b_zeroshot"
                    filename = f'./eval_results/llama3/{zeroshot_folder}/client{id}_round20_{data_name}.json'
                # print(filename)
                with open(filename, 'r') as fp:
                    result = json.load(fp)[-1]
                
                if data['type'] == 'multi-choice':
                    score = result['accuracy']
                elif data['type'] == 'open-ended':
                    if data['metric'] == 'F1':
                        score = 2*(result['precision']*result['recall']) / (result['precision'] + result['recall'])
                    elif data['metric'] == 'RougeL':
                        score = result['ROUGE_L'][0]
                summed_score += score
            
            client_scores.append(summed_score / len(data_index))
            
            if num_round < 20 and num_round > 0 and num_round % 5 == 0:
                summed_score = 0
                for d_idx in data_indices[num_round+1]:
                    data = client_data['datasets'][d_idx]
                    data_name = f"{data['dataset']}-{data['subset_id']}"
                    Method_name = Method.split('_')[0]
                    if '_25rounds_' in Method:
                        filename = f'./eval_results/{Method_name}/{Method}/client{id}_round{num_round*5+5}_{data_name}.json'
                    else:
                        filename = f'./eval_results/{Method_name}/{Method}/client{id}_round{num_round+1}_{data_name}.json'
                    # print(filename)
                    with open(filename, 'r') as fp:
                        result = json.load(fp)[-1]
                    
                    if data['type'] == 'multi-choice':
                        score = result['accuracy']
                    elif data['type'] == 'open-ended':
                        if data['metric'] == 'F1':
                            score = 2*(result['precision']*result['recall']) / (result['precision'] + result['recall'])
                        elif data['metric'] == 'RougeL':
                            score = result['ROUGE_L'][0]
                    summed_score += score
                
                client_scores.append(summed_score / len(data_indices[num_round+1]))
                    
        
        mode_scores[name] = client_scores

    # Plotting the scores
    plt.figure(figsize=(8, 4.8))
    plt.axes().set_facecolor("#F5F5F5")
    plt.rc('axes', labelsize=16)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    
    # y = [0,100,200,300,400,500,
    #      500,600,700,800,900,1000,
    #      1000,1100,1200,1300,1400,1500,
    #      1500,1600,1700,1800,1900,2000]
    
    y = [0,500,1000,1500,2000,2500,
         2500,3000,3500,4000,4500,5000,
         5000,5500,6000,6500,7000,7500,
         7500,8000,8500,9000,9500,10000]
    
    # compute AUC
    # print(f'client id: {id}')
    for mode, scores in mode_scores.items():
        # auc = 0
        # for i in range(len(scores)-1):
        #     y1 = y[i]
        #     y2 = y[i+1]
        #     if y1 == y2:
        #         continue
        #     score1 = scores[i]
        #     score2 = scores[i+1]
        #     auc += 0.5 * (score1+score2) * (y2 - y1)
        if plot_mode == "unseen_task":
            AUC[mode].append(sum(scores[:-1])/(len(scores)))
            final_score[mode].append(scores[-2])
        else:
            AUC[mode].append(sum(scores[:])/(len(scores)))
            final_score[mode].append(scores[-1])
        # print(f'{mode} AUC for client {id}: {auc}')
    
    if plot_mode == 'unseen_task':
        y = y[:-5]
    for mode, scores in mode_scores.items():
        if plot_mode == 'unseen_task':
            scores = scores[:-5]
        plt.plot(y[:len(scores)], scores, label=f'{mode}', linewidth=2.0)#, color=mode_color_dict[mode])#, marker='o')
    
    if plot_mode == 'unseen_task':
        range_max = 1501
    else:
        # range_max = 2001
        range_max = 10001
    for y_val in range(0, range_max, 500):
        plt.axvline(x=y_val, color='#FFAEAE', linestyle='--', linewidth=0.5)
        if y_val % 2500 == 0:
            plt.axvline(x=y_val, color='#FFAEAE', linestyle='--', linewidth=2.0)

    plt.title(f'Client {id} {plot_mode} Scores', fontsize=20)
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Score', fontsize=18)
    plt.legend(fontsize=10)
    plt.grid(axis='y')
    

    # Save the plot
    plt.savefig(f'plot_{plot_mode}_client_{id}.png')
    
for mode, aucs in AUC.items():
    print(f'{mode}: {aucs} | average AUC: {sum(aucs)/len(aucs)}')
    print(f'{mode}: {final_score[mode]} | average final ACC: {sum(final_score[mode])/len(final_score[mode])}')