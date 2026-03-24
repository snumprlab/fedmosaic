import jsonlines
import json
import csv
from collections import defaultdict
import os
import matplotlib.pyplot as plt

mode_method_dict = {
    # 'Frozen weight (1B)':'sft_bs4_svaeoptim_lr2e-5_sc7_1tasks_1rounds_fixitr1000_memonly',
    # 'Frozen weight (1B, rank32)':'sft_bs4_svaeoptim_lr2e-5_sc7_1tasks_1rounds_fixitr1000_memonly_rank32',
    # 'Frozen weight (3B)':'sft_bs4_svaeoptim_lr2e-5_sc8_1tasks_1rounds_fixitr1000_memonly',
    # 'Frozen weight (3B, rank32)':'sft_bs4_svaeoptim_lr2e-5_sc8_1tasks_1rounds_fixitr1000_memonly_rank32',
    # 'FedAvg(1B)':'sft_bs4_svaeoptim_lr2e-5_sc7_1tasks_1rounds_fixitr1000_fedavg_r15_memonly',
    # 'FedAvg(3B)':'sft_bs4_svaeoptim_lr2e-5_sc8_1tasks_1rounds_fixitr1000_fedavg_r15_memonly',
    # 'FedAvg(3B, rank32)':'sft_bs4_saveoptim_lr2e-5_sc8_4tasks_1rounds_fixitr1000_fedavg_r15_memonly_rank32',
    # 'FedOurs(1B)':'sft_bs4_svaeoptim_lr2e-5_sc7_1tasks_1rounds_fixitr1000_fedours_t0.2_r15_memonly',
    # 'FedOurs(3B)':'sft_bs4_svaeoptim_lr2e-5_sc8_1tasks_1rounds_fixitr1000_fedours_t0.2_r15_memonly',
    # 'FedOurs(3B, rank32)':'sft_bs4_svaeoptim_lr2e-5_sc8_1tasks_1rounds_fixitr1000_fedours_t0.2_rank32_r15_memonly',
    
    # 'Frozen weight (1B)':'sft_bs4_svaeoptim_lr2e-5_sc9_1tasks_1rounds_fixitr1000_memonly',
    # 'Frozen weight (1B, rank32)':'sft_bs4_svaeoptim_lr2e-5_sc9_1tasks_1rounds_fixitr1000_memonly_rank32',
    # 'Frozen weight (3B)':'sft_bs4_svaeoptim_lr2e-5_sc10_1tasks_1rounds_fixitr1000_memonly',
    # 'Frozen weight (3B, rank32)':'sft_bs4_svaeoptim_lr2e-5_sc10_1tasks_1rounds_fixitr1000_memonly_rank32',
    # 'FedAvg(1B)':'sft_bs4_svaeoptim_lr2e-5_sc9_1tasks_1rounds_fixitr1000_fedavg_r15_memonly',
    # 'FedAvg(3B)':'sft_bs4_svaeoptim_lr2e-5_sc10_1tasks_1rounds_fixitr1000_fedavg_r15_memonly',
    # 'FedAvg(3B, rank32)':'sft_bs4_saveoptim_lr2e-5_sc10_4tasks_1rounds_fixitr1000_fedavg_r15_memonly_rank32',
    # 'FedOurs(1B)':'sft_bs4_svaeoptim_lr2e-5_sc9_1tasks_1rounds_fixitr1000_fedours_t0.2_r15_memonly',
    # 'FedOurs(3B)':'sft_bs4_svaeoptim_lr2e-5_sc10_1tasks_1rounds_fixitr1000_fedours_t0.2_r15_memonly',
    # 'FedOurs(3B, rank32)':'sft_bs4_svaeoptim_lr2e-5_sc10_1tasks_1rounds_fixitr1000_fedours_t0.2_rank32_r15_memonly',
    
    # 'Frozen Weight': 'sft_bs4_saveoptim_lr2e-5_sc8_1tasks_1rounds_fixitr801_T0125_decay099',
    # 'SFT (r20)': 'sft_bs4_saveoptim_lr2e-5_sc8_1tasks_1rounds_fixitr801_T0125_decay099_sft_r20',
    # 'FedAVG (r20)': 'sft_bs4_saveoptim_lr2e-5_sc8_1tasks_1rounds_fixitr801_T0125_decay099_fedavg_r20',
    # 'Gradsim+MOE (r20)': 'sft_bs4_saveoptim_lr2e-5_sc8_1tasks_1rounds_fixitr801_T0125_decay099_ours_moe_r20',
    
    # 'Frozen Weight': 'sft_bs4_saveoptim_lr2e-5_sc10_1tasks_1rounds_fixitr801_T0125_decay099',
    # 'SFT (r20)': 'sft_bs4_saveoptim_lr2e-5_sc10_1tasks_1rounds_fixitr801_T0125_decay099_sft_r20',
    # 'FedAVG (r20)': 'sft_bs4_saveoptim_lr2e-5_sc10_1tasks_1rounds_fixitr801_T0125_decay099_fedavg_r20',
    # 'Gradsim+MOE (r20)': 'sft_bs4_saveoptim_lr2e-5_sc10_1tasks_1rounds_fixitr801_T0125_decay099_ours_moe_r20'
    
    # 'Frozen Weight': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr2e-5_5e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099',
    # 'FedAVG (r20)': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr2e-5_5e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_fedavg_r20',
    # 'Gradsim+MOE (r20)': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr2e-5_5e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20'
    
    # 'Frozen Weight': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr2e-5_5e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099',
    # 'FedAVG (r20)': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr2e-5_5e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_fedavg_r20',
    # 'Gradsim+MOE (r20)': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr2e-5_5e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20'
    
    
    # 'Frozen Weight': 'sft_bs4_saveoptim_lr1e-5_sc8_1tasks_1rounds_fixitr401_T0125_decay099',
    # 'FedAvg': 'sft_bs4_saveoptim_lr1e-5_sc8_1tasks_1rounds_fixitr401_T0125_decay099_fedavg_r20',
    # 'FedAvg + Weighted Avg (4)': 'sft_bs4_saveoptim_lr1e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_fedavg_r20_4',
    # 'GSAFed + Avg': 'sft_bs4_saveoptim_lr1e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_avg',
    # 'GSAFed + Weighted Avg (4)': 'sft_bs4_saveoptim_lr1e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_4',
    # 'GSAFed + Weighted Avg (12)': 'sft_bs4_saveoptim_lr1e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_12',
    # 'GSAFed + Weighted Avg (20)': 'sft_bs4_saveoptim_lr1e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'GSAFed + Weighted Avg (80)': 'sft_bs4_saveoptim_lr1e-5_sc8_1tasks_1rounds_fixitr401_T0125_decay099_ours_moe_r20',
    # 'FedAVG (r20) Merge': 'sft_bs4_saveoptim_lr2e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_fedavg_r20_merge',
    # 'Gradsim+MOE (r20) Merge': 'sft_bs4_saveoptim_lr2e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_merge',
    
    # 'Frozen Weight': 'sft_bs4_saveoptim_lr1e-5_sc10_1tasks_1rounds_fixitr401_T0125_decay099',
    # 'FedAvg': 'sft_bs4_saveoptim_lr1e-5_sc10_1tasks_1rounds_fixitr401_T0125_decay099_fedavg_r20',
    # 'GSAFed + Avg': 'sft_bs4_saveoptim_lr1e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_avg',
    # 'GSAFed + Weighted Avg (4)': 'sft_bs4_saveoptim_lr1e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_4',
    # 'GSAFed + Weighted Avg (12)': 'sft_bs4_saveoptim_lr1e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_12',
    # 'GSAFed + Weighted Avg (20)': 'sft_bs4_saveoptim_lr1e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'GSAFed + Weighted Avg (80)': 'sft_bs4_saveoptim_lr1e-5_sc10_1tasks_1rounds_fixitr401_T0125_decay099_ours_moe_r20',
     
    # 'FedAVG (r20) Merge': 'sft_bs4_saveoptim_lr2e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_fedavg_r20_merge',
    # 'Gradsim+MOE (r20) Merge': 'sft_bs4_saveoptim_lr2e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_merge',
    
    # 'Frozen Weight': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr1e-5_2e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099',
    # 'FedAvg': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr1e-5_2e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_fedavg_r20',
    # 'GSAFed + Avg': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr1e-5_2e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_avg',
    # 'GSAFed + Weighted Avg': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr1e-5_2e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20'
    
    # 'Frozen Weight': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr1e-5_2e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099',
    # 'FedAvg': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr1e-5_2e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_fedavg_r20',
    # 'GSAFed + Avg': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr1e-5_2e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_avg',
    # 'GSAFed + Weighted Avg': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr1e-5_2e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20'
    
    # 'Frozen Weight': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr1e-5_2e-5_sc7_1tasks_1rounds_fixitr201_T0125_decay099',
    # 'FedAVG (r20)': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr1e-5_2e-5_sc7_1tasks_1rounds_fixitr201_T0125_decay099_fedavg_r20',
    # 'Gradsim+MOE (r20)': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr1e-5_2e-5_sc7_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20'
    
    # 'Frozen Weight': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr1e-5_2e-5_sc9_1tasks_1rounds_fixitr201_T0125_decay099',
    # 'FedAVG (r20)': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr1e-5_2e-5_sc9_1tasks_1rounds_fixitr201_T0125_decay099_fedavg_r20',
    # 'Gradsim+MOE (r20)': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr1e-5_2e-5_sc9_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20'
    
    # 'Frozen Weight': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr2e-5_5e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099',
    # 'FedAVG (r20)': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr2e-5_5e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_fedavg_r20',
    # 'Gradsim+MOE (r20)': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr2e-5_5e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE 1e-5/1e-4': 'fedMultipqfullfreeze_homoAgg_sft_pca_bs4_saveoptim_lr1e-5_1e-4_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE 1e-5/5e-5': 'fedMultipqfullfreeze_homoAgg_sft_pca_bs4_saveoptim_lr1e-5_5e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE 2e-5/1e-4': 'fedMultipqfullfreeze_homoAgg_sft_pca_bs4_saveoptim_lr2e-5_1e-4_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE 2e-5/5e-5': 'fedMultipqfullfreeze_homoAgg_sft_pca_bs4_saveoptim_lr2e-5_5e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    
    # 'Gradsim+MOE homoOnly 1e-5/1e-4': 'fedMultipqfullfreeze_homoAggOnly_sft_pca_bs4_saveoptim_lr1e-5_1e-4_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE homoOnly 1e-5/5e-5': 'fedMultipqfullfreeze_homoAggOnly_sft_pca_bs4_saveoptim_lr1e-5_5e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE homoOnly 2e-5/1e-4': 'fedMultipqfullfreeze_homoAggOnly_sft_pca_bs4_saveoptim_lr2e-5_1e-4_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE homoOnly 2e-5/5e-5': 'fedMultipqfullfreeze_homoAggOnly_sft_pca_bs4_saveoptim_lr2e-5_5e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    
    # 'Frozen Weight': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr2e-5_5e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099',
    # 'FedAVG (r20)': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr2e-5_5e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_fedavg_r20',
    # # 'Gradsim+MOE (r20)': 'fedMultipqfullfreeze_homoAgg_pca_bs4_saveoptim_lr2e-5_5e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20',
    # 'Gradsim+MOE 1e-5/1e-4': 'fedMultipqfullfreeze_homoAgg_sft_pca_bs4_saveoptim_lr1e-5_1e-4_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE 1e-5/5e-5': 'fedMultipqfullfreeze_homoAgg_sft_pca_bs4_saveoptim_lr1e-5_5e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE 2e-5/1e-4': 'fedMultipqfullfreeze_homoAgg_sft_pca_bs4_saveoptim_lr2e-5_1e-4_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE 2e-5/5e-5': 'fedMultipqfullfreeze_homoAgg_sft_pca_bs4_saveoptim_lr2e-5_5e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    
    # 'Gradsim+MOE homoOnly 1e-5/1e-4': 'fedMultipqfullfreeze_homoAggOnly_sft_pca_bs4_saveoptim_lr1e-5_1e-4_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE homoOnly 1e-5/5e-5': 'fedMultipqfullfreeze_homoAggOnly_sft_pca_bs4_saveoptim_lr1e-5_5e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE homoOnly 2e-5/1e-4': 'fedMultipqfullfreeze_homoAggOnly_sft_pca_bs4_saveoptim_lr2e-5_1e-4_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE homoOnly 2e-5/5e-5': 'fedMultipqfullfreeze_homoAggOnly_sft_pca_bs4_saveoptim_lr2e-5_5e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    
    # 'Gradsim+MOE 1e-5/5e-5': 'fedMulti2pqfullfreeze_back_homoAgg_sft_pca_bs4_saveoptim_lr1e-5_5e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE homoOnly 1e-5/5e-5': 'fedMulti2pqfullfreeze_back_homoAggOnly_sft_pca_bs4_saveoptim_lr1e-5_5e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    
    # 'Gradsim+MOE 1e-5/1e-4': 'fedMulti2pqfullfreeze_back_homoAgg_sft_pca_bs4_saveoptim_lr1e-5_1e-4_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE 1e-5/5e-5': 'fedMulti2pqfullfreeze_back_homoAgg_sft_pca_bs4_saveoptim_lr1e-5_5e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE 2e-5/1e-4': 'fedMulti2pqfullfreeze_back_homoAgg_sft_pca_bs4_saveoptim_lr2e-5_1e-4_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE 2e-5/5e-5': 'fedMulti2pqfullfreeze_back_homoAgg_sft_pca_bs4_saveoptim_lr2e-5_5e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    
    # 'Gradsim+MOE homoOnly 1e-5/1e-4': 'fedMulti2pqfullfreeze_back_homoAggOnly_sft_pca_bs4_saveoptim_lr1e-5_1e-4_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE homoOnly 1e-5/5e-5': 'fedMulti2pqfullfreeze_back_homoAggOnly_sft_pca_bs4_saveoptim_lr1e-5_5e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE homoOnly 2e-5/1e-4': 'fedMulti2pqfullfreeze_back_homoAggOnly_sft_pca_bs4_saveoptim_lr2e-5_1e-4_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    # 'Gradsim+MOE homoOnly 2e-5/5e-5': 'fedMulti2pqfullfreeze_back_homoAggOnly_sft_pca_bs4_saveoptim_lr2e-5_5e-5_sc10_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    
    # 'Frozen Weight': 'sft_bs4_saveoptim_lr1e-5_sc12_1tasks_1rounds_fixitr51_T0125_decay099',
    # 'FedAvg': 'sft_bs4_saveoptim_lr1e-5_sc12_1tasks_1rounds_fixitr51_T0125_decay099_fedavg_r20',
    # 'GSAFed + Weighted Avg (4)': 'sft_bs4_saveoptim_lr1e-5_sc12_1tasks_1rounds_fixitr51_T0125_decay099_ours_moe_r20_4',
    
    'Random init':'sft_bs4_saveoptim_lr1e-5_sc8_1tasks_1rounds_fixitr401_T0125_decay099',
    'SFT':'sft_bs4_saveoptim_lr1e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_sft_r20',
    'FedIT':'sft_bs4_saveoptim_lr1e-5_sc8_1tasks_1rounds_fixitr401_T0125_decay099_fedavg_r20',
    'FedSIM':'sft_bs4_saveoptim_lr1e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_fedsim_r20_new',
    'FedDPA':'sft_bs4_saveoptim_lr1e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_feddpa_r20',
    'FedMKT':'sft_bs4_saveoptim_lr1e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_fedmkt_r20_new',
    'FedMosaic':'sft_bs4_saveoptim_lr1e-5_sc8_1tasks_1rounds_fixitr201_T0125_decay099_ours_moe_r20_20',
    
}


    

# METHOD='fedMultipqfullfreeze_homoAgg_sft'
# METHOD='fedMulti2pqfullfreeze_back_homoAgg_sft'
METHOD='sft_finetune'

# colors = ['#3A1730', '#C18A3D', '#588157', '#E63946', '#BCBD22', '#17BECF', '#457B9D']
# colors = ['#457B9D', '#314832', '#D8CFC0', '#E63946', '#3A1730', '#C18A3D', '#588157', '#38322C', '#BCBD22', '#17BECF']
colors = ['#3030FF', '#2ECC71', '#D04C3C', '#E3B448', '#1B6535', '#B1624E', '#BDC3C7', '#CBCE91', '#195190', '#FF5733', '#9B59B6', '#EF9DAF']

mode_color_dict={
    'Random init':colors[4],
    'SFT':'#000000',
    'FedIT':colors[-1],
    'FedSIM':colors[9],
    'FedDPA':colors[3],
    'FedMKT':colors[5],
    'FedMosaic':colors[0],
}

scenario_num = 8
with open(f"./scenarios/scenario-{scenario_num}.json") as fp:
    scenario = json.load(fp)

iters = [0, 25, 50, 75, 100, 125,150,175, 200] #, 250, 300, 350, 400, 450, 500, 600, 700]#, 800]
# iters = [0,5,10,15,20,25,30,35,40,50]
# iters = [0, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400]
# iters = [0, 5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350]
# iters = [0, ,5, 10, 20, 30, 40, 50, 100]
num_rounds = 1

# plot_mode = 'seen_task'
data_indices = [
    [0],[0],[0],[0],[0],
    [0],[0],[0],[0],[0],
]

for client_data in scenario:
    id = client_data['client_id']
    mode_scores = {}
    for mode in mode_method_dict.keys():
        Method = mode_method_dict[mode]
        client_scores = []
        for num_round in range(num_rounds):
            data_index = data_indices[num_round]
            done = False
            for iter in iters:
                summed_score = 0
                for d_idx in data_index:
                    data = client_data['datasets'][d_idx]
                    data_name = f"{data['dataset']}-{data['subset_id']}"
                    
                    try:
                        filename = f'./eval_results/{METHOD}/{Method}/client{id}_round{num_round+1}_iter{iter}_{data_name}.json'
                        with open(filename, 'r') as fp:
                            result = json.load(fp)[-1]
                    except Exception as e:
                        print(e)
                        done = True
                        break
                    
                    if data['type'] == 'multi-choice':
                        score = result['accuracy']
                    elif data['type'] == 'open-ended':
                        if data['metric'] == 'F1':
                            score = 2*(result['precision']*result['recall']) / (result['precision'] + result['recall'])
                        elif data['metric'] == 'RougeL':
                            score = result['ROUGE_L'][0]
                        elif data['metric'] == 'cider':
                            score = result['CIDEr'][0]
                    summed_score += score * 100
                if done:
                    break
                client_scores.append(summed_score / len(data_index))                    
        mode_scores[mode] = client_scores
        
    # Plotting the scores
    plt.figure(figsize=(6, 3.5))
    plt.axes().set_facecolor("#F5F5F5")
    plt.rc('axes', labelsize=16)
    # plt.rc('xtick', labelsize=16)
    # plt.rc('ytick', labelsize=16)
    
    y = iters#[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    for mode, scores in mode_scores.items():
        print(f'{data_name} | {mode} | AUC: {sum(scores)/len(scores)} | Final Acc: {scores[-1]}')
        plt.plot(y[:len(scores)], scores, label=f'{mode}', linewidth=3.0, color=mode_color_dict[mode])#, marker='o')
    
    # plt.title(f'{data_name}', fontsize=20)
    plt.title('RecipeQA: VisualCloze', fontsize=26)
    plt.xlabel('Iterations', fontsize=24)
    plt.ylabel('Accuracy (%)', fontsize=24)
    plt.tick_params(axis='both', labelsize=22)
    # plt.legend(fontsize=14)
    # plt.grid(axis='y')
    plt.grid(True)
    

    # Save the plot
    plt.savefig(f'plot_unseen_train_client_{id}_sc{scenario_num}.png', dpi=300, bbox_inches='tight')