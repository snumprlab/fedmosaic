import logging.config
import os
import random
import gc

import numpy as np
import torch
from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingConfig
import transformers
from utils.train_utils import get_VLMmodel, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, get_task_vectors, load_deepspeed, configure_online_datastream, get_keys_to_del

from federated_methods.method_manager import select_method
from utils.data_loader_VLM import LazySupervisedDataset, DataCollatorForSupervisedDataset
from typing import Dict

import copy
import json
from transformers import BitsAndBytesConfig
import time
import datetime
import torch.nn.functional as F

from federated_methods.CKA_feat_extract import cka_create_trainer, compute_rdm, rdm_similarity

os.environ["WANDB_DISABLED"] = "true"

def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"results/{training_args.mode}/{training_args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'results/{training_args.mode}/{training_args.note}/seed_{training_args.seed}.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    if training_args.local_rank == 0 or training_args.local_rank == -1: 
        logger.info(training_args)

    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)
    
    
    train_datalists, test_datalists = get_datalists(training_args, training_args.scenario)
    
    # select functions
    # set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules = select_method(training_args.mode)
    
    # create folder
    training_args.state_dir = training_args.state_dir + '_' + training_args.note
    if not os.path.exists(training_args.state_dir):
        os.makedirs(training_args.state_dir)
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}
    
    
    model_ids = {}
    model_list = {}
    models = {}
    global_state_dict_list = []
    local_state_dict_list = []
    old_local_state_dict_list = []
    for client_id in range(len(train_datalists)):
        train_datalist = train_datalists[client_id]
        model_id = train_datalist[0]['model_id']
        
        if model_id in model_list.keys():
            local_state_dict_list.append(copy.deepcopy(model_list[model_id]))
            old_local_state_dict_list.append(copy.deepcopy(model_list[model_id]))
            global_state_dict = copy.deepcopy(model_list[model_id])
            keys_to_del = get_keys_to_del(training_args, global_state_dict)
            for k in keys_to_del:
                del global_state_dict[k]
            global_state_dict_list.append(global_state_dict)
            
            model_ids[model_id].append(client_id)
        else:
            new_model_args = copy.deepcopy(model_args)
            new_model_args.model_name_or_path = model_id
            model, tokenizer, processor, new_data_args = get_VLMmodel(new_model_args, training_args, bnb_model_from_pretrained_args, data_args)
            
            if training_args.load_checkpoint is not None and not training_args.fedours:
                logger.info(f'load {training_args.load_checkpoint}')
                server_state_dict = torch.load(training_args.load_checkpoint, map_location='cpu')
                
                with torch.no_grad():
                    model.load_state_dict(server_state_dict, strict=False)
                
                if ('fedours' in training_args.load_checkpoint) and training_args.mode not in ['fedours', 'ours_generator', 'ours_generator2']:
                    local_state_dict = {}
                    for name in server_state_dict.keys():
                        if 'lora1' in name:
                            target_key = name.replace('lora1', 'lora')
                        elif 'ia3_l_1' in name:
                            target_key = name.replace('ia3_l_1', 'ia3_l')
                        local_state_dict[target_key] = server_state_dict[name]
                    
                    server_state_dict = local_state_dict
                
                with torch.no_grad():
                    model.load_state_dict(server_state_dict, strict=False)
                    
                if training_args.mode in ['fedours', 'ours_generator', 'ours_generator2']:
                    local_state_dict = {}
                    for name in server_state_dict.keys():
                        if 'lora1' in name:
                            target_key = name.replace('lora1', 'lora2')
                        elif 'ia3_l_1' in name:
                            target_key = name.replace('ia3_l_1', 'ia3_l_2')
                        local_state_dict[target_key] = server_state_dict[name]
                    
                    model.load_state_dict(local_state_dict, strict=False)
            
            
            global_state_dict = get_peft_state_maybe_zero_3(
                        model.named_parameters(), training_args.lora_bias
                    )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )
            global_state_dict.update(non_lora_state_dict)
            
            local_state_dict_list.append(copy.deepcopy(global_state_dict))
            old_local_state_dict_list.append(copy.deepcopy(global_state_dict))
            new_global_state_dict=copy.deepcopy(global_state_dict)
            keys_to_del = get_keys_to_del(training_args, new_global_state_dict)
            for k in keys_to_del:
                del new_global_state_dict[k]
            global_state_dict_list.append(new_global_state_dict)
            
            model_list[model_id] = global_state_dict
            
            models[model_id] = model
            
            model_ids[model_id] = [client_id]
            
    del model_list
    extra_state_dict_dict = {'model_ids':model_ids}
    
    ##############################################################################################
    # model keys: thkim0305/llama3.2_3B_vl, thkim0305/llama3.2_1B_vl, thkim0305/llama3.1_8B_vl
    model2 = models["thkim0305/llama3.2_1B_vl"]
    # model2 = models['thkim0305/qwen2.5_3B_vl']
    model = models["thkim0305/llama3.2_3B_vl"]
    
    model = model.to(device='cuda', dtype=torch.bfloat16)
    model2 = model2.to(device='cuda', dtype=torch.bfloat16)
    
    # load model
    # 0 6 8 9 vs 1 2 3 4 5 7
    # model_round = 20
    # model1_client = 6
    # model2_client = 7
    # # state_dict1 = torch.load(f'./client_states_feddualMultipqfullfreeze_CC_T05_bs4_saveoptim_lr2e-5_1e-4_sc132_4tasks_5rounds_fixitr100_T0125_decay099/{model1_client}_client_model_round{model_round}.pth', map_location='cpu')
    # # state_dict2 = torch.load(f'./client_states_feddualMultipqfullfreeze_CC_T05_bs4_saveoptim_lr2e-5_1e-4_sc132_4tasks_5rounds_fixitr100_T0125_decay099/{model2_client}_client_model_round{model_round}.pth', map_location='cpu')
    # state_dict1 = torch.load(f'./client_states_feddualMultipqfullfreeze_pca_T05_bs4_saveoptim_lr2e-5_1e-4_sc132_4tasks_5rounds_fixitr100_T0125_decay099/{model1_client}_client_model_round{model_round}.pth', map_location='cpu')
    # state_dict2 = torch.load(f'./client_states_feddualMultipqfullfreeze_pca_T05_bs4_saveoptim_lr2e-5_1e-4_sc132_4tasks_5rounds_fixitr100_T0125_decay099/{model2_client}_client_model_round{model_round}.pth', map_location='cpu')
    # # state_dict1 = torch.load(f'./client_states_fedMultipqfullfreeze_sft_bs4_saveoptim_lr2e-5_sc132_4tasks_5rounds_fixitr100_T0125_decay099/{model1_client}_client_model_round{model_round}.pth', map_location='cpu')
    # # state_dict2 = torch.load(f'./client_states_fedMultipqfullfreeze_sft_bs4_saveoptim_lr2e-5_sc132_4tasks_5rounds_fixitr100_T0125_decay099/{model2_client}_client_model_round{model_round}.pth', map_location='cpu')
    
    # with torch.no_grad():
    #     model.load_state_dict(state_dict2, strict=False)
    #     model2.load_state_dict(state_dict1, strict=False)
    
    # data_path = "dataset/llava_finetune/llava_v1_5_mix665k.json"
    data_path = "dataset/combined_data2.json"
    #  "/disk1/thkim/FederatedCL/dataset/llava_dataset/llava_finetune/llava_v1_5_mix665k.json"
    public_datalist = json.load(open(data_path, "r"))
    
    # Filter out items without the "image" key
    public_datalist = [item for item in public_datalist if "image" in item]
    
    random.shuffle(public_datalist)
    
    ##### B init #####
    public_datalist_ = public_datalist[:]
    
    data_module = make_supervised_data_module(client_data=public_datalist_, # sub_dataset
                                                tokenizer=tokenizer,
                                                processor=processor,
                                                data_args=copy.deepcopy(new_data_args))
    
    trainer = cka_create_trainer(model, tokenizer, training_args, data_module, model2)

    results = trainer.train()
    
    hidden_feat_1b = trainer.hidden_feat_1b
    hidden_feat_3b = trainer.hidden_feat_3b
    
    device = model.device
    
    trainer.deepspeed.empty_partition_cache()
    trainer.accelerator.free_memory()
    del model, model2, trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    cka_matrix = torch.zeros(len(hidden_feat_1b), len(hidden_feat_3b))
    
    for i, feat_1b in enumerate(hidden_feat_1b):
        for j, feat_3b in enumerate(hidden_feat_3b):
            feat_1b_gpu, feat_3b_gpu = feat_1b.to(device), feat_3b.to(device)
            rdm1 = compute_rdm(feat_1b_gpu, metric='correlation')
            rdm2 = compute_rdm(feat_3b_gpu, metric='correlation')
            cka_matrix[i,j] = rdm_similarity(rdm1.float(), rdm2.float())

    plot_cka(cka_matrix=cka_matrix, 
        first_layers=[str(i) for i in range(len(hidden_feat_1b))],
        second_layers=[str(i) for i in range(len(hidden_feat_3b))],
        first_name="1B",
        second_name="3B",
        save_path = 'rsa_plots',
        # title=f'feddualMultipqfullfreeze_round{model_round}_client{model1_client}_{model2_client}'
        # title=f'blockwise_dual_gradsim_round{model_round}_client{model1_client}_{model2_client}_otherdata'
        title='1B vs 3B unseen'
        )
    return
    ################################################################################################

def make_supervised_data_module(client_data, tokenizer: transformers.PreTrainedTokenizer, processor,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(client_data, tokenizer, data_args, processor)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def get_datalists(args, scenario_num):
    with open(f"./scenarios/scenario-{scenario_num}.json") as fp:
        scenario = json.load(fp)
    assert args.num_clients == len(scenario)

    train_datalists = {}
    test_datalists = {}
    
    max_iterations = args.num_iter
    rounds_per_task = args.num_rounds

    for client_data in scenario:
        client_id = client_data['client_id']
        train_datalist = []
        test_datalist = []
        for task_id, data in enumerate(client_data['datasets']):
            with open(f"./dataset/{data['dataset']}/train/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            random.shuffle(datalist)
            samplenum_per_rounds = int(len(datalist) / rounds_per_task)
            num_iter = max_iterations #max(int(max_iterations*samplenum_per_rounds/2000), 2) # 10000 / 5 = 2000
            for i in range(rounds_per_task):
                train_datalist.append(
                    {'datalist':datalist[i*samplenum_per_rounds:(i+1)*samplenum_per_rounds],
                     'num_iter': num_iter,
                     'task_id': task_id,
                     'model_id': client_data['model_id']})
            with open(f"./dataset/{data['dataset']}/test/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            test_datalist.append({
                "data_name": f"{data['dataset']}-{data['subset_id']}",
                "type": data['type'],
                "data": datalist,
                "train_start_round": rounds_per_task*task_id})
            
            train_datalists[client_id] = train_datalist
        test_datalists[client_id] = test_datalist

    return train_datalists, test_datalists

"""Utilities for plotting the CKA matrices."""

import matplotlib.pyplot as plt
import seaborn as sn

def plot_cka(
    cka_matrix: torch.Tensor,
    first_layers: list[str],
    second_layers: list[str],
    first_name: str = "First Model",
    second_name: str = "Second Model",
    save_path: str | None = None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "magma",
    show_ticks_labels: bool = False,
    short_tick_labels_splits: int | None = None,
    use_tight_layout: bool = True,
    show_annotations: bool = False,
    show_img: bool = False,
    show_half_heatmap: bool = False,
    invert_y_axis: bool = True,
    highlight_row_max:bool = True,
    highlight_col_max:bool = True,
) -> None:
    """Plot the CKA matrix obtained by calling CKA class __call__() method.

    Args:
        cka_matrix: the CKA matrix.
        first_layers: list of the names of the first model's layers.
        second_layers: list of the names of the second model's layers.
        first_name: name of the first model (default='First Model').
        second_name: name of the second model (default='Second Model').
        save_path: the path where to save the plot, if None then the plot will not be saved (default=None).
        title: the plot title, if None then a simple text with the name of both models will be used (default=None).
        vmin: values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.
        vmax: values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.
        cmap: the name of the colormap to use (default: 'magma').
        show_ticks_labels: whether to show the tick labels (default=False).
        short_tick_labels_splits: only works when show_tick_labels is True. If it is not None, the tick labels will
            be shortened to the defined sublayer starting from the deepest level. E.g.: if the layer name is
            'encoder.ff.linear' and this parameter is set to 1, then only 'linear' will be printed on the heatmap
            (default=None).
        use_tight_layout: whether to use a tight layout in order not to cut any label in the plot (default=True).
        show_annotations: whether to show the annotations on the heatmap (default=True).
        show_img: whether to show the plot (default=True).
        show_half_heatmap: whether to mask the upper left part of the heatmap since those valued are duplicates
            (default=False).
        invert_y_axis: whether to invert the y-axis of the plot (default=True).

    Raises:
        ValueError: if ``vmax`` or ``vmin`` are not defined together or both equal to None.
    """
    # Deal with vmin and vmax
    if (vmin is not None) ^ (vmax is not None):
        raise ValueError("'vmin' and 'vmax' must be defined together or both equal to None.")

    vmin = min(vmin, torch.min(cka_matrix).item()) if vmin is not None else vmin
    vmax = max(vmax, torch.max(cka_matrix).item()) if vmax is not None else vmax

    # Build the mask
    mask = torch.tril(torch.ones_like(cka_matrix, dtype=torch.bool), diagonal=-1) if show_half_heatmap else None

    # Build the heatmap
    if mask is not None:
        ax = sn.heatmap(cka_matrix.cpu(), vmin=vmin, vmax=vmax, annot=show_annotations, cmap=cmap, mask=mask.cpu().numpy())
    else:
        ax = sn.heatmap(cka_matrix.cpu(), vmin=vmin, vmax=vmax, annot=show_annotations, cmap=cmap)
    if invert_y_axis:
        ax.invert_yaxis()

    ax.set_xlabel(f"{second_name} layers", fontsize=12)
    ax.set_ylabel(f"{first_name} layers", fontsize=12)

    # Deal with tick labels
    if show_ticks_labels:
        if short_tick_labels_splits is None:
            ax.set_xticklabels(second_name)
            ax.set_yticklabels(first_name)
        else:
            ax.set_xticklabels(["-".join(module.split(".")[-short_tick_labels_splits:]) for module in second_layers])
            ax.set_yticklabels(["-".join(module.split(".")[-short_tick_labels_splits:]) for module in first_layers])

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

    # Put the title if passed
    if title is not None:
        ax.set_title(title, fontsize=14)
    else:
        title = f"{first_name} vs {second_name}"
        ax.set_title(title, fontsize=14)

    # ========== NEW SECTION: Highlight row/column maxima ==========
    # Because the heatmap uses matrix indexing:
    #   row -> y
    #   col -> x
    # By default, row=0 is on top. If `invert_y_axis` is used, row=0 is at the bottom visually.
    # We'll show two ways: coordinate approach with or without inverting y. 

    n_rows, n_cols = cka_matrix.shape
    cka_cpu = cka_matrix.cpu()

    # 1) Highlight the highest value in each row
    if highlight_row_max:
        for i in range(n_rows):
            # Column index of max in row i
            j = torch.argmax(cka_cpu[i, :]).item()
            # Because the heatmap is drawn with top row = index 0 at the top
            # If we invert_yaxis(), row i is visually at "n_rows - 1 - i"
            # but the coordinate system for scatter is still "i" from the top,
            # so we *do not* offset it further if we used `ax.invert_yaxis()`.
            # We just do: (x=j + 0.5, y=i + 0.5) for the center of the cell.
            #
            # You can change marker style/color below:
            ax.scatter(
                j + 0.5,
                i + 0.5,  # remains i+0.5 even if we invert the axis
                s=120,
                marker='o',
                edgecolors='white',
                facecolors='none',
                linewidths=2
            )

    # 2) Highlight the highest value in each column
    if highlight_col_max:
        for j in range(n_cols):
            # Row index of max in column j
            i = torch.argmax(cka_cpu[:, j]).item()
            # Same note about invert_yaxis applies
            ax.scatter(
                j + 0.5,
                i + 0.5,
                s=120,
                marker='*',
                edgecolors='yellow',
                facecolors='none',
                linewidths=2
            )
    # ========== END NEW SECTION ==========

    
    # Set the layout to tight if the corresponding parameter is True
    if use_tight_layout:
        plt.tight_layout()

    # Save the plot to the specified path if defined
    if save_path is not None:
        title = title.replace("/", "-")
        path_rel = f"{save_path}/{title}.png"
        plt.savefig(path_rel, dpi=400, bbox_inches="tight")

    # Show the image if the user chooses to do so
    if show_img:
        plt.show()

if __name__ == "__main__":
    main()