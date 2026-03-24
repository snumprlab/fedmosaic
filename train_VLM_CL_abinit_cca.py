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

from federated_methods.AB_init import ABInit_create_trainer
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
            keys_to_del = get_keys_to_del(training_args, global_state_dict, data_args)
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
            keys_to_del = get_keys_to_del(training_args, new_global_state_dict, data_args)
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
    # llm models: meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B-Instruct
    model2 = models["thkim0305/llama3.2_1B_vl"]
    # model2 = models['thkim0305/qwen2.5_0.5B_vl']
    # model2 = models["meta-llama/Llama-3.2-1B-Instruct"]
    
    # data_path = "/disk1/thkim/FederatedCL/dataset/llava_dataset/llava_finetune/llava_v1_5_mix665k.json"
    #  "/disk1/thkim/FederatedCL/dataset/llava_dataset/llava_finetune/llava_v1_5_mix665k.json"
    # data_path = 'chatbotIT.json'
    data_path = 'dataset/llava_finetune/llava_v1_5_mix665k_updated.json'
    public_datalist = json.load(open(data_path, "r"))
    
    # Filter out items without the "image" key
    public_datalist = [item for item in public_datalist if "image" in item]
    
    random.shuffle(public_datalist)

    ##### A_PCA init #####
    public_datalist_ = public_datalist[:200]
    
    data_module = make_supervised_data_module(client_data=public_datalist_, # sub_dataset
                                                tokenizer=tokenizer,
                                                processor=processor,
                                                data_args=copy.deepcopy(new_data_args))
    
    # train bigger model
    model = models["thkim0305/llama3.2_3B_vl"]
    # model = models['thkim0305/qwen2.5_3B_vl']
    # model = models["thkim0305/llama3.1_8B_vl"]
    # model = models["meta-llama/Llama-3.2-3B-Instruct"]
    from federated_methods.A_init_PCA import A_PCA_Init_create_trainer
    trainer = A_PCA_Init_create_trainer(model, tokenizer, training_args, data_module, model2, data_args, train_A = True)

    results = trainer.train()

    lora_A_input_1bs = trainer.lora_A_input_1b
    lora_A_input_3bs = trainer.lora_A_input_3b
    
    layer_name_1b = trainer.layer_name_1b
    layer_name_3b = trainer.layer_name_3b
    
    state_dict = get_peft_state_maybe_zero_3(
        model.named_parameters(), training_args.lora_bias
    )
    
    state_dict2 = get_peft_state_maybe_zero_3(
        model2.named_parameters(), training_args.lora_bias
    )
    lora_r = 128
    for idx, (lora_A_input_1b, lora_A_input_3b) in enumerate(zip(lora_A_input_1bs, lora_A_input_3bs)):
        _, _, V_1b = torch.pca_lowrank(lora_A_input_1b.float(), q=lora_r)
        _, _, V_3b = torch.pca_lowrank(lora_A_input_3b.float(), q=lora_r)

        with torch.no_grad():
            print("state_dict[layer_name_3b[idx]]", state_dict[layer_name_3b[idx]].shape, "V_3b[:, :lora_r]", V_3b[:, :lora_r].shape)
            print("state_dict2[layer_name_1b[idx]]", state_dict2[layer_name_1b[idx]].shape, "V_1b[:, :lora_r]", V_1b[:, :lora_r].shape)
            state_dict[layer_name_3b[idx]] = V_3b[:, :lora_r].to(torch.bfloat16).T
            state_dict2[layer_name_1b[idx]] = V_1b[:, :lora_r].to(torch.bfloat16).T


    output_dir2 = os.path.join(training_args.state_dir, f"llava_1b_PCA_orthnormal_init.pth")
    # output_dir2 = os.path.join(training_args.state_dir, f"qwen_0.5b_PCA_orthnormal_init.pth")
    torch.save(state_dict2, output_dir2)
    
    output_dir = os.path.join(training_args.state_dir, f"llava_3b_PCA_orthnormal_init.pth")
    # output_dir = os.path.join(training_args.state_dir, f"qwen_3b_PCA_orthnormal_init.pth")
    torch.save(state_dict, output_dir)

    trainer.deepspeed.empty_partition_cache()
    trainer.accelerator.free_memory()
    del trainer, lora_A_input_1bs, lora_A_input_3bs
    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    ##### A init (using CCA, not training) #####
    model.load_state_dict(state_dict, strict=False) 
    
    # load pretrained 1b weight
    # state_dict2 = torch.load('llava_1b_blockwise_pca_init.pth', map_location='cpu')
    # state_dict2 = torch.load('llava_1b_blockwise_half_pca_init.pth', map_location='cpu')
    
    model2.load_state_dict(state_dict2, strict=False) 

    public_datalist_ = public_datalist[2000:2100]
    data_module = make_supervised_data_module(client_data=public_datalist_, # sub_dataset
                                                tokenizer=tokenizer,
                                                processor=processor,
                                                data_args=copy.deepcopy(new_data_args))
    
    trainer = A_PCA_Init_create_trainer(model, tokenizer, training_args, data_module, model2, data_args, train_A = True, take_input=False)

    results = trainer.train()

    lora_A_output_1bs = trainer.lora_A_input_1b
    lora_A_output_3bs = trainer.lora_A_input_3b
    
    layer_name_1b = trainer.layer_name_1b
    layer_name_3b = trainer.layer_name_3b
    
    state_dict = get_peft_state_maybe_zero_3(
        model.named_parameters(), training_args.lora_bias
    )
    
    state_dict2 = get_peft_state_maybe_zero_3(
        model2.named_parameters(), training_args.lora_bias
    )


    gamma=1e-4
    
    def matrix_inv_sqrt(S):
        S_U, S_S, S_Vt = torch.linalg.svd(S)
        S_neg_sqrt = S_U @ torch.diag(S_S**(-1/2)) @ S_Vt
        S_sqrt = S_U @ torch.diag(S_S**(1/2)) @ S_Vt
        return S_neg_sqrt, S_sqrt
    
    for idx, (X1, X3) in enumerate(zip(lora_A_output_1bs, lora_A_output_3bs)):
        X1 = X1.cuda()
        X3 = X3.cuda()
        X1_centered = X1 - X1.mean(dim=0, keepdim=True)
        X3_centered = X3 - X3.mean(dim=0, keepdim=True)
        
        S11 = X1_centered.t() @ X1_centered + gamma*torch.eye(X1.shape[1]).cuda() # shape: [d1, d1]
        S33 = X3_centered.t() @ X3_centered + gamma*torch.eye(X3.shape[1]).cuda() # shape: [d3, d3]
        # S13 = X1_centered.t() @ X3_centered  # shape: [d1, d3]
        S31 = X3_centered.t() @ X1_centered

        S11_neg_sqrt, S11_sqrt = matrix_inv_sqrt(S11.to(torch.float32))
        S33_neg_sqrt, S33_sqrt = matrix_inv_sqrt(S33.to(torch.float32))
        
        # M = S11_neg_sqrt @ S13.to(torch.float32) @ S33_neg_sqrt
        M = S33_neg_sqrt @ S31.to(torch.float32) @ S11_neg_sqrt
        # M will have shape [d1, d3].
        # Perform SVD:  M = U * Sigma * V^T
        U, Sigma, Vt = torch.linalg.svd(M, full_matrices=False)
        
        # print(S33_neg_sqrt.shape, Vt.T.shape, U.T.shape, S11_sqrt.T.shape)
        
        # P1 = S11_inv_sqrt @ U  # shape: [d1, d1]
        # P3 = S33_inv_sqrt @ Vt.mT  # shape: [d3, d3]
        
        with torch.no_grad():
            # mapping_mat = (S11_neg_sqrt @ Vt.T @ U.T @ S33_sqrt.T).T
            # state_dict[layer_name_3b[idx]] = (mapping_mat @ state_dict2[layer_name_1b[idx]].to(torch.float32).cuda()).to(torch.bfloat16).detach().cpu()
            
            state_dict[layer_name_3b[idx]] = ((S33_neg_sqrt @ U.T).T @ state_dict[layer_name_3b[idx]].to(torch.float32).cuda()).to(torch.bfloat16).detach().cpu()
            state_dict2[layer_name_1b[idx]] = ((S11_neg_sqrt @ Vt.T).T @ state_dict2[layer_name_1b[idx]].to(torch.float32).cuda()).to(torch.bfloat16).detach().cpu()
        
    output_dir = os.path.join(training_args.state_dir, f"llava_3b_orthnormal_init_CCA_A.pth")
    # output_dir = os.path.join(training_args.state_dir, f"qwen_3b_orthnormal_init_FT_AB.pth")
    torch.save(state_dict, output_dir)
    
    output_dir = os.path.join(training_args.state_dir, f"llava_1b_orthnormal_init_CCA_A.pth")
    # output_dir = os.path.join(training_args.state_dir, f"qwen_3b_orthnormal_init_FT_AB.pth")
    torch.save(state_dict2, output_dir)
    
    model.load_state_dict(state_dict, strict=False) 
    model2.load_state_dict(state_dict2, strict=False)

    ##### B init #####
    public_datalist_ = public_datalist[7000:7100]
    # public_datalist_ = public_datalist[7000:7080]
    data_module = make_supervised_data_module(client_data=public_datalist_, # sub_dataset
                                                tokenizer=tokenizer,
                                                processor=processor,
                                                data_args=copy.deepcopy(new_data_args))
    
    trainer = ABInit_create_trainer(model, tokenizer, training_args, data_module, model2, data_args, train_A = False)

    results = trainer.train()
    
    lora_B_output_1b = trainer.lora_B_output_1b
    lora_B_output_3b = trainer.lora_B_output_3b
    
    layer_name_1b = trainer.layer_name_1b
    layer_name_3b = trainer.layer_name_3b
    
    state_dict = get_peft_state_maybe_zero_3(
        model.named_parameters(), training_args.lora_bias
    )
    
    state_dict2 = get_peft_state_maybe_zero_3(
        model2.named_parameters(), training_args.lora_bias
    )


    gamma=1e-4
    
    def matrix_inv_sqrt(S):
        S_U, S_S, S_Vt = torch.linalg.svd(S)
        S_neg_sqrt = S_U @ torch.diag(S_S**(-1/2)) @ S_Vt
        S_sqrt = S_U @ torch.diag(S_S**(1/2)) @ S_Vt
        return S_neg_sqrt, S_sqrt
    
    for idx, (X1, X3) in enumerate(zip(lora_B_output_1b, lora_B_output_3b)):
        X1 = X1.cuda()
        X3 = X3.cuda()
        X1_centered = X1 - X1.mean(dim=0, keepdim=True)
        X3_centered = X3 - X3.mean(dim=0, keepdim=True)
        
        S11 = X1_centered.t() @ X1_centered + gamma*torch.eye(X1.shape[1]).cuda() # shape: [d1, d1]
        S33 = X3_centered.t() @ X3_centered + gamma*torch.eye(X3.shape[1]).cuda() # shape: [d3, d3]
        # S13 = X1_centered.t() @ X3_centered  # shape: [d1, d3]
        S31 = X3_centered.t() @ X1_centered

        S11_neg_sqrt, S11_sqrt = matrix_inv_sqrt(S11.to(torch.float32))
        S33_neg_sqrt, S33_sqrt = matrix_inv_sqrt(S33.to(torch.float32))
        
        # M = S11_neg_sqrt @ S13.to(torch.float32) @ S33_neg_sqrt
        M = S33_neg_sqrt @ S31.to(torch.float32) @ S11_neg_sqrt
        # M will have shape [d1, d3].
        # Perform SVD:  M = U * Sigma * V^T
        U, Sigma, Vt = torch.linalg.svd(M, full_matrices=False)
        
        # print(S33_neg_sqrt.shape, Vt.T.shape, U.T.shape, S11_sqrt.T.shape)
        
        # P1 = S11_inv_sqrt @ U  # shape: [d1, d1]
        # P3 = S33_inv_sqrt @ Vt.mT  # shape: [d3, d3]
        
        # for coeff in [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.0,10.0,100.0]:
        #     try:
        #         P1_pinv = torch.linalg.pinv(P1 + (torch.eye(P1.shape[0])*coeff).cuda())  # shape [k, d1]
        #         break
        #     except Exception as e:
        #         print(coeff, e)
        with torch.no_grad():
            # mapping_mat = (S33_neg_sqrt @ Vt.T @ U.T @ S11_sqrt.T).T
            mapping_mat = (S11_neg_sqrt @ Vt.T @ U.T @ S33_sqrt.T).T
            state_dict[layer_name_3b[idx]] = (mapping_mat @ state_dict2[layer_name_1b[idx]].to(torch.float32).cuda()).to(torch.bfloat16).detach().cpu()
            
    for key in state_dict.keys():
        if 'lora_P' in key:
            state_dict[key] = torch.zeros_like(state_dict[key])
    
    for key in state_dict2.keys():
        if 'lora_P' in key:
            state_dict2[key] = torch.zeros_like(state_dict2[key])
    
    output_dir2 = os.path.join(training_args.state_dir, f"llava_1b_final_init.pth")
    
    torch.save(state_dict2, output_dir2)
    
    output_dir = os.path.join(training_args.state_dir, f"llava_3b_final_init.pth")
    # output_dir = os.path.join(training_args.state_dir, f"qwen_3b_orthnormal_init_FT_AB.pth")
    torch.save(state_dict, output_dir)
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

if __name__ == "__main__":
    main()