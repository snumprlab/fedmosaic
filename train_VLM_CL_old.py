import logging.config
import os
import random
import gc

import numpy as np
import torch
from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingConfig
import transformers
from utils.train_utils import get_VLMmodel, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, get_task_vectors, load_deepspeed, configure_online_datastream, get_keys_to_del, make_supervised_data_module

from federated_methods.method_manager import select_method
from utils.data_loader_VLM import LazySupervisedDataset, DataCollatorForSupervisedDataset
from typing import Dict

import copy
import json
from transformers import BitsAndBytesConfig
import time
import datetime
import torch.nn.functional as F

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
    if training_args.mode in ['perada', 'feddat']:
        training_args.gradient_accumulation_steps = training_args.gradient_accumulation_steps//2
    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)
    torch.cuda.manual_seed_all(training_args.seed)

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
    
    train_datalists, test_datalists = get_datalists(training_args, training_args.scenario)
    
    # select functions
    set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules = select_method(training_args.mode)
    
    # create folder
    training_args.state_dir = training_args.state_dir + '_' + training_args.note
    if not os.path.exists(training_args.state_dir):
        os.makedirs(training_args.state_dir)
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}
    
    LAYER_INDEX = 5 if data_args.is_multimodal else 4
    
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
                # server_state_dict = torch.load(training_args.load_checkpoint, map_location='cpu')
                load_round = int(training_args.load_checkpoint.split('round')[-1][:-4])+1
                load_dir = training_args.load_checkpoint.split('/')[0]
                prev_local_state_dict_list = []
                for local_id in range(10):
                    prev_local_state_dict_list.append(torch.load(f"{load_dir}/{local_id}_client_model_round{load_round}.pth", map_location='cpu'))
                
                state_dict = get_peft_state_maybe_zero_3(
                        model.named_parameters(), training_args.lora_bias
                    )
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    model.named_parameters()
                )
                state_dict.update(non_lora_state_dict)
                
                keys_to_del = get_keys_to_del(training_args, state_dict, data_args)
                for k in keys_to_del:
                    del state_dict[k]
                global_state_dict = state_dict
                
                new_global_state_dict = {}
            
                weights = torch.ones(10)
                sim_sum = weights.sum() #- weights[client_id]
            
                # # weights[client_id] = sim_sum
                # # sim_sum += sim_sum
                cur_layer_num = []
                for k in global_state_dict.keys():
                    if 'layers.' in k:
                        cur_layer_num.append(int(k.split('.')[LAYER_INDEX]))
                cur_layer_num = sorted(list(set(cur_layer_num)))
                
                for name in global_state_dict.keys():
                    new_param = 0
                    target_key = name
                    
                    for id in range(len(prev_local_state_dict_list)):
                        # if id == client_id:
                        #     continue
                        # else:
                        splited = target_key.split('.')
                        # if layer number is different
                        layer_num = []
                        for k in prev_local_state_dict_list[id].keys():
                            if 'layers.' in k:
                                layer_num.append(int(k.split('.')[LAYER_INDEX]))
                        layer_num = len(set(layer_num)) // 4
                        
                        target_layers = [layer_num*1 -1,layer_num*2 -1,layer_num*3 -1,layer_num*4 -1]
                        if cur_layer_num[-1] != target_layers[-1]: # if different size
                            if int(splited[5]) == cur_layer_num[0]: # mid layer
                                splited[5] = str(target_layers[0])
                            elif int(splited[5]) == cur_layer_num[1]: # last layer 
                                splited[5] = str(target_layers[1])
                            elif int(splited[5]) == cur_layer_num[2]: # last layer 
                                splited[5] = str(target_layers[2])
                            elif int(splited[5]) == cur_layer_num[3]: # last layer 
                                splited[5] = str(target_layers[3])
                            new_target_key = '.'.join(splited)
                        else:
                            new_target_key = target_key
                        new_param += weights[id]*prev_local_state_dict_list[id][new_target_key] / sim_sum
                        
                    new_global_state_dict[name] = new_param
                    
                    new_global_state_dict[name] = new_param
                
                if 'zero3' in training_args.deepspeed:
                    load_deepspeed(new_global_state_dict, model, strict=False)
                else:
                    model.load_state_dict(new_global_state_dict, strict=False) 
                del state_dict
            
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
    
    if data_args.is_multimodal and training_args.use_task_vector and 'thkim0305/llama3.2_1B_vl' not in models.keys():
        new_model_args = copy.deepcopy(model_args)
        new_model_args.model_name_or_path = 'thkim0305/llama3.2_1B_vl'
        model2, _,_,_ = get_VLMmodel(new_model_args, training_args, bnb_model_from_pretrained_args, data_args)
        models['thkim0305/llama3.2_1B_vl'] = model2
    elif not data_args.is_multimodal and training_args.use_task_vector and 'meta-llama/Llama-3.2-1B' not in models.keys():
        new_model_args = copy.deepcopy(model_args)
        new_model_args.model_name_or_path = 'meta-llama/Llama-3.2-1B'
        model2, _,_,_ = get_VLMmodel(new_model_args, training_args, bnb_model_from_pretrained_args, data_args)
        models['meta-llama/Llama-3.2-1B'] = model2
    
    del model_list
    extra_state_dict_dict = {'model_ids':model_ids, 'models':models}
    extra_state_dict_dict['LAYER_INDEX'] = LAYER_INDEX
    
    if training_args.fedours:
        logger.info(f'load task vector {training_args.load_checkpoint}')
        tv_weights = torch.load(training_args.load_checkpoint, map_location='cpu')
        prev_task_vectors = tv_weights['task_vectors']
        prev_local_state_dict_list = tv_weights['local_state_dict_list']
        
        current_task_vectors = get_task_vectors(model, tokenizer, processor, train_datalists, training_args, data_args, global_state_dict_list, make_supervised_data_module, models['thkim0305/llama3.2_1B_vl'])
    else:
        current_task_vectors = None

    training_loss = [[] for i in range(training_args.num_clients)]
    
    # start federated learning
    start_time = time.time()
    frac_clients = 1
    
    memory = [[] for id in range(training_args.num_clients)]
    memory_count = [np.array([]) for id in range(training_args.num_clients)]
    memory_size = training_args.memory_size
    total_batchsize = training_args.per_gpu_train_batch_size*training_args.world_size*training_args.gradient_accumulation_steps
    init_lr = training_args.learning_rate
    mm_init_lr = training_args.mm_projector_lr
    final_lr = training_args.final_lr
    mm_final_lr = training_args.mm_final_lr
    
    total_rounds = training_args.num_rounds * training_args.num_tasks
    last_task_id = [-1 for _ in range(training_args.num_clients)]
    fisher_olds = [None for _ in range(training_args.num_clients)]
    task_vectors = [None for _ in range(training_args.num_clients)]
    
    original_weights = {}
    if 'tv' in training_args.mode:
        if 'pq' in training_args.mode:
            for n,p in model.base_model.language_model.model.layers[-1].named_parameters():
                if 'lora2_P' in n or 'lora2_Q' in n or 'lora_P' in n or 'lora_Q' in n:
                    original_weights[n] = p.clone().detach().cpu().flatten()
        else:
            for n,p in model.base_model.language_model.model.layers[-1].named_parameters():
                if 'lora' in n:
                    original_weights[n] = p.clone().detach().cpu().flatten()
    lr_step = (init_lr - final_lr)/total_rounds
    mm_lr_step = (mm_init_lr - mm_final_lr)/total_rounds
    for curr_round in range(total_rounds):
        old_local_state_dict_list = [copy.deepcopy(local_state_dict_list[i]) for i in range(len(local_state_dict_list))]
        
        if curr_round > 0 and training_args.use_task_vector:
            path = os.path.join(training_args.state_dir, f"round{curr_round}_task_vector_local_weights.pth")
            tv_weight = {'task_vectors': task_vectors}#, 'local_state_dict_list': old_local_state_dict_list}
            torch.save(tv_weight, path)
            
            # task vector layerwise cosine sim
            if 'tv' in training_args.mode:
                sims = []
                for layer_name in task_vectors[0].keys():
                    task_vector = F.normalize(torch.stack([task_vectors[i][layer_name] for i in range(len(task_vectors))], dim=0), dim=-1)
                    sim = torch.matmul(task_vector,
                                    torch.transpose(task_vector, 1, 0))
                    if sim.sum() != 0:
                        sim = torch.transpose(sim, 1, 0)
                        sims.append(sim)
                sim = torch.stack(sims, dim=0).mean(dim=0)
            elif 'excludemean' in training_args.mode:
                sim = torch.ones(training_args.num_clients, training_args.num_clients)
            else:
                # vectorize cosine sim and then average them
                sims = []
                if 'pqgrad' in training_args.mode or 'pqfisher' in training_args.mode:
                    for grad_idx in range(len(task_vectors[0])):
                        task_vector = F.normalize(torch.stack([tv[grad_idx] for tv in task_vectors], dim=0), dim=-1)
                        sim = torch.matmul(task_vector,
                                        torch.transpose(task_vector, 1, 0))
                        sim = torch.transpose(sim, 1, 0)
                        sims.append(sim)
                
                else:
                    for grad_idx in range(task_vectors[0].shape[-1]):
                        task_vector = F.normalize(torch.stack([tv[:,grad_idx] for tv in task_vectors], dim=0), dim=-1)
                        sim = torch.matmul(task_vector,
                                        torch.transpose(task_vector, 1, 0))
                        sim = torch.transpose(sim, 1, 0)
                        sims.append(sim)
                
                sim = torch.stack(sims, dim=0).mean(dim=0)
            # sim = torch.ones(10,10)
            
            extra_state_dict_dict['task_similarity'] = sim
            print("task similarity matrix:")
            print(sim)
        
        # clients turn
        cids = np.arange(training_args.num_clients).tolist()
        num_selection = int(round(training_args.num_clients*frac_clients)) 
        selected_ids = sorted(random.sample(cids, num_selection)) 
        if training_args.local_rank == 0 or training_args.local_rank == -1: 
            logger.info(f"Round {curr_round} | selected_ids: {selected_ids}\n")
        
        # selected_ids = cids
        training_args.learning_rate = init_lr - lr_step*curr_round
        training_args.mm_projector_lr = mm_init_lr - mm_lr_step*curr_round
        if curr_round > 0 and training_args.is_wsd:
            training_args.warmup_ratio = 0
            training_args.warmup_steps = 0
        for idx in range(num_selection):
            client_id = selected_ids[idx]
            
            ##### simulate online memory insertion & get_batch ####
            sub_dataset = train_datalists[client_id][curr_round]['datalist']
            num_iterations = train_datalists[client_id][curr_round]['num_iter']
            
            task_id = train_datalists[client_id][curr_round]['task_id']
            
            test_datalist = test_datalists[client_id]
            
            model_id = train_datalists[client_id][curr_round]['model_id']
            new_model_args = copy.deepcopy(model_args)
            new_model_args.model_name_or_path = model_id
            new_data_args = copy.deepcopy(data_args)
            new_data_args.model_name_for_dataarg = model_id
            # model,_,_,_ = get_VLMmodel(new_model_args, training_args, bnb_model_from_pretrained_args, new_data_args)
            model = models[model_id]
            
            extra_state_dict_dict['client_id'] = client_id
            extra_state_dict_dict['curr_round'] = curr_round
            extra_state_dict_dict['test_datalist'] = test_datalist
            extra_state_dict_dict['processor'] = processor
            extra_state_dict_dict['data_args'] = copy.deepcopy(new_data_args)
            extra_state_dict_dict['tokenizer'] = tokenizer
            if training_args.use_task_id:
                extra_state_dict_dict['task_id'] = task_id

            load_state_dict(model, global_state_dict_list[client_id], old_local_state_dict_list, client_id, training_args, extra_state_dict_dict)
            print('model loading done')
            
            if training_args.fedours:
                global_state_dict = global_state_dict_list[client_id]
                sims = []
                for grad_idx in range(prev_task_vectors[0].shape[-1]):
                    task_vector = F.normalize(torch.stack([tv[:,grad_idx] for tv in prev_task_vectors] + [current_task_vectors[client_id][:,grad_idx]], dim=0), dim=-1)
                    sim = torch.matmul(task_vector,
                                    torch.transpose(task_vector, 1, 0))
                    sim = torch.transpose(sim, 1, 0)
                    sims.append(sim)
                
                sim = torch.stack(sims, dim=0).mean(dim=0)
                
                print(sim)
                new_global_state_dict = {}
            
                weights = sim[-1][:-1].clone()
                
                weights = (weights/training_args.softmax_temp).softmax(dim=0)
                
                sim_sum = weights.sum() #- weights[client_id]
            
                # # weights[client_id] = sim_sum
                # # sim_sum += sim_sum
                cur_layer_num = []
                for k in global_state_dict.keys():
                    if 'layers.' in k:
                        cur_layer_num.append(int(k.split('.')[LAYER_INDEX]))
                cur_layer_num = sorted(list(set(cur_layer_num)))
                
                for name in global_state_dict.keys():
                    new_param = 0
                    if 'lora' in name:
                        target_key = name.replace('lora', 'lora2')
                    elif 'ia3_l' in name:
                        target_key = name.replace('ia3_l', 'ia3_l_2')
                    
                    for id in range(len(prev_local_state_dict_list)):
                        # if id == client_id:
                        #     continue
                        # else:
                        splited = target_key.split('.')
                        # if layer number is different
                        layer_num = []
                        for k in prev_local_state_dict_list[id].keys():
                            if 'layers.' in k:
                                layer_num.append(int(k.split('.')[LAYER_INDEX]))
                        layer_num = len(set(layer_num)) // 4
                        
                        target_layers = [layer_num*1 -1,layer_num*2 -1,layer_num*3 -1,layer_num*4 -1]
                        if cur_layer_num[-1] != target_layers[-1]: # if different size
                            if int(splited[5]) == cur_layer_num[0]: # mid layer
                                splited[5] = str(target_layers[0])
                            elif int(splited[5]) == cur_layer_num[1]: # last layer 
                                splited[5] = str(target_layers[1])
                            elif int(splited[5]) == cur_layer_num[2]: # last layer 
                                splited[5] = str(target_layers[2])
                            elif int(splited[5]) == cur_layer_num[3]: # last layer 
                                splited[5] = str(target_layers[3])
                            new_target_key = '.'.join(splited)
                        else:
                            new_target_key = target_key
                        new_param += weights[id]*prev_local_state_dict_list[id][new_target_key] / sim_sum
                    
                    new_global_state_dict[name] = new_param
                
                if 'zero3' in training_args.deepspeed:
                    load_deepspeed(new_global_state_dict, model, strict=False)
                else:
                    model.load_state_dict(new_global_state_dict, strict=False) 
            
            datalist = configure_online_datastream(sub_dataset, num_iterations, training_args, client_id, memory, memory_count, memory_size, total_batchsize)
            data_module = make_supervised_data_module(client_data=datalist, # sub_dataset
                                                tokenizer=tokenizer,
                                                processor=processor,
                                                data_args=copy.deepcopy(new_data_args))
            
            if training_args.local_rank == 0 or training_args.local_rank == -1: 
                logger.info(f'Round {curr_round} | train client {client_id} | num samples {len(sub_dataset)}')

            # ===== Train local model on the client side =====
            if training_args.use_fisher:
                extra_state_dict_dict['fisher_old'] = fisher_olds[client_id]
                
            if training_args.use_task_vector:
                extra_state_dict_dict['task_vector'] = task_vectors[client_id]
                extra_state_dict_dict['model2'] = models['thkim0305/llama3.2_1B_vl'] if data_args.is_multimodal else models['meta-llama/Llama-3.2-1B']
            
            trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict)

            results = trainer.train()
            training_loss[client_id].append(results.training_loss)
            if training_args.use_fisher:
                fisher_olds[client_id] = trainer.fisher_old
            
            if training_args.use_task_vector:
                if 'tv' in training_args.mode:
                    new_task_vectors={}
                    if 'pq' in training_args.mode:
                        for n,p in model.base_model.language_model.model.layers[-1].named_parameters():
                            if 'lora2_P' in n or 'lora2_Q' in n or 'lora_P' in n or 'lora_Q' in n:
                                new_task_vectors[n] = p.clone().detach().cpu().flatten() - original_weights[n]
                    else:
                        for n,p in model.base_model.language_model.model.layers[-1].named_parameters():
                            if 'lora' in n:
                                new_task_vectors[n] = p.clone().detach().cpu().flatten() - original_weights[n]
                    task_vectors[client_id] = new_task_vectors
                elif 'excludemean' in training_args.mode:
                    task_vectors[client_id] = torch.ones(1)
                else:
                    task_vectors[client_id] = trainer.task_vector
                    
                    if data_args.is_multimodal:
                        models['thkim0305/llama3.2_1B_vl'] = models['thkim0305/llama3.2_1B_vl'].cpu()
                    else:
                        models['meta-llama/Llama-3.2-1B'] = models['meta-llama/Llama-3.2-1B'].cpu()
            
            if training_args.local_rank == 0 or training_args.local_rank == -1: 
                path = os.path.join(training_args.state_dir, f"{client_id}_trainer_state.json")
                trainer.state.save_to_json(path)
            
            model.config.use_cache = True
            
            # save local model
            output_dir = os.path.join(training_args.state_dir, f"{client_id}_client_model_round{curr_round+1}.pth")

            state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), training_args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )
            state_dict.update(non_lora_state_dict)
            
            # if 'full' in training_args.mode:
            #     print("Determinant of P")
            #     for k in state_dict.keys():
            #         if 'lora_P' in k or 'lora1_P' in k or 'lora2_P' in k:
            #             square_matrix = state_dict[k].to(torch.float32)
            #             print(k, torch.det(square_matrix))
            
            if (training_args.local_rank == 0 or training_args.local_rank == -1):# and (curr_round+1)%(total_rounds/20) == 0:
                torch.save(state_dict, output_dir)
            
            if 'fedquad' in training_args.mode and not training_args.immediate_ema_update:
                for k, v in trainer.ema_module1.items():
                    state_dict[k] = v.detach().cpu()
                for k, v in trainer.ema_module2.items():
                    state_dict[k] = v.detach().cpu()
            
            local_state_dict_list[client_id] = copy.deepcopy(state_dict)
            
            # local_state_dict = getattr(trainer, 'global_weight', None)
            # if local_state_dict is not None:
            #     local_state_dict_list[client_id] = copy.deepcopy(local_state_dict)
            
            trainer.deepspeed.empty_partition_cache()
            trainer.accelerator.free_memory()
            del trainer
            model = model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"done Round {curr_round} client {client_id} | elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))} | ")
            
        
        aggregate_state_dict(global_state_dict_list, local_state_dict_list, selected_ids, num_selection, training_args, **extra_state_dict_dict)
        
        # Save server model
        # if training_args.mode != 'fedours':
        #     if (training_args.local_rank == 0 or training_args.local_rank == -1): 
        #         torch.save(global_state_dict_list[0], os.path.join(training_args.state_dir, f"server_model_round{curr_round}.pth"))
            
    if training_args.use_task_vector:
        path = os.path.join(training_args.state_dir, f"round{curr_round+1}_task_vector_local_weights.pth")
        tv_weight = {'task_vectors': task_vectors}#, 'local_state_dict_list': local_state_dict_list}
        torch.save(tv_weight, path)
        
        # task_vector = F.normalize(torch.stack(task_vectors, dim=0), dim=-1)
        # sim = torch.matmul(task_vector,
        #                 torch.transpose(task_vector, 1, 0))
        # sim = torch.transpose(sim, 1, 0)
        # sim = (sim+1)/2
        
        # sims = []
        # for grad_idx in range(task_vectors[0].shape[-1]):
        #     task_vector = F.normalize(torch.stack([tv[:,grad_idx] for tv in task_vectors], dim=0), dim=-1)
        #     sim = torch.matmul(task_vector,
        #                     torch.transpose(task_vector, 1, 0))
        #     sim = torch.transpose(sim, 1, 0)
        #     sims.append(sim)
        
        # sim = torch.stack(sims, dim=0).mean(dim=0)
        
        # extra_state_dict_dict['task_similarity'] = sim
        # extra_state_dict_dict['curr_round'] += 1
        # for client_id in range(training_args.num_clients):
        #     load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict)
    logger.info("total done\n")

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
