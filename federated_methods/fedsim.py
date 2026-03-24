import torch
from torch import nn
from federated_methods.fedavg import LLaVATrainerFEDAVG
            
def fedsim_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    training_args.max_seq_length = training_args.model_max_length
    training_args.packing=False
    trainer = LLaVATrainerFEDSIM(model=model,
        tokenizer=tokenizer,
        args=training_args,
        client_id = extra_state_dict_dict['client_id'],
        curr_round = extra_state_dict_dict['curr_round'],
        test_datalist=extra_state_dict_dict['test_datalist'],
        processor=extra_state_dict_dict['processor'],
        data_args=extra_state_dict_dict['data_args'],
        task_vector=extra_state_dict_dict['task_vector'] if 'task_vector' in extra_state_dict_dict else None,
        fisher_old=extra_state_dict_dict['fisher_old'] if 'fisher_old' in extra_state_dict_dict else None,
        fisher_freq=extra_state_dict_dict['fisher_freq'] if 'fisher_freq' in extra_state_dict_dict else 5,
        model2=extra_state_dict_dict['model2'] if 'model2' in extra_state_dict_dict else None,
        **data_module,
        )
    return trainer

class LLaVATrainerFEDSIM(LLaVATrainerFEDAVG):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        # global forward
        model.module.set_state('lora1')
        _, outputs = super(LLaVATrainerFEDSIM, self).compute_loss(model, inputs, return_outputs=True,num_items_in_batch=num_items_in_batch)     
        # local forward
        model.module.set_state('lora2')
        _, local_outputs = super(LLaVATrainerFEDSIM, self).compute_loss(model, inputs, return_outputs=True,num_items_in_batch=num_items_in_batch) 
        
        final_logits = outputs['logits'] + local_outputs['logits']
        labels = inputs['labels']
        # Shift so that tokens < n predict n
        shift_logits = final_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels) 
        
        return (loss, outputs) if return_outputs else loss