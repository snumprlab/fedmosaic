from transformers import LlamaForCausalLM, AutoModel, AutoModelForCausalLM
import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from transformers.utils import logging

from models.duallora_moe.dualmoeloralayer import DualMOELoraLayer
from models.dual_pqlora_freeze_full_moe.dual_pqloralayer_freeze_full_moe import PQMOELoraFullFreezeLayer
logger = logging.get_logger(__name__)

class CustomLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        
        self.active_state = 'gate'
    
    def set_state(self, state):
        assert state in ['lora1', 'lora2', 'gate'], state
        self.active_state = state
        
        for name, module in self.named_modules():
            
                module.set_state(state)

    def activate_all(self):
        for name, module in self.named_modules():
            if isinstance(module, DualMOELoraLayer) or isinstance(module, PQMOELoraFullFreezeLayer):
                module.activate_all()

    def activate_lora1(self):
        for name, module in self.named_modules():
            if isinstance(module, DualMOELoraLayer) or isinstance(module, PQMOELoraFullFreezeLayer):
                module.activate_lora1()
    
    def activate_lora2(self):
        for name, module in self.named_modules():
            if isinstance(module, DualMOELoraLayer) or isinstance(module, PQMOELoraFullFreezeLayer):
                module.activate_lora2()
        