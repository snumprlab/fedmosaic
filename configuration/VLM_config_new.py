from dataclasses import dataclass, field
import transformers
from typing import Optional, Any

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None) #"liuhaotian/llava-v1.5-7b"
    model_type: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v1")
    max_new_tokens: Optional[int] = field(default=512)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'pad'
    model_name_for_dataarg: Optional[str] = field(default=None)
    
    get_prompt: bool = False

@dataclass
class TrainingConfig(transformers.TrainingArguments):
    
    is_eval: bool = False
    is_prompt: bool = False
    round_to_eval: int = None
    eval_temp: float = 0.0
    eval_server: bool = False
    unseen_task: bool = False
    eval_client: int = field(default=None)
    eval_iter: int = field(default=None)
    zeroshot: bool = False
    eval_all: bool = False
    eval_client_start: int = field(default=None)
    eval_client_end: int = field(default=None)
    eval_client_eval_start: int = field(default=None)
    eval_client_eval_end: int = field(default=None)
    
    num_iter:int = field(default=100)

    # CCA config
    gamma:float = field(default=0.05)

    # cl config
    mode: str = field(default="er")
    # dataset: str = field(default="cifar10")
    scenario: int = field(default=1)
    note: str = field(default=None)
    eval_period: int = field(default=100)
    online_iter: float = field(default=1.0)
    
    anytime_eval: bool = False
    anytime_eval_freq: int = 10

    # federated learning
    num_clients: int = 5
    num_rounds: float = 20
    num_tasks: int = 4
    iter_per_round: int = 1
    state_dir: str = field(default="./client_states")
    final_lr: float = field(default=1e-6)
    mm_final_lr: float = field(default=1e-6)
    
    # continual learning
    memory_size: int = 100000
    is_streamonly: bool = True
    use_task_id: bool = False
    online_stream_T: float = 0.125
    online_stream_count_decay_ratio: float = 0.99
    
    # prompt tuning args
    prompt_num: int = field(default=100)
    
    optim: str = field(default="adamw_torch")
    is_wsd: str = field(default=None)
    decay_ratio: float = field(default=1.0)
    save_optim: bool = field(default=False)
    
    temp_batchsize: int = field(default=2)

    cache_dir: Optional[str] = field(default=None)
    
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=8,
        metadata={"help": "How many bits to use."}
    )

    is_hetero_model: bool = False

    # lora config
    lora_enable: bool = True
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = 2e-5
    group_by_modality_length: bool = field(default=True)
    
    ia3_enable: bool = False
    
    
    # generator config
    generator_output_size: int = 256
    generator_hidden_dim: int = 32
    generator_hidden_feature: int = 16
    set_state:str = 'gate'
    ema_ratio:float = 0.996
    key_embed_size:int = 64
    pool_size:int = 4
    prompt_top_k:int = 1
    
    use_task_vector:bool = False
    use_fisher:bool = False
    
    load_checkpoint:str = None
    fedours:bool = False
    
    load_pretrained_random:bool = False
    load_pretrained_orthnorm:bool = False
    load_pretrained_pca:bool = False
    randomize_B:bool = False
    randomize_orth_B:bool = False
    A_ensure_orth:bool = False
    
    softmax_temp: float = 0.2
    save_per_step: bool = False
    
    taskloss_weight: float = 0.1
    distill_weight: float = 0.1
    
    immediate_ema_update:bool = False
    share_ema:bool = False
    
    # hypergradient
    use_hypergradient:bool = False
    hypergrad_lr:float = 1e-8
    
    # incremental client setup
    is_incremental_client_scenario: bool = False
    
    fisher_freq: int = 5
    
    num_serverdistill: int = 80
    
    load_pretrained_cca: bool = False
    
    is_continual: bool = True
    
    iter_to_get_grad: int = 20
    
    gradient_ratio: float = 1.0
    gradient_noise_type: str = "gaussian"
    gradient_noise_std: float = 0.0
    
    is_cross_model_series: bool = False