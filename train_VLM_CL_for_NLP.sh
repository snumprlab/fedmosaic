# CIL CONFG
NOTE="NLPNEW_sft_bs4_nosaveoptim_COSINE_lr1e-4_5e-4_sc90_2tasks_2rounds_fixitr30_T0125_decay099_SEED1"
MODE="sft"
MODEL_ARCH="llama3_8b" # lava gema_
RND_SEED=1

# fed ar
SCENARIO=90
NUM_ROUNDS=2
NUM_TASKS=2
NUM_CLIENTS=52
MODEL_MAX_LEN=20000
NUM_ITER=30

###
ANYTIME_EVAL=False
ANYTIME_EVAL_FREQ=1
##
MEMORY_SIZE=100000
IS_STREAMONLY=False

LORA_ENABLE=True
IA3_ENABLE=False

USE_TASK_ID=False
USE_PROMPT=False

SAVE_OPTIM=False

USE_TASK_VECTOR=False
USE_FISHER=False

GENERATOR_OUTPUT_SIZE=1024
GENERATOR_HIDDEN_DIM=8
GENERATOR_HIDDEN_FEATURE=8
KEY_EMBED_SIZE=64
POOL_SIZE=4
PROMPT_TOP_K=1
EMA_RATIO=0.9

BATCHSIZE=1

LR=1e-4
MM_PROJECTOR_LR=5e-4 #3e-4
FINAL_LR=$LR #3e-4
MM_FINAL_LR=$MM_PROJECTOR_LR #3e-4
OPT_NAME="adamw_torch" # adam8bit_bnb adamw_torch
SCHED_NAME="cosine" #cosine
WARMUP_RATIO=0.1 # SHOULD BE 0.03 / NUM_ROUNDS
DECAY_RATIO=0.9

if [ "$MODEL_ARCH" == "llava" ]; then
    MODEL_NAME="llava-hf/llava-1.5-7b-hf"
    VERSION="v1"
    MODEL_TYPE="llama"
    BITS=16

elif [ "$MODEL_ARCH" == "llama3_1b" ]; then
    MODEL_NAME="thkim0305/llama3.2_1B_vl"
    VERSION="llama3"
    MODEL_TYPE="llama3"
    BITS=16
elif [ "$MODEL_ARCH" == "llama3_3b" ]; then
    MODEL_NAME="thkim0305/llama3.2_3B_vl"
    VERSION="llama3"
    MODEL_TYPE="llama3"
    BITS=16
elif [ "$MODEL_ARCH" == "llama3_8b" ]; then
    MODEL_NAME="thkim0305/llama3.1_8B_vl"
    VERSION="llama3"
    MODEL_TYPE="llama3"
    BITS=16
else
    echo "Undefined setting"
    exit 1
fi

# --master_port 29500
# --num_gpus=4

# LOAD_CHECKPOINT="client_states_fedours_bs4_saveoptim_lr4e-5_sc5_4tasks_5rounds_fixitr100_t0.2_memonly_rank32/round15_task_vector_local_weights.pth"
LOAD_CHECKPOINT="client_states_fedavg_bs4_saveoptim_lr2e-5_sc5_4tasks_5rounds_fixitr100/server_model_round14.pth"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --master_port 29637 \
    --include localhost:0 \
    train_VLM_CL.py \
    --deepspeed ./deepspeed_script/zero2.json \
    --model_name_or_path $MODEL_NAME \
    --model_name_for_dataarg $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --version $VERSION \
    --num_clients $NUM_CLIENTS \
    --model_max_length $MODEL_MAX_LEN \
    --num_rounds $NUM_ROUNDS \
    --num_tasks $NUM_TASKS \
    --scenario $SCENARIO \
    --gradient_checkpointing True \
    --num_train_epochs 1 \
    --num_iter $NUM_ITER \
    --gradient_accumulation_steps 4 \
    --bits $BITS \
    --bf16 True \
    --tf32 True \
    --mode $MODE --dataloader_num_workers 2 \
    --seed $RND_SEED \
    --optim $OPT_NAME \
    --lr_scheduler_type $SCHED_NAME \
    --weight_decay 0. \
    --warmup_ratio $WARMUP_RATIO \
    --decay_ratio $DECAY_RATIO \
    --learning_rate $LR --per_gpu_train_batch_size $BATCHSIZE \
    --final_lr $FINAL_LR --mm_final_lr $MM_FINAL_LR \
    --mm_projector_lr $MM_PROJECTOR_LR \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --logging_steps 2 \
    --note $NOTE \
    --memory_size $MEMORY_SIZE \
    --is_streamonly $IS_STREAMONLY \
    --anytime_eval $ANYTIME_EVAL \
    --anytime_eval_freq $ANYTIME_EVAL_FREQ \
    --prompt_num 1 \
    --lora_enable $LORA_ENABLE \
    --ia3_enable $IA3_ENABLE \
    --use_task_id $USE_TASK_ID \
    --get_prompt $USE_PROMPT \
    --generator_output_size $GENERATOR_OUTPUT_SIZE \
    --generator_hidden_dim $GENERATOR_HIDDEN_DIM \
    --generator_hidden_feature $GENERATOR_HIDDEN_FEATURE \
    --ema_ratio $EMA_RATIO \
    --key_embed_size $KEY_EMBED_SIZE \
    --pool_size $POOL_SIZE \
    --prompt_top_k $PROMPT_TOP_K \
    --save_optim $SAVE_OPTIM \
    --use_task_vector $USE_TASK_VECTOR \
    --use_fisher $USE_FISHER \
    --fedours False \
    --is_hetero_model True \
    --load_pretrained_orthnorm True \
    --A_ensure_orth True \
    --randomize_orth_B False \
    --softmax_temp 0.5 \
    --fisher_freq 10 \
    --is_continual True \
    --get_prompt True \
    --is_cross_model_series False \
    --is_multimodal False \
    --lora_r 16 \
    --lora_alpha 32 \
    --output_dir "./results/test/" #> ./nohup/${NOTE}.log 2>&1 &

# --eval_period $EVAL_PERIOD
# lr_scheduler_type
#  --load_checkpoint $LOAD_CHECKPOINT \
    # --lora_r 32 \
    # --lora_alpha 64
#    --load_pretrained_orthnorm True \
#    --A_ensure_orth True \
#    --randomize_orth_B False \