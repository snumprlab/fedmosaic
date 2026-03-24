#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
# CIL CONFIG
NOTE="new_fedours_moe_T05_freq10_bs4_saveoptim_r16_32_lr3e-4_5e-4_sc205_4tasks_5rounds_fixitr29_T0125_decay099"
MODE="fedours_moe"
MODEL_ARCH="llama3_3b" # llava llama3_1b llama3_3b

# fed args
SCENARIO=205
NUM_ROUNDS=5
NUM_TASKS=4
NUM_CLIENTS=5
MODEL_MAX_LEN=20000
MAX_NEW_TOKENS=512

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
else
    echo "Undefined setting"
    exit 1
fi

# ROUND_TO_EVALS=$2
ROUND_TO_EVALS=(20)
ITER_TO_EVAL=0

for ((index=0; index<${#ROUND_TO_EVALS[@]}; index++)); do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$1 python eval_FS_LLM_math.py \
        --is_eval True \
        --model_name_or_path $MODEL_NAME \
        --model_name_for_dataarg $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --version $VERSION \
        --scenario $SCENARIO \
        --num_rounds $NUM_ROUNDS \
        --num_tasks $NUM_TASKS \
        --num_clients $NUM_CLIENTS \
        --model_max_length $MODEL_MAX_LEN \
        --max_new_tokens $MAX_NEW_TOKENS \
        --bits $BITS \
        --bf16 True \
        --tf32 True \
        --note $NOTE \
        --mode $MODE \
        --eval_server False \
        --unseen_task False \
        --zeroshot False \
        --lora_enable True \
        --ia3_enable False \
        --generator_output_size 512 \
        --generator_hidden_dim 8 \
        --generator_hidden_feature 8 \
        --key_embed_size 64 \
        --prompt_top_k 1 \
        --pool_size 40 \
        --set_state "gate" \
        --is_prompt False \
        --use_task_vector False \
        --is_multimodal False \
        --lora_r 16 \
        --lora_alpha 32 \
        --round_to_eval ${ROUND_TO_EVALS[$index]} \
        --output_dir "./nohup" > ./nohup/${NOTE}_eval_round${ROUND_TO_EVALS[$index]}_math.log 2>&1 & #_iter${ITER_TO_EVAL}
done
# --eval_period $EVAL_PERIOD
#--eval_iter $ITER_TO_EVAL \