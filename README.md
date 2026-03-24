# PFCL

## Environment setting
```bash
conda create -n fcl2 python=3.10
conda activate fcl2
pip install transformers==4.47.1
pip install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.5.8 --no-build-isolation
pip install peft==0.14.0 bitsandbytes pandas kornia opencv-python timm torch_optimizer easydict pycocoevalcap sentencepiece protobuf trl==0.8.6 deepspeed==0.15.2 loguru captum POT jsonlines numpy==1.26.4 accelerate==0.29.3 nevergrad
pip install -U scikit-learn
```

- old scenario (기존 실험에 사용했던 시나리오)
    - sc132: DRAKE 1B 3B hetero (llama)
    - sc137: DRAKE 1B 3B 8B hetero (llama)
    - sc1137: DRAKE 1B 3B 8B hetero (qwen)
    - sc203: fs-llm hetero (llama)
    - sc262: fed-aya hetero (llama)

- new scenario (hetero client 제대로 설정한 시나리오들)
    - scenario-0~5: DRAKE (sc1 - 3B homo, sc4 - 1B/3B hetero, sc5 - 1B/3B/8B hetero)
    - scenario-7~12: unseen
    - scenario-20~24: HFLB (sc21 - 3B homo, sc23 - 1B/3B hetero)
    - scenario-60~ : fed-aya (sc62)
    - scenario-70~ : fed-llm (sc74)
    - sc103 - DRAKE Qwen hetero


- Large scale NLP 실험:
    - 시나리오: 90 (52 clients), 93 (36 clients)
    - 데이터셋: gdrive files download 1QJK0JtrmrD2AsZvk78WpqT0_TsDHJXVc
        - `dataset` 폴더 안에서 다운 후 `tar -xvf nlp_datasets.tar`
    - 핵심 하이퍼파라미터:
        - lr 1e-4/5e-4 
        - batch size 4
        - cosine scheduler
        - 2 rounds per task
        - 30 rounds (for sft, reduce the iteration for baseline)
        - lora r 16 / lora alpha 32
    - `train_VLM_CL_for_NLP.sh` 참고
    - 테스트셋은 현재 task당 15개씩만 가져오도록 설정되어있음 (eval_VLM_CL.py L655)

### 실험돌릴 때 설정해야하는 argument
```
--is_continual True 혹은 False -> PFCL or PFL 실험
```

- ours 
    - MODE="feddualMultipqfullfreeze_homoAgg_moe"
    - USE_TASK_VECTOR=True / USE_FISHER=True -> ours 빼고 나머지 방법에서는 둘 다 False로 해야 에러 없음.
    - NUM_ITER=94
    - --gradient_accumulation_steps 4
- sft/fedavg/feddpa
    - NUM_ITER=100
    - --gradient_accumulation_steps 4
- fedsim/takfl
    - NUM_ITER=75
    - --gradient_accumulation_steps 4
- fedmkt
    - NUM_ITER=60
    - --gradient_accumulation_steps 4
- ditto/perada
    - NUM_ITER=50
    - --gradient_accumulation_steps 8
- feddat
    - NUM_ITER=43
    - --gradient_accumulation_steps 8

- DRAKE / HFLB
    - LR=2e-5
    - MM_PROJECTOR_LR=5e-5
    - --is_multimodal True
- FS-LLM / Fed-aya
    - LR=3e-4
    - MM_PROJECTOR_LR=5e-4
    - --is_multimodal False
    - --lora_r 16
    - --lora_alpha 32 

- init weight for 1B, 3B

    - final core
```bash
llava 1b (4block): gdrive files download 1_sNX5TjuDjr9Sw4zOimlHxee8RAAljcs
llava 3b (4block): gdrive files download 1xCewPkTCfMAjWSix7q8AiJnpAIbhbDOx
llava 8b (4block): gdrive files download 1LmhTUJBGSfF2mWWcjF7YDUzrVWEXYTBP
llava 1b (8block): gdrive files download 1_iUI_KmCTKhwXI_l7cJ-C1O21XpEwOS7
llava 3b (8block): gdrive files download 1xt3-Y-HGshNTsK-4EyOrD7G2BsPMxIk4

qwen 0.5b (4block): gdrive files download 1z0zr3Iat5bsIXZCisekdSDOAeXxjwblK
qwen 1.5b (4block): gdrive files download 13j55g6kdkRlQ5ddhHee5qwbVSiVweiYw
qwen 3b (4block): gdrive files download 194uQ6Dsq9fMTkqjzCZ2LrToTxVjSYUtP

llama 1b (4block): gdrive files download 1zSN14FDncBRdKXwq2S0a_hpU_eAZMMjq
llama 3b (4block): gdrive files download 13J_P8AmWW49e8ojRazuaHHay4ke60G0r
llama 8b (4block): gdrive files download 1HNM0D9OxLBFfNCVFdnZj98i7c0TAM5nD

new
llava 1b (4block): gdrive files download 1TAeiWq36ewrwVCgId3TKpXTxnRu4rDtN
llava 3b (4block): gdrive files download 1ppeB7CxiD8yhaAWY5Gvz4OpUPA7yQc5_

llava 1b (8block): gdrive files download 18shu4ixvCJCu1eAMbbMCTFuLocJk7ZFf
llava 3b (8block): gdrive files download 1i1T-oZT_e2jqOL-NgDXrhX1d3OnBTDfZ

fsllm data - gdrive files download 1IZ6yrU5tIw44rqKUoblEJw3ROtmDCl1g
fsllm testset - gdrive files download 1cFczfI93f-u_vjc1D2f6clA_Cx7twMEH
chatbotit.json - gdrive files download 1Jk74xpVj7WXRWBzwBKBidh5iS884SGIn
```

    - blockwise (4 layers) - A random
```bash
gdrive files download 17ZJwam-dnm1_F186XRfIAJQe0SRCUWGr
gdrive files download 1zx-jltyAuqfyuAABGlZR7zGfZ4K6h_rh
```

    - blockwise (4 layers) - A orthnorm
```bash
gdrive files download 1b90paEEbwvv608S7KW-XcXKcCcXWAyhS
gdrive files download 1gUh9JvVpJJqOXW0mzoWKiAF_0OuGFcKW
```
    - blockwise (4 layers) - PCA
```bash
gdrive files download 1YxdZKJosR53oyi_kMiMb8Fu9WldIyUT5
gdrive files download 1iQYOO5VNRTA282JZCsNlSWLF4x4n6R5c
gdrive files download 1jSDpPlJVVoZJilhK73pbAMdTK-vMCxZ8
```
    - blockwise (4 layers) - PCA (new!)
```bash
gdrive files download 1aaJYQKo6UcvLgupg2t71NDI8V6oHEz-s
gdrive files download 1jRVjKI8wMqZ9hCrEEfVT1e6cZmLDbNBm
```

    - blockwise (4 layers) - orthnormal (new!)
```bash
gdrive files download 1_sNX5TjuDjr9Sw4zOimlHxee8RAAljcs
gdrive files download 1phhio5oPV_Aa10yk8y8OuM_SbUCDL2KD
gdrive files download 1xCewPkTCfMAjWSix7q8AiJnpAIbhbDOx
```

    - blockwise (8 layers, back) - orthnormal (new!)
```bash
gdrive files download 1K04V3W9XtildXF9jAPqXhTO_9p0SAo3d
gdrive files download 1Ma5Uu9IlO7_Dc-ra6eOzr3p4mw5OV_p4
gdrive files download 1-0GiYQQjrzxRT9ZvkqGyyKRFt-dF6Ldi
```

    - blockwise (2 layers) - PCA
```bash
gdrive files download 194GaXFFlz3-T7rQQpnzQI1sp6FgIKA0D
gdrive files download 1wnh1gelZrf810q2hSMHg0K_j_FZKp1jU
gdrive files download 1GBfjgvISWsf7_7_oy8y_WVaihQLyBCR9
```
    - blockwise (8 layers, back) - PCA
```bash
gdrive files download 13wGqVnjsTJh1hV0Jl4lG9KKE_OrAfFP4
gdrive files download 1BrwJHL3jcfS5ULgKdbahD3X3QOMgEP4G
```
    - blockwise (8 layers, front) - PCA
```bash
gdrive files download 1KPbBjo_VaBjuXefgMYu6WtbNJbT0uPhd
gdrive files download 1fp1PlRmnrX3rz2QPvCnkKQfCl7TGjcwy
```

    - blockwise Optimal (2 layers) - PCA
```bash
gdrive files download 1LG81Bsz24FnQQnD7YFA_2GGfg03syQYg
gdrive files download 1xa_WluV_fe0RjW0bGndM97Gdk_QB9Igk
```
    - blockwise Optimal (4 layers) - PCA
```bash
gdrive files download 1H7er2B6W3uLQUCkDlncFuED-opuNpEhy
gdrive files download 1py9w8dDAQ-iG7m8x-s1Woj3l3h8_MvXu
```
    - blockwise Optimal (8 layers) - PCA
```bash
gdrive files download 1RdwczYGzEmQetTizqo6s63OAGLP8yJj8
gdrive files download 1rjqa8tLrrCx3K5p167Z8EruqYEaJ7P7H
```

    - blockwise - PCA (R256)
```bash
gdrive files download 1IS4fEK0-KmVJ9rgmCtIzhYEDjpJWSW8x
gdrive files download 1slnzeTumli3s5nTWkpLzJfRq0KW8-cUt
```
    - blockwise - PCA (R512)
```bash
gdrive files download 1tHjBCnYtObyQibuqdSrk7vvCEBRJemkA
gdrive files download 1IZLEG2F1F2YiR-5QhwqldLnJvhup45Ao
```
    - blockwise - PCA (R1024)
```bash
gdrive files download 1uDJ3eMnsbdHcfxqphpPL57ovk1HqKEOU
gdrive files download 1ADtJlXtyMWNVDk_W9GWG5CLawj64tiQ6
```

    - llama - blockwise - PCA (R16)
```bash
gdrive files download 1lYofrpJhLfFdg4-ayZnQ73-qIZVSdBxk
gdrive files download 1sygf0_hkyMYxMCbeX6RUmSjbfAZ3SZ3I
```

    - llava (qwen) - blockwise PCA (0.5b, 1.5b, 3b)
```bash
gdrive files download 1s7SwWMHEZI6-1sEpyfgjTgYxB7EBdwxU
gdrive files download 1wVZ5Vrw9418sxaWL7INq518PvrAfQ9oy
gdrive files download 1HFUFyjiGhIST9SSXsX87q8-3lHCg2zHL
```

    - llava - blockwise all CCA (1b, 3b)
```bash
gdrive files download 1WrJuHX0iQ5z15BC8pR3-h5RO4qVoDuRH
gdrive files download 1T3uawEzFkd5uSwtX8UeROHaIDO7RJHsq
```

## New data instruction

1. In `dataset` folder, run the following script files:
```bash
bash Fashion.sh
bash HRVQA.sh
bash Pair_TF.sh
bash KGQA.sh
bash Bongard.sh
bash iconqa.sh
bash CoInstruct.sh
bash Visual_Relation.sh
bash Visual_storytelling.sh
bash MultiVQA_small.sh
```

data jsons
```bash
gdrive files download 1Sr2QDnn6lH71fc8L6sol0O8Bcw5ZcKK8
gdrive files download 141f9sLFjpRjfqzk6nWQSusonuf9VqITI
```

2. Run the following preprocessing python codes:

- Pair_TF
```bash
python ./preprocess/preprocess_NLVR2.py
python ./preprocess/preprocess_PatternCom.py
```

- KGQA
```bash
python ./preprocess/preprocess_WebQA.py
python ./preprocess/preprocess_TQA.py
python ./preprocess/preprocess_AQUA.py
```

- Bongard
```bash
python ./preprocess/preprocess_Bongard2.py
python ./preprocess/preprocess_Bongard_query.py
python ./preprocess/preprocess_Bongard_HOI.py
python ./preprocess/preprocess_Bongard_HOI_query.py
```

- IconQA
```bash
python ./preprocess/preprocess_iconqa.py
```

- CoInstruct
```bash
python ./preprocess/preprocess_coinstruct.py
```

- Visual_Relation
```bash
python ./preprocess/preprocess_SpotDiff.py
python ./preprocess/preprocess_Bird2Words.py
python ./preprocess/preprocess_CLEVR.py
python ./preprocess/preprocess_IEdit.py
```

- Visual_storytelling
```bash
python ./preprocess/preprocess_PororoSV.py
python ./preprocess/preprocess_FlintstonesSV.py
python ./preprocess/preprocess_VIST.py
python ./preprocess/preprocess_AESOP.py
```

- MultiVQA_small
```bash
python ./preprocess/preprocess_VISION.py
python ./preprocess/preprocess_VizWiz.py
python ./preprocess/preprocess_MIT.py
```

## Dataset preprocessing Again

```bash
python ./preprocess_again2/preprocess_again.py
python ./preprocess_again2/preprocess_aqua_again.py
python ./preprocess_again2/preprocess_bongard_hoi_again.py
python ./preprocess_again2/preprocess_bongard_hoi_query_again.py
python ./preprocess_again2/preprocess_coinstruct_db_again.py
python ./preprocess_again2/preprocess_fashioniq_again.py
python ./preprocess_again2/preprocess_patterncom_again.py
```
