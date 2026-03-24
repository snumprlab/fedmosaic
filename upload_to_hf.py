from huggingface_hub import login, HfApi
login(token='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')
api = HfApi()

api.upload_folder(
    folder_path="client_states_takfl_bs4_nosaveoptim_COSINE_r16_32_lr3e-4_sc262_4tasks_5rounds_fixitr38_T0125_decay099",
    path_in_repo="client_states_takfl_bs4_nosaveoptim_COSINE_r16_32_lr3e-4_sc262_4tasks_5rounds_fixitr38_T0125_decay099",
    allow_patterns=["*_client_model_round20.pth","*_client_model_round15.pth", "*_client_model_round10.pth", "*_client_model_round5.pth"],
    repo_id="thkim0305/llm_baselines",
    repo_type="model",
)