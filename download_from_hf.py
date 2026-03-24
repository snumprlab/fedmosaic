from huggingface_hub import snapshot_download, hf_hub_download, login
login(token='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')
#hf_hub_download(repo_id="thkim0305/feddat_baselines", filename="client_states_fdlora_bs4_saveoptim_lr2e-5_sc315_4tasks_5rounds_fixitr100_T0125_decay099/*.pth", local_dir="client_states_fdlora_bs4_saveoptim_lr2e-5_sc315_4tasks_5rounds_fixitr100_T0125_decay099")
# snapshot_download(repo_id="thkim0305/drake_baselines", local_dir='./', allow_patterns=["client_states_feddualMulti2pqfullfreeze_back_homoAgg_moe_NEW_T05_bs4_saveoptim_lr2e-5_sc132_4tasks_5rounds_fixitr97_T0125_decay099_SEED2/*.pth"])
# snapshot_download(repo_id="thkim0305/pfcl_ablations", local_dir='./', allow_patterns=["client_states_ditto_NOCONT_bs4_saveoptim_lr2e-5_5e-5_sc132_4tasks_5rounds_fixitr50_T0125_decay099/*.pth"])

snapshot_download(repo_id="thkim0305/llm_baselines",local_dir='./',allow_patterns=["client_states_feddualMultipqfullfreeze_homoAgg_moe_NOCONT_bs4_saveoptim_lr3e-4_sc203_r16_32_4tasks_5rounds_fixtir29_T0125_decay099/*.pth"])
