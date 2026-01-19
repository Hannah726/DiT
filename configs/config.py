def get_config(obs_window=12, seed=0):
    max_events = {6: 164, 12: 243, 24: 367}
    
    return {
        'data_dir': f'data/processed_{obs_window}',
        'codes_dir': f'data/processed_{obs_window}',
        'time_file': f'mimiciv_con_time_{obs_window}.npy',
        'rqvae_checkpoint': f'data/codebook/{obs_window}/train_RQVAE_indep.pkl',
        'checkpoint_dir': f'checkpoints/maskdit_{obs_window}h_seed{seed}',
        
        'obs_window': obs_window,
        'seed': seed,
        'max_event_size': max_events[obs_window],
        'num_codes': 8,
        
        'codebook_size': 1025,
        'mask_token_id': 1025,
        'mask_schedule': 'cosine',
        'mask_ratio_min': 0.05,
        'mask_ratio_max': 0.95,
        
        'time_dim': 1,
        'time_proj_dim': 128,
        'time_pad_value': -1.0,
        
        'rqvae_dim': 256,
        'latent_dim': 128,
        'hidden_dim': 512,
        'num_layers': 8,
        'num_heads': 8,
        'dropout': 0.1,
        'code_aggregation': 'sum',
        'freeze_codebook': True,
        
        'batch_size': 64,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'warmup_steps': 1000,
        'epochs': 100,
        'gradient_accumulation_steps': 1,
        'grad_clip': 1.0,
        'use_amp': True,
        
        'num_iterations': 10,
        'temperature': 1.0,
        
        'log_interval': 100,
        'val_interval': 1,
        'save_interval': 5,
        'early_stopping_patience': 10,
        'project_name': 'MaskDiT-EHR',
        'run_name': f'maskdit_{obs_window}h_seed{seed}',
        'use_wandb': False,
        'num_workers': 4,
        'compile_model': False,
    }


def get_quick_test_config(obs_window=12):
    config = get_config(obs_window, seed=0)
    config.update({
        'data_fraction': 0.1,
        'epochs': 5,
        'val_interval': 1,
        'run_name': f'maskdit_{obs_window}h_quicktest'
    })
    return config