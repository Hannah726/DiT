def get_config(obs_window=12, seed=0):
    max_events = {6: 164, 12: 243, 24: 367}
    
    return {
        'data_dir': f'data/processed_{obs_window}',
        'codes_dir': f'data/processed_{obs_window}',
        'time_file': f'mimiciv_hi_pad_time.npy',
        'rqvae_checkpoint': f'data/codebook/{obs_window}/train_RQVAE_indep.pkl',
        'checkpoint_dir': f'outputs/checkpoints/maskdit_{obs_window}h_seed{seed}',
        
        'obs_window': obs_window,
        'seed': seed,
        'max_event_size': max_events[obs_window],

        'spatial_dim': 4,
        'num_quantizers': 2,
        'codebook_size': 1024,
        'mask_token_id': 1024,
        
        'mask_schedule': 'cosine',
        'mask_ratio_min': 0.05,
        'mask_ratio_max': 0.95,
        'label_smoothing': 0.05,
        'nested_mask_k2_ratio': 0.3,
        
        'time_dim': 1,
        'time_proj_dim': 128,
        'time_pad_value': -1.0,
        
        'rqvae_dim': 256,
        'd_model': 256,
        'num_layers': 12,
        'num_heads': 8,
        'dropout': 0.1,
        'use_adaln': True,
        'freeze_codebook': False,
        
        'batch_size': 64,
        'lr': 5e-5,
        'weight_decay': 0.01,
        'warmup_steps': 2000,
        'epochs': 200,
        'gradient_accumulation_steps': 1,
        'grad_clip': 0.5,
        'use_amp': False,
        
        'num_iterations': 10,
        'temperature': 1,
        
        'comprehensive_val_interval': 10,
        'log_interval': 100,
        'val_interval': 1,
        'save_interval': 10,
        'early_stopping_patience': 10,
        'project_name': 'MaskDiT-EHR',
        'run_name': f'maskdit_{obs_window}h_seed{seed}',
        'use_wandb': False,
        'num_workers': 8,
        'compile_model': True,
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