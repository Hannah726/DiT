"""
Detailed GPU Memory Profiler
"""
import torch
import torch.nn as nn
from models.ehr_diffusion import EHRDiffusionModel
from models.gaussian_diffusion import TimeAwareGaussianDiffusion
import gc

def format_bytes(bytes):
    """Format bytes to GB"""
    return f"{bytes / 1024**3:.3f} GB"

def profile_training_step(batch_size, config, enable_recon=True):
    """Profile memory usage during one training step"""
    
    print(f"\n{'='*70}")
    print(f"Profiling: batch_size={batch_size}, recon_weight={'0.1' if enable_recon else '0.0'}")
    print(f"{'='*70}\n")
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    device = torch.device('cuda')
    
    # ====================
    # 1. Model Creation
    # ====================
    print("Step 1: Creating model...")
    model = EHRDiffusionModel(
        vocab_size=2385,
        type_vocab_size=7,
        dpe_vocab_size=15,
        **config
    ).to(device)
    
    diffusion = TimeAwareGaussianDiffusion(
        timesteps=1000,
        beta_schedule='linear'
    )
    
    mem_after_model = torch.cuda.memory_allocated()
    print(f"  Model parameters: {format_bytes(mem_after_model)}")
    
    # ====================
    # 2. Optimizer
    # ====================
    print("\nStep 2: Creating optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    mem_after_optimizer = torch.cuda.memory_allocated()
    print(f"  Optimizer states: {format_bytes(mem_after_optimizer - mem_after_model)}")
    print(f"  Total so far: {format_bytes(mem_after_optimizer)}")
    
    # ====================
    # 3. Create Input Data
    # ====================
    print(f"\nStep 3: Creating input batch (batch_size={batch_size})...")
    B = batch_size
    N = 243
    L = 128
    
    input_ids = torch.randint(0, 2385, (B, N, L), device=device)
    type_ids = torch.randint(0, 7, (B, N, L), device=device)
    dpe_ids = torch.randint(0, 15, (B, N, L), device=device)
    con_time = torch.randn(B, N, 1, device=device)
    event_mask = torch.ones(B, N, L, device=device)
    
    mem_after_data = torch.cuda.memory_allocated()
    print(f"  Input tensors: {format_bytes(mem_after_data - mem_after_optimizer)}")
    print(f"  Total so far: {format_bytes(mem_after_data)}")
    
    # ====================
    # 4. Forward Pass (Encoding)
    # ====================
    print("\nStep 4: Forward pass (encoding)...")
    model.train()
    
    with torch.cuda.amp.autocast(enabled=True):
        joint_latent, event_refined, time_refined, prompt_weights, true_length, event_level_mask = model.encode(
            input_ids, type_ids, dpe_ids, con_time,
            event_mask=event_mask
        )
        
        mem_after_encode = torch.cuda.memory_allocated()
        print(f"  Encoding activations: {format_bytes(mem_after_encode - mem_after_data)}")
        print(f"  Total so far: {format_bytes(mem_after_encode)}")
        
        # ====================
        # 5. Diffusion Forward
        # ====================
        print("\nStep 5: Diffusion forward...")
        t = torch.randint(0, 1000, (B,), device=device).long()
        noise = torch.randn_like(joint_latent)
        noisy_latent, actual_noise = diffusion.q_sample(joint_latent, t, noise=noise, return_noise=True)
        
        mem_after_diffusion = torch.cuda.memory_allocated()
        print(f"  Diffusion tensors: {format_bytes(mem_after_diffusion - mem_after_encode)}")
        
        # ====================
        # 6. DiT Forward
        # ====================
        print("\nStep 6: DiT forward (denoising)...")
        prompts = model.pattern_prompts.prompts if model.use_prompts else None
        predicted_noise = model.dit(
            noisy_latent, t,
            condition=None,
            prompts=prompts,
            mask=event_level_mask
        )
        
        mem_after_dit = torch.cuda.memory_allocated()
        print(f"  DiT activations: {format_bytes(mem_after_dit - mem_after_diffusion)}")
        print(f"  Total so far: {format_bytes(mem_after_dit)}")
        
        # ====================
        # 7. Boundary Prediction
        # ====================
        print("\nStep 7: Boundary prediction...")
        predicted_x0 = diffusion.predict_start_from_noise(noisy_latent, t, predicted_noise)
        denoised_event_latent = predicted_x0[..., :model.pattern_dim]
        bin_logits, bin_probs, predicted_length = model.boundary_predictor(denoised_event_latent)
        
        mem_after_boundary = torch.cuda.memory_allocated()
        print(f"  Boundary prediction: {format_bytes(mem_after_boundary - mem_after_dit)}")
        
        # ====================
        # 8. Loss Computation
        # ====================
        print("\nStep 8: Loss computation...")
        diff_loss = nn.functional.mse_loss(
            predicted_noise * event_level_mask.unsqueeze(-1),
            actual_noise * event_level_mask.unsqueeze(-1),
            reduction='sum'
        ) / event_level_mask.sum()
        
        boundary_loss = model.boundary_predictor.compute_loss(
            bin_logits, predicted_length, true_length.float(), mask=event_level_mask
        )
        
        recon_loss = 0.0
        if enable_recon and config.get('recon_weight', 0) > 0:
            print("  Computing reconstruction loss...")
            boundary_length_for_mask = true_length
            positions = torch.arange(L, device=device).unsqueeze(0).unsqueeze(0)
            boundary_mask_train = (positions < boundary_length_for_mask.unsqueeze(-1)).float()
            combined_mask = event_mask * boundary_mask_train
            
            denoised_event_latent_for_recon = predicted_x0[..., :model.pattern_dim]
            denoised_time_latent_for_recon = predicted_x0[..., model.pattern_dim:2*model.pattern_dim]
            
            event_recon_loss, _ = model.event_decoder.compute_reconstruction_loss(
                denoised_event_latent_for_recon,
                input_ids, type_ids, dpe_ids,
                mask=combined_mask,
                target_length=boundary_length_for_mask
            )
            
            time_recon_loss, _ = model.time_decoder.compute_reconstruction_loss(
                denoised_time_latent_for_recon,
                con_time,
                mask=event_level_mask
            )
            
            recon_loss = event_recon_loss + time_recon_loss
        
        mem_after_loss = torch.cuda.memory_allocated()
        print(f"  Loss computation: {format_bytes(mem_after_loss - mem_after_boundary)}")
        
        total_loss = diff_loss + 0.5 * boundary_loss + 0.1 * recon_loss
    
    mem_before_backward = torch.cuda.memory_allocated()
    print(f"\n{'*'*70}")
    print(f"Memory BEFORE backward: {format_bytes(mem_before_backward)}")
    print(f"{'*'*70}")
    
    # ====================
    # 9. Backward Pass
    # ====================
    print("\nStep 9: Backward pass...")
    optimizer.zero_grad()
    
    scaler = torch.cuda.amp.GradScaler()
    scaler.scale(total_loss).backward()
    
    mem_after_backward = torch.cuda.memory_allocated()
    mem_peak = torch.cuda.max_memory_allocated()
    
    print(f"  Gradients: {format_bytes(mem_after_backward - mem_before_backward)}")
    print(f"\n{'='*70}")
    print(f"FINAL MEMORY USAGE:")
    print(f"  Allocated: {format_bytes(mem_after_backward)}")
    print(f"  Peak (max): {format_bytes(mem_peak)}")
    print(f"{'='*70}\n")
    
    # Cleanup
    del model, diffusion, optimizer, input_ids, total_loss
    torch.cuda.empty_cache()
    gc.collect()
    
    return mem_peak

def find_safe_batch_size(config):
    """Binary search for safe batch size"""
    
    print("\n" + "="*70)
    print("PHASE 1: Testing WITH reconstruction loss (recon_weight=0.1)")
    print("="*70)
    
    batch_sizes_to_test = [16, 24, 32, 40, 48, 56, 64]
    safe_batch_with_recon = 16
    
    for batch_size in batch_sizes_to_test:
        try:
            peak_mem = profile_training_step(batch_size, config, enable_recon=True)
            safe_batch_with_recon = batch_size
            print(f"\n✅ batch_size={batch_size} PASSED (peak: {format_bytes(peak_mem)})\n")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ batch_size={batch_size} OOM\n")
                torch.cuda.empty_cache()
                break
            else:
                raise e
    
    print("\n" + "="*70)
    print("PHASE 2: Testing WITHOUT reconstruction loss (recon_weight=0.0)")
    print("="*70)
    
    safe_batch_without_recon = 16
    
    for batch_size in batch_sizes_to_test:
        try:
            peak_mem = profile_training_step(batch_size, config, enable_recon=False)
            safe_batch_without_recon = batch_size
            print(f"\n✅ batch_size={batch_size} PASSED (peak: {format_bytes(peak_mem)})\n")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ batch_size={batch_size} OOM\n")
                torch.cuda.empty_cache()
                break
            else:
                raise e
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print(f"Safe batch size WITH recon (0.1):    {safe_batch_with_recon}")
    print(f"Safe batch size WITHOUT recon (0.0): {safe_batch_without_recon}")
    print("="*70)
    
    return safe_batch_with_recon, safe_batch_without_recon

if __name__ == '__main__':
    config = {
        'event_dim': 64,
        'time_dim': 32,
        'pattern_dim': 96,
        'num_prompts': 12,
        'hidden_dim': 256,
        'num_layers': 8,
        'num_heads': 4,
        'max_token_len': 128,
        'dropout': 0.1,
        'use_prompts': True,
        'use_demographics': False,
        'demographic_dim': 2,
        'boundary_weight': 0.5,
        'recon_weight': 0.1
    }
    
    find_safe_batch_size(config)