import torch
import numpy as np


def cosine_schedule(step, total_steps, min_ratio=0.05, max_ratio=0.95):
    progress = step / max(total_steps - 1, 1)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
    return min_ratio + (max_ratio - min_ratio) * cosine_decay


def linear_schedule(step, total_steps, min_ratio=0.05, max_ratio=0.95):
    progress = step / max(total_steps - 1, 1)
    return max_ratio - (max_ratio - min_ratio) * progress


def nested_random_mask(codes, mask_ratio, mask_token_id=1024, valid_mask=None, k2_ratio=0.3):
    B, N, L = codes.shape
    device = codes.device

    if isinstance(mask_ratio, float):
        mask_ratio = torch.full((B,), mask_ratio, device=device)
    
    noise_event = torch.rand(B, N, device=device)
    if valid_mask is not None:
        noise_event = torch.where(valid_mask.bool(), noise_event, torch.ones_like(noise_event) * 2.0)
    
    num_valid = valid_mask.sum(dim=1) if valid_mask is not None else torch.full((B,), N, device=device)
    num_mask_event = (num_valid * mask_ratio).long().clamp(min=1)
    
    sorted_noise, _ = torch.sort(noise_event, dim=1)
    threshold_event = sorted_noise.gather(1, (num_mask_event - 1).view(-1, 1).clamp(min=0, max=N-1))
    mask_event = noise_event <= threshold_event # (B, N)
    
    # 2. Partial-level mask (K2 only or subset of codes)
    noise_code = torch.rand(B, N, L, device=device)
    # Only apply partial mask to unmasked events
    partial_eligible = (~mask_event)
    if valid_mask is not None:
        partial_eligible = partial_eligible & valid_mask.bool()
    
    noise_code = torch.where(partial_eligible.unsqueeze(-1), noise_code, torch.ones_like(noise_code) * 2.0)
    num_mask_code = (L * k2_ratio) # Fixed ratio for codes within an event
    num_mask_code = int(max(1, num_mask_code))
    
    sorted_noise_code, _ = torch.sort(noise_code, dim=2)
    threshold_code = sorted_noise_code[:, :, num_mask_code-1:num_mask_code]
    mask_code = noise_code <= threshold_code # (B, N, L)
    
    # 3. Combine
    final_mask = mask_event.unsqueeze(-1) | mask_code
    masked_codes = codes.clone()
    masked_codes[final_mask] = mask_token_id
    
    return masked_codes, final_mask


def sample_mask_ratio(schedule_fn, total_steps, batch_size=1, device='cpu'):
    steps = torch.randint(0, total_steps, (batch_size,), device=device)
    mask_ratios = torch.tensor(
        [schedule_fn(s.item(), total_steps) for s in steps],
        device=device, dtype=torch.float32
    )
    return mask_ratios


def unmask_by_confidence(codes, logits, num_unmask, mask_token_id=1024):
    B, N, L, V = logits.shape
    device = codes.device
    
    # Hannah, we use temperature-based multinomial sampling to break the "Code 1023" dominance
    # 1. Apply temperature and get probabilities
    # logits: (B, N, L, V)
    probs = torch.softmax(logits, dim=-1)
    
    # 2. Sample from distribution instead of argmax
    B, N, L, V = logits.shape
    probs_flat = probs.view(-1, V)
    predictions_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
    predictions = predictions_flat.view(B, N, L)
    
    # 3. Get confidence (max prob) for scheduling
    confidence, _ = probs.max(dim=-1) # (B, N, L)
    
    is_masked = (codes == mask_token_id)
    # Only consider confidence for currently masked positions
    confidence = torch.where(is_masked, confidence, torch.zeros_like(confidence) - 1.0)
    
    conf_flat = confidence.view(B, -1)
    # Ensure we don't try to unmask more than available
    actual_masked_per_sample = is_masked.sum().item() // B
    num_unmask = min(num_unmask, actual_masked_per_sample)
    
    if num_unmask <= 0:
        return codes

    # 4. Pick the most confident positions to unmask
    _, topk_indices = torch.topk(conf_flat, k=num_unmask, dim=1)
    
    # 5. Update only the selected positions
    updates = predictions.view(B, -1).gather(1, topk_indices)
    codes_flat = codes.view(B, -1)
    codes_flat.scatter_(1, topk_indices, updates)
    
    return codes_flat.view(B, N, L)
