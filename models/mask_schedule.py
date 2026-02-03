import torch
import numpy as np


def cosine_schedule(step, total_steps, min_ratio=0.05, max_ratio=0.95):
    progress = step / max(total_steps - 1, 1)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
    return min_ratio + (max_ratio - min_ratio) * cosine_decay


def linear_schedule(step, total_steps, min_ratio=0.05, max_ratio=0.95):
    progress = step / max(total_steps - 1, 1)
    return max_ratio - (max_ratio - min_ratio) * progress


def nested_random_mask(codes, mask_ratio, mask_token_id=1024, valid_mask=None, k2_independent_ratio=0.3, spatial_dim=4, num_quantizers=2):
    B, N, total_codes = codes.shape
    device = codes.device

    masked_codes = codes.clone()
    mask_positions = torch.zeros_like(codes, dtype=torch.bool)

    if isinstance(mask_ratio, float):
        mask_ratio = torch.full((B,), mask_ratio, device=device)

    for b in range(B):
        if valid_mask is not None:
            valid_pos = valid_mask[b]
            num_valid = valid_pos.sum().item()
        else:
            valid_pos = torch.ones(N, dtype=torch.bool, device=device)
            num_valid = N

        if num_valid == 0:
            continue

        num_mask = max(1, int(num_valid * mask_ratio[b].item()))
        valid_indices = torch.where(valid_pos)[0]
        perm = torch.randperm(len(valid_indices), device=device)
        mask_indices = valid_indices[perm[:num_mask]]

        masked_codes[b, mask_indices, :] = mask_token_id
        mask_positions[b, mask_indices, :] = True

        unmasked = valid_pos.clone()
        unmasked[mask_indices] = False
        unmasked_indices = torch.where(unmasked)[0]

        if len(unmasked_indices) > 0:
            num_k2_mask = max(1, int(len(unmasked_indices) * k2_independent_ratio))
            perm_k2 = torch.randperm(len(unmasked_indices), device=device)
            k2_mask_indices = unmasked_indices[perm_k2[:num_k2_mask]]

            for s in range(spatial_dim):
                col = s * num_quantizers + 1
                masked_codes[b, k2_mask_indices, col] = mask_token_id
                mask_positions[b, k2_mask_indices, col] = True

    return masked_codes, mask_positions


def sample_mask_ratio(schedule_fn, total_steps, batch_size=1, device='cpu'):
    steps = torch.randint(0, total_steps, (batch_size,), device=device)
    mask_ratios = torch.tensor(
        [schedule_fn(s.item(), total_steps) for s in steps],
        device=device, dtype=torch.float32
    )
    return mask_ratios


def unmask_by_confidence(codes, logits, num_unmask, mask_token_id=1024):
    B, N, num_total, V = logits.shape
    updated_codes = codes.clone()

    probs = torch.softmax(logits, dim=-1)
    confidence, predictions = probs.max(dim=-1)

    for b in range(B):
        masked = (codes[b] == mask_token_id)

        if masked.sum() == 0:
            continue

        masked_confidence = confidence[b][masked]
        k = min(num_unmask, len(masked_confidence))

        if k == 0:
            continue

        topk_values, topk_local_indices = torch.topk(masked_confidence, k)
        masked_indices = torch.where(masked.flatten())[0]
        topk_global_indices = masked_indices[topk_local_indices]

        topk_n = topk_global_indices // num_total
        topk_c = topk_global_indices % num_total

        updated_codes[b, topk_n, topk_c] = predictions[b, topk_n, topk_c]

    return updated_codes
