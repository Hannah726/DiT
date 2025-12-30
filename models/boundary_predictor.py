"""
Boundary Distribution Predictor
Predicts probability distribution over sequence lengths
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryDistributionPredictor(nn.Module):
    """
    Boundary (Length) Distribution Predictor
    
    Predicts: P(length | event, prompts) for each event
    
    Key Design:
        - Conditioned on BOTH event latent AND prompt patterns
        - Outputs distribution over [0, max_len]
        - Supports deterministic (argmax) or stochastic (sample) inference
    
    Args:
        event_dim: Event latent dimension (from PatternDiscoveryPrompts output)
        prompt_dim: Prompt dimension (should match PatternDiscoveryPrompts.hidden_dim)
        hidden_dim: MLP hidden dimension
        max_len: Maximum sequence length (e.g., 128)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        event_dim: int = 96,
        prompt_dim: int = 96,
        hidden_dim: int = 128,
        max_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.event_dim = event_dim
        self.prompt_dim = prompt_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        # Combine event + prompt summary
        input_dim = event_dim + prompt_dim
        
        # MLP predictor
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_len + 1)  # +1 for length=0 (all padding)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, event_latent, prompt_weights, prompts):
        """
        Args:
            event_latent: (B, N, event_dim) - refined event latents
            prompt_weights: (B, N, K) - pattern activation weights
            prompts: (K, prompt_dim) - learnable prompts (from PatternDiscoveryPrompts)
        
        Returns:
            length_logits: (B, N, max_len+1) - raw logits
            length_dist: (B, N, max_len+1) - probability distribution
        """
        # Compute prompt summary as weighted sum
        # prompt_weights: (B, N, K), prompts: (K, D)
        # -> (B, N, D)
        prompts_expanded = prompts.unsqueeze(0).unsqueeze(0)  # (1, 1, K, D)
        prompt_weights_expanded = prompt_weights.unsqueeze(-1)  # (B, N, K, 1)
        
        prompt_summary = (prompt_weights_expanded * prompts_expanded).sum(dim=2)  # (B, N, D)
        
        # Combine event and prompt
        combined = torch.cat([event_latent, prompt_summary], dim=-1)  # (B, N, event_dim+prompt_dim)
        
        # Predict length distribution
        length_logits = self.predictor(combined)  # (B, N, max_len+1)
        length_dist = F.softmax(length_logits, dim=-1)
        
        return length_logits, length_dist
    
    def sample_length(self, length_dist, temperature=1.0, deterministic=False):
        """
        Sample or select sequence length
        
        Args:
            length_dist: (B, N, max_len+1) - probability distribution
            temperature: Sampling temperature (lower = more deterministic)
            deterministic: If True, use argmax
        
        Returns:
            (B, N) - predicted lengths [0, max_len]
        """
        if deterministic:
            # Argmax: most probable length
            return torch.argmax(length_dist, dim=-1)
        else:
            # Sample with temperature
            logits = torch.log(length_dist + 1e-10) / temperature
            B, N, L = logits.shape
            
            # Reshape and sample
            logits_flat = logits.view(B * N, L)
            probs_flat = F.softmax(logits_flat, dim=-1)
            samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)  # (B*N,)
            
            return samples.view(B, N)
    
    def compute_loss(self, length_logits, true_length, mask=None):
        """
        Cross-entropy loss for length prediction
        
        Args:
            length_logits: (B, N, max_len+1) - predicted logits
            true_length: (B, N) - ground truth lengths [0, max_len]
            mask: (B, N) - valid event mask
        
        Returns:
            loss: Scalar loss
        """
        B, N = true_length.shape
        
        # Flatten
        logits_flat = length_logits.view(B * N, -1)
        targets_flat = true_length.view(B * N)
        
        # Cross-entropy
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # (B*N,)
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(B * N)
            loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss