"""
Event Decoder: Latent -> Structured Tokens
Reconstructs (token, type, dpe) from event latents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class EventDecoder(nn.Module):
    """
    Decodes event latent vectors back to structured token sequences
    
    Architecture:
        1. Expand latent to hidden dimension
        2. Generate token-level features
        3. Predict token/type/dpe distributions
        4. Predict mask (valid vs padding tokens)
    
    Args:
        event_dim: Input event latent dimension
        hidden_dim: Hidden layer dimension
        vocab_sizes: Dict of vocabulary sizes for each channel
        max_token_len: Maximum tokens per event
        dropout: Dropout rate
    
    Note:
        The decoder now learns to predict which token positions are valid (not padding).
        This is critical for generating realistic sequences with proper padding.
    """
    
    def __init__(
        self,
        event_dim: int = 64,
        hidden_dim: int = 256,
        vocab_sizes: dict = None,  # {'token': int, 'type': int, 'dpe': int}
        max_token_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if vocab_sizes is None:
            vocab_sizes = {'token': 2385, 'type': 7, 'dpe': 15}
        
        self.event_dim = event_dim
        self.hidden_dim = hidden_dim
        self.vocab_sizes = vocab_sizes
        self.max_token_len = max_token_len
        
        # Expand event latent
        self.latent_proj = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Generate token-level representations
        self.token_expander = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * max_token_len)
        )
        
        # Transformer for token-level refinement
        self.token_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Output heads for each channel
        self.token_head = nn.Linear(hidden_dim, vocab_sizes['token'])
        self.type_head = nn.Linear(hidden_dim, vocab_sizes['type'])
        self.dpe_head = nn.Linear(hidden_dim, vocab_sizes['dpe'])
        
        # Mask prediction head: predicts whether each token position is valid (not padding)
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, event_latents, return_logits=False):
        """
        Args:
            event_latents: (B, N, event_dim) - event latent vectors
            return_logits: Whether to return raw logits or sampled IDs
        
        Returns:
            If return_logits=True:
                dict with keys 'token', 'type', 'dpe', 'mask'
                - 'token', 'type', 'dpe': (B, N, L, vocab_size) - logits
                - 'mask': (B, N, L) - mask logits (before sigmoid)
            Else:
                dict with keys 'token', 'type', 'dpe', 'mask'
                - 'token', 'type', 'dpe': (B, N, L) - sampled IDs (padding positions set to 0)
                - 'mask': (B, N, L) - predicted mask (1 for valid tokens, 0 for padding)
        """
        B, N, _ = event_latents.shape
        L = self.max_token_len
        
        # Project event latents
        x = self.latent_proj(event_latents)  # (B, N, hidden_dim)
        
        # Expand to token level
        x_expanded = self.token_expander(x)  # (B, N, hidden_dim * L)
        x_expanded = rearrange(x_expanded, 'b n (l d) -> b n l d', l=L)  # (B, N, L, hidden_dim)
        
        # Flatten for transformer
        x_flat = rearrange(x_expanded, 'b n l d -> (b n) l d')  # (B*N, L, hidden_dim)
        
        # Refine with transformer
        x_refined = self.token_transformer(x_flat)  # (B*N, L, hidden_dim)
        
        # Reshape back
        x_refined = rearrange(x_refined, '(b n) l d -> b n l d', b=B, n=N)  # (B, N, L, hidden_dim)
        
        # Predict token/type/dpe
        token_logits = self.token_head(x_refined)  # (B, N, L, vocab_token)
        type_logits = self.type_head(x_refined)    # (B, N, L, vocab_type)
        dpe_logits = self.dpe_head(x_refined)      # (B, N, L, vocab_dpe)
        
        # Predict mask (valid token positions)
        mask_logits = self.mask_head(x_refined)  # (B, N, L, 1)
        mask_logits = mask_logits.squeeze(-1)     # (B, N, L)
        
        if return_logits:
            return {
                'token': token_logits,
                'type': type_logits,
                'dpe': dpe_logits,
                'mask': mask_logits
            }
        else:
            # Sample IDs (greedy decoding)
            token_ids = token_logits.argmax(dim=-1)  # (B, N, L)
            type_ids = type_logits.argmax(dim=-1)
            dpe_ids = dpe_logits.argmax(dim=-1)
            
            # Predict mask: sigmoid to get probabilities, then threshold at 0.5
            mask_probs = torch.sigmoid(mask_logits)  # (B, N, L)
            predicted_mask = (mask_probs > 0.5).float()  # (B, N, L)
            
            # Apply mask: set padding positions to 0
            token_ids = token_ids * predicted_mask.long()
            type_ids = type_ids * predicted_mask.long()
            dpe_ids = dpe_ids * predicted_mask.long()
            
            return {
                'token': token_ids,
                'type': type_ids,
                'dpe': dpe_ids,
                'mask': predicted_mask
            }
    
    def compute_reconstruction_loss(self, event_latents, target_tokens, target_types, target_dpes, mask=None):
        """
        Compute cross-entropy loss for reconstruction and mask prediction
        
        Args:
            event_latents: (B, N, event_dim)
            target_tokens: (B, N, L) - ground truth token IDs
            target_types: (B, N, L) - ground truth type IDs
            target_dpes: (B, N, L) - ground truth DPE IDs
            mask: (B, N, L) - valid token mask (1 for valid, 0 for padding)
        
        Returns:
            loss: Scalar reconstruction loss
            loss_dict: Dictionary of loss components
        """
        # Get logits (including mask logits)
        logits = self.forward(event_latents, return_logits=True)
        
        B, N, L = target_tokens.shape
        
        # Create target mask from target_tokens if not provided
        if mask is None:
            # Valid tokens are those > 0
            mask = (target_tokens > 0).float()  # (B, N, L)
        
        # Reshape for cross entropy
        token_logits = rearrange(logits['token'], 'b n l v -> (b n l) v')
        type_logits = rearrange(logits['type'], 'b n l v -> (b n l) v')
        dpe_logits = rearrange(logits['dpe'], 'b n l v -> (b n l) v')
        mask_logits = rearrange(logits['mask'], 'b n l -> (b n l)')  # (B*N*L,)
        
        target_tokens_flat = rearrange(target_tokens, 'b n l -> (b n l)')
        target_types_flat = rearrange(target_types, 'b n l -> (b n l)')
        target_dpes_flat = rearrange(target_dpes, 'b n l -> (b n l)')
        mask_flat = rearrange(mask, 'b n l -> (b n l)')  # (B*N*L,)
        
        # Compute token/type/dpe losses
        token_loss = F.cross_entropy(token_logits, target_tokens_flat, reduction='none')
        type_loss = F.cross_entropy(type_logits, target_types_flat, reduction='none')
        dpe_loss = F.cross_entropy(dpe_logits, target_dpes_flat, reduction='none')
        
        # Apply mask to token/type/dpe losses (only compute loss on valid tokens)
        token_loss = (token_loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        type_loss = (type_loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        dpe_loss = (dpe_loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        
        # Compute mask prediction loss (binary classification: valid vs padding)
        # Use BCE with logits for numerical stability
        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits, 
            mask_flat, 
            reduction='mean'
        )
        
        # Total loss: reconstruction losses + mask prediction loss
        total_loss = token_loss + type_loss + dpe_loss + mask_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'token': token_loss.item(),
            'type': type_loss.item(),
            'dpe': dpe_loss.item(),
            'mask': mask_loss.item()
        }
        
        return total_loss, loss_dict