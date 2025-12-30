"""
Event Decoder: Latent -> Structured Tokens
Reconstructs (token, type, dpe) from event latents with boundary constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class EventDecoder(nn.Module):
    """
    Decodes event latent vectors back to structured token sequences
    
    Key Updates:
        - Supports boundary-constrained generation
        - Can apply predicted length masks during decoding
        - Maintains original multi-channel prediction
    
    Architecture:
        1. Expand latent to hidden dimension
        2. Generate token-level features
        3. Predict token/type/dpe distributions
    
    Args:
        event_dim: Input event latent dimension
        hidden_dim: Hidden layer dimension
        vocab_sizes: Dict of vocabulary sizes for each channel
        max_token_len: Maximum tokens per event
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        event_dim: int = 96,  # Changed from 64 to match refined event dim
        hidden_dim: int = 256,
        vocab_sizes: dict = None,
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
        
        # Length embedding for length-aware decoding
        self.length_embedding = nn.Embedding(max_token_len + 1, hidden_dim)
        
        # Expand event latent (now with length information)
        self.latent_proj = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Length-aware projection: combine event latent with length embedding
        self.length_aware_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # event + length
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
    
    def forward(self, event_latents, boundary_mask=None, target_length=None, return_logits=False):
        """
        Args:
            event_latents: (B, N, event_dim) - event latent vectors
            boundary_mask: (B, N, L) - optional boundary constraint mask
                          1 for valid positions, 0 for padding
            target_length: (B, N) - optional target length for length-aware decoding
                          If None, will be inferred from boundary_mask
            return_logits: Whether to return raw logits or sampled IDs
        
        Returns:
            If return_logits=True:
                dict with keys 'token', 'type', 'dpe'
                - Each: (B, N, L, vocab_size) - logits
            Else:
                dict with keys 'token', 'type', 'dpe'
                - Each: (B, N, L) - sampled IDs with padding=0
        """
        B, N, _ = event_latents.shape
        L = self.max_token_len
        
        # Project event latents
        event_h = self.latent_proj(event_latents)  # (B, N, hidden_dim)
        
        # Length-aware decoding: incorporate target length information
        if target_length is not None:
            # Embed target length
            length_emb = self.length_embedding(target_length)  # (B, N, hidden_dim)
            # Combine event and length information
            x = self.length_aware_proj(torch.cat([event_h, length_emb], dim=-1))  # (B, N, hidden_dim)
        elif boundary_mask is not None:
            # Infer length from boundary_mask
            inferred_length = boundary_mask.sum(dim=-1).long()  # (B, N)
            inferred_length = torch.clamp(inferred_length, 0, self.max_token_len)
            length_emb = self.length_embedding(inferred_length)  # (B, N, hidden_dim)
            x = self.length_aware_proj(torch.cat([event_h, length_emb], dim=-1))  # (B, N, hidden_dim)
        else:
            # No length information, use event latent only
            x = event_h
        
        # Expand to token level
        x_expanded = self.token_expander(x)  # (B, N, hidden_dim * L)
        x_expanded = rearrange(x_expanded, 'b n (l d) -> b n l d', l=L)
        
        # Flatten for transformer
        x_flat = rearrange(x_expanded, 'b n l d -> (b n) l d')
        
        # Refine with transformer
        x_refined = self.token_transformer(x_flat)
        
        # Reshape back
        x_refined = rearrange(x_refined, '(b n) l d -> b n l d', b=B, n=N)
        
        # Predict token/type/dpe
        token_logits = self.token_head(x_refined)  # (B, N, L, vocab_token)
        type_logits = self.type_head(x_refined)
        dpe_logits = self.dpe_head(x_refined)
        
        if return_logits:
            return {
                'token': token_logits,
                'type': type_logits,
                'dpe': dpe_logits
            }
        else:
            # Sample IDs (greedy decoding)
            token_ids = token_logits.argmax(dim=-1)  # (B, N, L)
            type_ids = type_logits.argmax(dim=-1)
            dpe_ids = dpe_logits.argmax(dim=-1)
            
            # Apply boundary mask if provided
            if boundary_mask is not None:
                mask_long = boundary_mask.long()
                token_ids = token_ids * mask_long
                type_ids = type_ids * mask_long
                dpe_ids = dpe_ids * mask_long
            
            return {
                'token': token_ids,
                'type': type_ids,
                'dpe': dpe_ids
            }
    
    def compute_reconstruction_loss(self, event_latents, target_tokens, 
                                   target_types, target_dpes, mask=None, target_length=None):
        """
        Compute cross-entropy loss for reconstruction
        
        Args:
            event_latents: (B, N, event_dim)
            target_tokens: (B, N, L) - ground truth token IDs
            target_types: (B, N, L) - ground truth type IDs
            target_dpes: (B, N, L) - ground truth DPE IDs
            mask: (B, N, L) - valid token mask (1 for valid, 0 for padding)
            target_length: (B, N) - target length for length-aware decoding (optional)
        
        Returns:
            loss: Scalar reconstruction loss
            loss_dict: Dictionary of loss components
        """
        # Get logits with length-aware decoding
        logits = self.forward(event_latents, boundary_mask=mask, target_length=target_length, return_logits=True)
        
        B, N, L = target_tokens.shape
        
        # Create mask from target_tokens if not provided
        if mask is None:
            mask = (target_tokens > 0).float()
        
        # Reshape for cross entropy
        token_logits = rearrange(logits['token'], 'b n l v -> (b n l) v')
        type_logits = rearrange(logits['type'], 'b n l v -> (b n l) v')
        dpe_logits = rearrange(logits['dpe'], 'b n l v -> (b n l) v')
        
        target_tokens_flat = rearrange(target_tokens, 'b n l -> (b n l)')
        target_types_flat = rearrange(target_types, 'b n l -> (b n l)')
        target_dpes_flat = rearrange(target_dpes, 'b n l -> (b n l)')
        mask_flat = rearrange(mask, 'b n l -> (b n l)')
        
        # Compute losses
        token_loss = F.cross_entropy(token_logits, target_tokens_flat, reduction='none')
        type_loss = F.cross_entropy(type_logits, target_types_flat, reduction='none')
        dpe_loss = F.cross_entropy(dpe_logits, target_dpes_flat, reduction='none')
        
        # Apply mask
        token_loss = (token_loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        type_loss = (type_loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        dpe_loss = (dpe_loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        
        # Total loss
        total_loss = token_loss + type_loss + dpe_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'token': token_loss.item(),
            'type': type_loss.item(),
            'dpe': dpe_loss.item()
        }
        
        return total_loss, loss_dict