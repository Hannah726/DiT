"""
MDLM: Masked Diffusion Language Model for EHR Generation

Core innovation: Combining BERT-style masked language modeling 
with diffusion-inspired iterative refinement for EHR token generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class TimeEmbedding(nn.Module):
    """
    Embed time information (discrete or continuous)
    """
    def __init__(self, hidden_dim: int, discrete: bool = True, 
                 time_vocab_size: int = 10, time_dim: int = 2):
        super().__init__()
        self.discrete = discrete
        
        if discrete:
            # Time as tokens (e.g., [0-9] × 2 digits)
            self.time_embed = nn.Embedding(time_vocab_size, hidden_dim)
            self.time_dim = time_dim
        else:
            # Time as continuous value
            self.time_mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
    
    def forward(self, time):
        """
        Args:
            time: (B, 243, time_dim) if discrete, (B, 243, 1) if continuous
        Returns:
            (B, 243, hidden_dim)
        """
        if self.discrete:
            # time: (B, 243, 2) with values 0-9
            # Embed each digit and sum/mean
            emb = self.time_embed(time)  # (B, 243, 2, D)
            return emb.mean(dim=2)  # (B, 243, D)
        else:
            return self.time_mlp(time)  # (B, 243, D)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        """
        return x + self.pe[:, :x.size(1)]


class MDLM(nn.Module):
    """
    Masked Diffusion Language Model
    
    Combines:
    - Masked token prediction (BERT-style)
    - Iterative refinement (diffusion-inspired)
    - Multi-modal EHR tokens (text/type/dpe)
    """
    
    def __init__(
        self,
        vocab_size_text: int = 2000,  # reduced vocab
        vocab_size_type: int = 100,
        vocab_size_dpe: int = 20,
        max_events: int = 243,
        max_tokens_per_event: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        time_discrete: bool = True,
        time_vocab_size: int = 10,
        time_dim: int = 2,
        mask_token_id: int = 0  # Reserve 0 for [MASK]
    ):
        super().__init__()
        
        self.vocab_size_text = vocab_size_text
        self.vocab_size_type = vocab_size_type
        self.vocab_size_dpe = vocab_size_dpe
        self.max_events = max_events
        self.max_tokens_per_event = max_tokens_per_event
        self.hidden_dim = hidden_dim
        self.mask_token_id = mask_token_id
        self.time_discrete = time_discrete
        
        # Token embeddings for each modality
        self.text_embed = nn.Embedding(vocab_size_text, hidden_dim)
        self.type_embed = nn.Embedding(vocab_size_type, hidden_dim)
        self.dpe_embed = nn.Embedding(vocab_size_dpe, hidden_dim)
        
        # Time embedding
        self.time_embed = TimeEmbedding(
            hidden_dim, 
            discrete=time_discrete,
            time_vocab_size=time_vocab_size,
            time_dim=time_dim
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_events)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Prediction heads
        self.text_head = nn.Linear(hidden_dim, vocab_size_text)
        self.type_head = nn.Linear(hidden_dim, vocab_size_type)
        self.dpe_head = nn.Linear(hidden_dim, vocab_size_dpe)
        
        if time_discrete:
            self.time_head = nn.Linear(hidden_dim, time_vocab_size)
        else:
            self.time_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            )
    
    def create_mask_schedule(self, t: float, strategy: str = 'linear'):
        """
        Create masking ratio based on diffusion timestep t ∈ [0, 1]
        
        t=0: no masking (clean data)
        t=1: full masking
        
        Args:
            t: diffusion timestep
            strategy: 'linear', 'cosine', 'sqrt'
        """
        if strategy == 'linear':
            return t
        elif strategy == 'cosine':
            return 1 - math.cos(t * math.pi / 2)
        elif strategy == 'sqrt':
            return math.sqrt(t)
        else:
            return t
    
    def apply_masking(
        self, 
        tokens: torch.Tensor, 
        time_data: torch.Tensor,
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply masking to tokens
        
        Args:
            tokens: (B, max_events, max_tokens, 3) - [text, type, dpe]
            time_data: (B, max_events, time_dim)
            mask_ratio: fraction of tokens to mask
        
        Returns:
            masked_tokens: (B, max_events, max_tokens, 3)
            masked_time: (B, max_events, time_dim)
            mask: (B, max_events, max_tokens) - binary mask
        """
        B, E, T, _ = tokens.shape
        
        # Create random mask
        mask = torch.rand(B, E, T, device=tokens.device) < mask_ratio
        
        # Apply mask to tokens
        masked_tokens = tokens.clone()
        masked_tokens[mask] = self.mask_token_id
        
        # Also mask time
        time_mask = torch.rand(B, E, device=tokens.device) < mask_ratio
        masked_time = time_data.clone()
        if self.time_discrete:
            masked_time[time_mask] = self.mask_token_id
        else:
            masked_time[time_mask] = 0.0  # or some sentinel value
        
        return masked_tokens, masked_time, mask, time_mask
    
    def forward(
        self,
        tokens: torch.Tensor,
        time_data: torch.Tensor,
        mask_ratio: Optional[float] = None,
        event_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training
        
        Args:
            tokens: (B, max_events, max_tokens, 3) - [text, type, dpe]
            time_data: (B, max_events, time_dim)
            mask_ratio: if provided, apply masking. Otherwise use tokens as-is
            event_mask: (B, max_events) - binary mask for valid events
        
        Returns:
            Dictionary with predictions and losses
        """
        B, E, T, _ = tokens.shape
        
        # Apply masking if mask_ratio provided
        if mask_ratio is not None:
            masked_tokens, masked_time, token_mask, time_mask = \
                self.apply_masking(tokens, time_data, mask_ratio)
        else:
            masked_tokens = tokens
            masked_time = time_data
            token_mask = None
            time_mask = None
        
        # Flatten event and token dimensions for embedding
        # (B, E, T, 3) → (B, E*T, 3)
        flat_tokens = masked_tokens.reshape(B, E * T, 3)
        
        # Embed each modality
        text_emb = self.text_embed(flat_tokens[:, :, 0])  # (B, E*T, D)
        type_emb = self.type_embed(flat_tokens[:, :, 1])
        dpe_emb = self.dpe_embed(flat_tokens[:, :, 2])
        
        # Combine modalities (simple addition)
        token_emb = text_emb + type_emb + dpe_emb  # (B, E*T, D)
        
        # Reshape back to (B, E, T, D) and pool over tokens
        token_emb = token_emb.reshape(B, E, T, self.hidden_dim)
        event_emb = token_emb.mean(dim=2)  # (B, E, D)
        
        # Add time embedding
        time_emb = self.time_embed(masked_time)  # (B, E, D)
        combined = event_emb + time_emb
        
        # Add positional encoding
        combined = self.pos_encoder(combined)  # (B, E, D)
        
        # Create attention mask for padding
        if event_mask is not None:
            # event_mask: (B, E) - 1 for valid, 0 for padding
            # Transformer expects: 0 for valid, -inf for padding
            attn_mask = ~event_mask.bool()  # (B, E)
        else:
            attn_mask = None
        
        # Transformer
        h = self.transformer(
            combined,
            src_key_padding_mask=attn_mask
        )  # (B, E, D)
        
        # Expand back to token level for prediction
        h_expanded = h.unsqueeze(2).expand(B, E, T, self.hidden_dim)
        h_flat = h_expanded.reshape(B, E * T, self.hidden_dim)
        
        # Predict original tokens
        pred_text = self.text_head(h_flat)  # (B, E*T, vocab_text)
        pred_type = self.type_head(h_flat)
        pred_dpe = self.dpe_head(h_flat)
        
        # Predict time
        pred_time = self.time_head(h)  # (B, E, vocab_time) or (B, E, 1)
        
        # Reshape predictions
        pred_text = pred_text.reshape(B, E, T, self.vocab_size_text)
        pred_type = pred_type.reshape(B, E, T, self.vocab_size_type)
        pred_dpe = pred_dpe.reshape(B, E, T, self.vocab_size_dpe)
        
        result = {
            'pred_text': pred_text,
            'pred_type': pred_type,
            'pred_dpe': pred_dpe,
            'pred_time': pred_time,
            'token_mask': token_mask,
            'time_mask': time_mask
        }
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        n_steps: int = 100,
        temperature: float = 1.0,
        device: str = 'cuda'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate EHR sequences via iterative refinement
        
        Args:
            batch_size: number of sequences to generate
            n_steps: number of refinement steps
            temperature: sampling temperature
            device: device to run on
        
        Returns:
            tokens: (batch_size, max_events, max_tokens, 3)
            time: (batch_size, max_events, time_dim)
        """
        # Start with all [MASK] tokens
        tokens = torch.full(
            (batch_size, self.max_events, self.max_tokens_per_event, 3),
            self.mask_token_id,
            dtype=torch.long,
            device=device
        )
        
        if self.time_discrete:
            time_data = torch.full(
                (batch_size, self.max_events, 2),
                self.mask_token_id,
                dtype=torch.long,
                device=device
            )
        else:
            time_data = torch.zeros(
                (batch_size, self.max_events, 1),
                device=device
            )
        
        # Iteratively unmask
        for step in range(n_steps):
            # Progress from fully masked (t=1) to unmasked (t=0)
            t = 1.0 - (step + 1) / n_steps
            
            # Forward pass
            output = self.forward(tokens, time_data, mask_ratio=None)
            
            # Sample from predictions
            pred_text = output['pred_text'] / temperature
            pred_type = output['pred_type'] / temperature
            pred_dpe = output['pred_dpe'] / temperature
            
            # Determine which tokens to unmask this step
            # Gradually unmask more tokens as we progress
            unmask_ratio = (step + 1) / n_steps
            current_mask = tokens[:, :, :, 0] == self.mask_token_id
            to_unmask = torch.rand_like(current_mask.float()) < unmask_ratio
            to_unmask = to_unmask & current_mask  # Only unmask currently masked
            
            # Sample and update
            if to_unmask.any():
                tokens[:, :, :, 0][to_unmask] = torch.argmax(
                    pred_text[to_unmask], dim=-1
                )
                tokens[:, :, :, 1][to_unmask] = torch.argmax(
                    pred_type[to_unmask], dim=-1
                )
                tokens[:, :, :, 2][to_unmask] = torch.argmax(
                    pred_dpe[to_unmask], dim=-1
                )
            
            # Update time
            if self.time_discrete:
                time_mask = time_data[:, :, 0] == self.mask_token_id
                time_to_unmask = torch.rand_like(time_mask.float()) < unmask_ratio
                time_to_unmask = time_to_unmask & time_mask
                
                if time_to_unmask.any():
                    pred_time = output['pred_time'] / temperature  # (B, E, vocab_time)
                    # Sample first digit
                    sampled_digit = torch.argmax(
                        pred_time[time_to_unmask], dim=-1
                    )
                    time_data[:, :, 0][time_to_unmask] = sampled_digit
                    # For simplicity, use same prediction for second digit
                    # TODO: Implement separate prediction head for second digit
                    if time_data.shape[2] > 1:
                        time_data[:, :, 1][time_to_unmask] = sampled_digit
            else:
                time_data = output['pred_time']
        
        return tokens, time_data


if __name__ == "__main__":
    model = MDLM(
        vocab_size_text=2000,
        vocab_size_type=100,
        vocab_size_dpe=20,
        hidden_dim=128,
        num_layers=3
    )
    
    # Test forward
    B, E, T = 2, 243, 128
    tokens = torch.randint(0, 2000, (B, E, T, 3))
    time = torch.randint(0, 10, (B, E, 2))
    
    output = model(tokens, time, mask_ratio=0.15)
    print("Output keys:", output.keys())
    print("Text predictions shape:", output['pred_text'].shape)
    
    # Test generation
    gen_tokens, gen_time = model.generate(batch_size=2, n_steps=10, device='cpu')
    print("Generated tokens shape:", gen_tokens.shape)
    print("Generated time shape:", gen_time.shape)
