"""
Pattern Discovery Prompts for Event-Time Joint Modeling
Self-learning prompts that discover event-time association patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatternDiscoveryPrompts(nn.Module):
    """
    Pattern Discovery via Learnable Prompts
    
    Core Innovation:
        - K learnable prompts represent different event-time patterns
        - Events and times cross-attend to prompts to discover patterns
        - Prompts mediate bidirectional information flow
        - NO manual definition needed - patterns emerge from data
    
    Flow:
        1. Event → Prompts: Which patterns does this event activate?
        2. Time → Prompts: Which patterns does this time relate to?
        3. Cross-modal refinement:
           - Event enriched by time's patterns
           - Time enriched by event's patterns
    
    Args:
        event_dim: Event latent dimension
        time_dim: Time embedding dimension
        hidden_dim: Unified hidden dimension after projection
        num_prompts: Number of learnable patterns (K)
        num_heads: Attention heads for cross-attention
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        event_dim: int = 64,
        time_dim: int = 32,
        hidden_dim: int = 96,
        num_prompts: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.event_dim = event_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.num_prompts = num_prompts
        self.num_heads = num_heads
        
        # Learnable prompts: K pattern vectors
        # Initialized with small random values
        self.prompts = nn.Parameter(torch.randn(num_prompts, hidden_dim) * 0.02)
        
        # Project event and time to unified hidden_dim
        self.event_proj = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Cross-attention: Event/Time attend to Prompts
        self.event_to_prompts = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.time_to_prompts = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion: Combine original + cross-modal context
        self.event_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.time_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
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
    
    def forward(self, event_latent, time_emb, mask=None):
        """
        Args:
            event_latent: (B, N, event_dim) - encoded events
            time_emb: (B, N, time_dim) - encoded times
            mask: (B, N) - valid event mask (1=valid, 0=padding)
        
        Returns:
            event_refined: (B, N, hidden_dim) - events + time patterns
            time_refined: (B, N, hidden_dim) - times + event patterns
            prompt_weights: (B, N, K) - pattern activation weights
        """
        B, N = event_latent.shape[:2]
        
        # Project to unified space
        event_h = self.event_proj(event_latent)  # (B, N, D)
        time_h = self.time_proj(time_emb)        # (B, N, D)
        
        # Expand prompts for batch
        prompts = self.prompts.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)
        
        # Convert mask for attention (True = padding)
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask.bool()  # (B, N)
        
        # Step 1: Event attends to Prompts
        # Q: what patterns does this event activate?
        # K,V: prompt patterns
        event_context, event_attn = self.event_to_prompts(
            query=event_h,
            key=prompts,
            value=prompts,
            key_padding_mask=None,  # Prompts always valid
            need_weights=True,
            average_attn_weights=True
        )
        # event_context: (B, N, D) - prompt info relevant to each event
        # event_attn: (B, N, K) - which prompts are activated
        
        # Step 2: Time attends to Prompts
        time_context, time_attn = self.time_to_prompts(
            query=time_h,
            key=prompts,
            value=prompts,
            key_padding_mask=None,
            need_weights=True,
            average_attn_weights=True
        )
        # time_context: (B, N, D) - prompt info relevant to each time
        # time_attn: (B, N, K)
        
        # Step 3: Cross-modal refinement
        # Event enriched by time's activated patterns
        event_refined = event_h + time_context
        event_refined = self.event_fusion(event_refined)
        
        # Time enriched by event's activated patterns
        time_refined = time_h + event_context
        time_refined = self.time_fusion(time_refined)
        
        # Average attention weights for downstream use
        prompt_weights = (event_attn + time_attn) / 2  # (B, N, K)
        
        return event_refined, time_refined, prompt_weights