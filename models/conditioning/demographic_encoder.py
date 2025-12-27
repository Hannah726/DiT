"""
Demographic Encoder for Patient Attributes
Encodes age, sex, and other static features into conditioning vectors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DemographicEncoder(nn.Module):
    """
    Encoder for patient demographic information
    
    Supports:
        - Age (continuous, normalized)
        - Sex (binary)
        - Extensible to other demographics
    
    Args:
        demographic_dim: Input dimension (default: 2 for age + sex)
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
        dropout: Dropout rate
        use_embedding: Whether to use embeddings for categorical features
    """
    
    def __init__(
        self,
        demographic_dim: int = 2,
        hidden_dim: int = 128,
        output_dim: int = 256,
        dropout: float = 0.1,
        use_embedding: bool = False
    ):
        super().__init__()
        
        self.demographic_dim = demographic_dim
        self.output_dim = output_dim
        self.use_embedding = use_embedding
        
        if use_embedding:
            # Separate processing for continuous and categorical
            self.age_proj = nn.Linear(1, hidden_dim // 2)
            self.sex_embedding = nn.Embedding(2, hidden_dim // 2)
            input_dim = hidden_dim
        else:
            input_dim = demographic_dim
        
        # MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, demographics):
        """
        Args:
            demographics: (B, demographic_dim) - patient demographics
                         demographics[:, 0] = age (normalized to [0, 1])
                         demographics[:, 1] = sex (0 or 1)
        
        Returns:
            (B, output_dim) - demographic embeddings
        """
        if self.use_embedding:
            # Separate age and sex
            age = demographics[:, 0:1]  # (B, 1)
            sex = demographics[:, 1].long()  # (B,)
            
            # Project age
            age_emb = self.age_proj(age)  # (B, hidden_dim // 2)
            
            # Embed sex
            sex_emb = self.sex_embedding(sex)  # (B, hidden_dim // 2)
            
            # Concatenate
            x = torch.cat([age_emb, sex_emb], dim=-1)  # (B, hidden_dim)
        else:
            x = demographics
        
        # Encode
        emb = self.encoder(x)  # (B, output_dim)
        
        return emb


class ConditionalProjection(nn.Module):
    """
    Projects demographic embeddings to different conditioning formats
    Supports multiple conditioning mechanisms (AdaLN, cross-attention, etc.)
    """
    
    def __init__(
        self,
        demographic_dim: int = 256,
        target_dim: int = 512,
        num_projections: int = 1
    ):
        super().__init__()
        
        self.projections = nn.ModuleList([
            nn.Linear(demographic_dim, target_dim)
            for _ in range(num_projections)
        ])
    
    def forward(self, demographic_emb, projection_idx=0):
        """
        Args:
            demographic_emb: (B, demographic_dim)
            projection_idx: Which projection to use
        
        Returns:
            (B, target_dim) - projected embeddings
        """
        return self.projections[projection_idx](demographic_emb)