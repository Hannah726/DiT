"""
Minimal Adaptive Prompt Generator (Demographics Only)
Generates patient-specific prompts based on age and sex
"""

import torch
import torch.nn as nn


class AdaptivePromptGenerator(nn.Module):
    """
    Minimal Adaptive Prompt Generator
    
    Generates two types of prompts:
    1. Global learnable prompts - capture universal patterns in data
    2. Demographic-adaptive prompts - patient-specific based on age/sex
    
    Total prompts per patient: 32 (global) + 8 (demographic) = 40
    
    Args:
        demographic_dim: Input demographic dimension (default: 2 for age + sex)
        num_global_prompts: Number of global learnable prompts
        num_demographic_prompts: Number of demographic-conditioned prompts
        prompt_dim: Dimension of each prompt (should match latent_dim)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        demographic_dim: int = 2,
        num_global_prompts: int = 32,
        num_demographic_prompts: int = 8,
        prompt_dim: int = 96,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.demographic_dim = demographic_dim
        self.num_global_prompts = num_global_prompts
        self.num_demographic_prompts = num_demographic_prompts
        self.prompt_dim = prompt_dim
        
        # Global learnable prompts - capture universal patterns
        self.global_prompts = nn.Parameter(
            torch.randn(1, num_global_prompts, prompt_dim) * 0.02
        )
        
        # Demographic encoder - maps (age, sex) to patient-specific prompts
        self.demographic_encoder = nn.Sequential(
            nn.Linear(demographic_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_demographic_prompts * prompt_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with Xavier uniform"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, demographics):
        """
        Generate adaptive prompts for a batch of patients
        
        Args:
            demographics: (B, demographic_dim) 
                         demographics[:, 0] = normalized age [0, 1]
                         demographics[:, 1] = sex (0 or 1)
        
        Returns:
            prompts: (B, num_global_prompts + num_demographic_prompts, prompt_dim)
        """
        B = demographics.shape[0]
        
        global_p = self.global_prompts.expand(B, -1, -1)
        demo_emb = self.demographic_encoder(demographics)
        demo_p = demo_emb.view(B, self.num_demographic_prompts, self.prompt_dim)
        
        all_prompts = torch.cat([global_p, demo_p], dim=1)
        return all_prompts
    
    def get_num_prompts(self):
        """Return total number of prompts per patient"""
        return self.num_global_prompts + self.num_demographic_prompts
    
    def get_num_parameters(self):
        """Return number of trainable parameters in prompt generator"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)