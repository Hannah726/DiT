"""
Code Embedder: Discrete codes -> Continuous latent
Converts RQ-VAE codes (8 discrete indices) to continuous latent vectors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CodeEmbedder(nn.Module):
    """
    Embeds discrete RQ-VAE codes into continuous latent space
    
    Args:
        codebook_size: Size of RQ-VAE codebook (default: 1024)
        rqvae_dim: Original RQ-VAE codebook dimension (default: 256)
        latent_dim: Target latent dimension for diffusion (default: 128)
        num_codes: Number of codes per event (default: 8)
        aggregation: How to aggregate codes ('mean', 'sum', 'max')
        freeze_codebook: Whether to freeze codebook during training
    """
    
    def __init__(
        self,
        codebook_size: int = 1024,
        rqvae_dim: int = 256,
        latent_dim: int = 128,
        num_codes: int = 8,
        aggregation: str = 'mean',
        freeze_codebook: bool = False
    ):
        super().__init__()
        
        self.codebook_size = codebook_size
        self.rqvae_dim = rqvae_dim
        self.latent_dim = latent_dim
        self.num_codes = num_codes
        self.aggregation = aggregation
        
        self.codebook = nn.Embedding(codebook_size, rqvae_dim)
        
        if freeze_codebook:
            self.codebook.weight.requires_grad = False
        
        if rqvae_dim != latent_dim:
            self.proj = nn.Sequential(
                nn.Linear(rqvae_dim, latent_dim),
                nn.LayerNorm(latent_dim)
            )
        else:
            self.proj = None
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
    
    def load_rqvae_codebook(self, rqvae_checkpoint_path):
        """
        Load pretrained codebook from RQ-VAE checkpoint
        
        Args:
            rqvae_checkpoint_path: Path to RQ-VAE checkpoint (.pkl file)
        """
        checkpoint = torch.load(rqvae_checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        codebook_keys = [
            'module.quantizer.embedding.weight',
            'quantizer.embedding.weight',
            'rq_vae.quantizer.embedding.weight',
            'embedding.weight'
        ]
        
        codebook_weight = None
        for key in codebook_keys:
            if key in state_dict:
                codebook_weight = state_dict[key]
                break
        
        if codebook_weight is None:
            available_keys = [k for k in state_dict.keys() if 'embedding' in k.lower()]
            if available_keys:
                print(f"Available embedding keys: {available_keys}")
                print(f"Trying first match: {available_keys[0]}")
                codebook_weight = state_dict[available_keys[0]]
            else:
                raise KeyError(
                    f"Could not find codebook in checkpoint. "
                    f"Available keys: {list(state_dict.keys())[:10]}"
                )
        
        if codebook_weight.shape[0] != self.codebook_size:
            raise ValueError(
                f"Codebook size mismatch: expected {self.codebook_size}, "
                f"got {codebook_weight.shape[0]}"
            )
        
        if codebook_weight.shape[1] != self.rqvae_dim:
            print(
                f"Warning: RQ-VAE dim mismatch. Expected {self.rqvae_dim}, "
                f"got {codebook_weight.shape[1]}. Using checkpoint dim."
            )
            self.rqvae_dim = codebook_weight.shape[1]
            self.codebook = nn.Embedding(self.codebook_size, self.rqvae_dim)
        
        self.codebook.weight.data.copy_(codebook_weight)
        print(f"Loaded RQ-VAE codebook: {codebook_weight.shape}")
    
    def forward(self, codes):
        """
        Args:
            codes: (B, N, 8) - discrete code indices from RQ-VAE
        
        Returns:
            (B, N, latent_dim) - continuous latent vectors
        """
        B, N, num_codes = codes.shape
        
        if num_codes != self.num_codes:
            raise ValueError(
                f"Expected {self.num_codes} codes per event, got {num_codes}"
            )
        
        code_emb = self.codebook(codes)
        
        if self.aggregation == 'mean':
            code_agg = code_emb.mean(dim=2)
        elif self.aggregation == 'sum':
            code_agg = code_emb.sum(dim=2)
        elif self.aggregation == 'max':
            code_agg = code_emb.max(dim=2)[0]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        if self.proj is not None:
            code_agg = self.proj(code_agg)
        
        return code_agg