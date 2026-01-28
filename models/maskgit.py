import torch
import torch.nn as nn
from typing import Optional, Tuple

from models.embedder import CodeEmbedder
from models.decoder import CodeDecoder
from models.dit import DiT
from models.mask_schedule import random_mask


class EHRDiffusion(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.codebook_size = config["codebook_size"]
        self.mask_token_id = config["mask_token_id"]
        self.latent_dim = config["latent_dim"]
        self.num_codes = config["num_codes"]
        
        self.code_embedder = CodeEmbedder(
            vocab_size=config["vocab_size"],  # 1026
            codebook_size=config["codebook_size"],  # 1024
            rqvae_dim=config["rqvae_dim"],
            latent_dim=config["latent_dim"],
            num_codes=config["num_codes"],
            aggregation=config.get("code_aggregation", "mean"),
            freeze_codebook=config.get("freeze_codebook", False)
        )
        
        self.dit = DiT(config)
        
        self.code_decoder = CodeDecoder(
            latent_dim=config["latent_dim"],
            hidden_dim=config["hidden_dim"],
            codebook_size=config["codebook_size"],  # 1024
            num_codes=config["num_codes"],
            dropout=config["dropout"]
        )

    
    def load_rqvae_codebook(self, rqvae_checkpoint_path):
        self.code_embedder.load_rqvae_codebook(rqvae_checkpoint_path)
    
    def encode(self, codes):
        return self.code_embedder(codes)
    
    def decode(self, code_latent, return_logits=False):
        return self.code_decoder(code_latent, return_logits=return_logits)
    
    def forward_with_mask(
        self,
        codes: torch.Tensor,
        time_gaps: torch.Tensor,
        mask_ratio: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masked_codes, masked_positions = random_mask(
            codes, mask_ratio, self.mask_token_id, valid_mask
        )
        
        latent = self.encode(masked_codes)
        latent = self.dit(latent, mask_ratio, time_gaps, valid_mask)
        logits = self.decode(latent, return_logits=True)
        
        return logits, masked_positions
    
    def forward(self, codes, time_gaps, gamma, mask=None):
        latent = self.encode(codes)
        latent = self.dit(latent, gamma, time_gaps, mask)
        logits = self.decode(latent, return_logits=True)
        return logits