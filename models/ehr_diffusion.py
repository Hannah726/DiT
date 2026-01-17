import torch
import torch.nn as nn
from typing import Optional, Dict

from models.embedder import CodeEmbedder
from models.decoder import CodeDecoder
from models.dit import DiT


class EHRDiffusion(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.codebook_size = config["codebook_size"]
        self.latent_dim = config["latent_dim"]
        self.num_codes = config["num_codes"]
        
        self.code_embedder = CodeEmbedder(
            codebook_size=config["codebook_size"],
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
            codebook_size=config["codebook_size"],
            num_codes=config["num_codes"],
            dropout=config["dropout"]
        )
    
    def load_rqvae_codebook(self, rqvae_checkpoint_path):
        self.code_embedder.load_rqvae_codebook(rqvae_checkpoint_path)
    
    def encode(self, codes):
        return self.code_embedder(codes)
    
    def decode(self, code_latent, return_logits=False):
        return self.code_decoder(code_latent, return_logits=return_logits)
    
    def denoise(self, noisy_latent, t, time_ids, mask=None):
        return self.dit(noisy_latent, t, time_ids, mask)
    
    def training_step(self, codes, time_ids, mask=None):
        code_latent = self.encode(codes)
        code_logits = self.decode(code_latent, return_logits=True)
        predicted_codes = code_logits.argmax(dim=-1)
        return {
            'code_latent': code_latent,
            'predicted_codes': predicted_codes,
            'code_logits': code_logits
        }
    
    def forward(self, codes, time_ids=None, mask=None):
        output = self.training_step(codes, time_ids, mask)
        return output['predicted_codes']