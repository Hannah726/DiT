import torch
import torch.nn as nn


class CodeEmbedder(nn.Module):
    
    def __init__(
        self,
        codebook_size: int = 1025,
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
                raise KeyError(f"Could not find codebook in checkpoint")
        
        if codebook_weight.shape[0] != 1024:
            raise ValueError(
                f"Codebook size mismatch: expected 1024, got {codebook_weight.shape[0]}"
            )
        
        if codebook_weight.shape[1] != self.rqvae_dim:
            print(
                f"Warning: RQ-VAE dim mismatch. Expected {self.rqvae_dim}, "
                f"got {codebook_weight.shape[1]}. Using checkpoint dim."
            )
            self.rqvae_dim = codebook_weight.shape[1]
            self.codebook = nn.Embedding(self.codebook_size, self.rqvae_dim)
        
        self.codebook.weight.data[:1024].copy_(codebook_weight)
        print(f"Loaded RQ-VAE codebook: {codebook_weight.shape}")
        print(f"Mask token embedding initialized randomly")
    
    def forward(self, codes):
        B, N, num_codes = codes.shape
        
        if num_codes != self.num_codes:
            raise ValueError(f"Expected {self.num_codes} codes per event, got {num_codes}")
        
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