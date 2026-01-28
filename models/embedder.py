import torch
import torch.nn as nn


class CodeEmbedder(nn.Module):
    
    def __init__(
        self,
        vocab_size: int = 1026,
        codebook_size: int = 1024,
        rqvae_dim: int = 256,
        latent_dim: int = 128,
        num_codes: int = 8,
        aggregation: str = 'sum',
        freeze_codebook: bool = False
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.codebook_size = codebook_size
        self.rqvae_dim = rqvae_dim
        self.latent_dim = latent_dim
        self.num_codes = num_codes
        self.aggregation = aggregation
        self.freeze_codebook = freeze_codebook
        
        # Use single embedding for both RQ-VAE codes and mask token
        self.codebook = nn.Embedding(vocab_size, rqvae_dim)
        
        # Don't freeze here - will handle in load_rqvae_codebook
        
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
            'quantize_model.layers.0._codebook.embed',
            'quantize_model.layers.1._codebook.embed',
            'module.quantizer.embedding.weight',
            'quantizer.embedding.weight',
        ]
        
        codebook_weight = None
        for key in codebook_keys:
            if key in state_dict:
                codebook_weight = state_dict[key]
                print(f"Found codebook at: {key}")
                break
        
        if codebook_weight is None:
            available_keys = [k for k in state_dict.keys() if 'codebook' in k.lower() or 'embed' in k.lower()]
            raise KeyError(
                f"Could not find codebook in checkpoint. "
                f"Available keys: {available_keys[:5]}"
            )
        
        # Reshape if needed: [1, 1024, 256] -> [1024, 256]
        if len(codebook_weight.shape) == 3:
            codebook_weight = codebook_weight.squeeze(0)
        
        if codebook_weight.shape[0] != self.codebook_size:
            raise ValueError(
                f"Codebook size mismatch: expected 1024, got {codebook_weight.shape[0]}"
            )
        
        if codebook_weight.shape[1] != self.rqvae_dim:
            print(
                f"Warning: RQ-VAE dim mismatch. Expected {self.rqvae_dim}, "
                f"got {codebook_weight.shape[1]}. Using checkpoint dim."
            )
            self.rqvae_dim = codebook_weight.shape[1]
            self.codebook = nn.Embedding(self.vocab_size, self.rqvae_dim)
        
        # Load only 0-1023 from RQ-VAE
        self.codebook.weight.data[:self.codebook_size].copy_(codebook_weight)
        print(f"Loaded RQ-VAE codebook: {codebook_weight.shape}")
        print(f"Embeddings {self.codebook_size} (padding) and {self.codebook_size+1} (mask) initialized randomly")
        
        
        if self.freeze_codebook:
            def freeze_rqvae_only(grad):
                grad[:1024] = 0
                return grad
            self.codebook.weight.register_hook(freeze_rqvae_only)
            print(f"Frozen RQ-VAE codes (0-{self.codebook_size-1}), embeddings {self.codebook_size}-{self.codebook_size+1} trainable")
        else:
            print("All embeddings trainable")
    
    def forward(self, codes):
        B, N, num_codes = codes.shape
        
        if num_codes != self.num_codes:
            raise ValueError(f"Expected {self.num_codes} codes per event, got {num_codes}")
        
        # Direct embedding lookup (handles mask token automatically)
        code_emb = self.codebook(codes)  # (B, N, num_codes, rqvae_dim)
        
        # Aggregate across num_codes dimension
        if self.aggregation == 'mean':
            code_agg = code_emb.mean(dim=2)
        elif self.aggregation == 'sum':
            code_agg = code_emb.sum(dim=2)
        elif self.aggregation == 'max':
            code_agg = code_emb.max(dim=2)[0]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Project if needed
        if self.proj is not None:
            code_agg = self.proj(code_agg)
        
        return code_agg