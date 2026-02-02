import torch
import torch.nn as nn


class HierarchicalEmbedder(nn.Module):
    
    def __init__(
        self,
        codebook_size: int = 1024,
        mask_token_id: int = 1024,
        rqvae_dim: int = 256,
        num_quantizers: int = 2,
        freeze_codebook: bool = False
    ):
        super().__init__()
        
        self.codebook_size = codebook_size
        self.mask_token_id = mask_token_id
        self.rqvae_dim = rqvae_dim
        self.num_quantizers = num_quantizers
        self.freeze_codebook = freeze_codebook
        
        vocab_size = codebook_size + 1
        self.codebook_0 = nn.Embedding(vocab_size, rqvae_dim)
        self.codebook_1 = nn.Embedding(vocab_size, rqvae_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def load_rqvae_codebook(self, rqvae_checkpoint_path):
        checkpoint = torch.load(rqvae_checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        codebook_0_weight = None
        codebook_1_weight = None
        
        for key in state_dict.keys():
            if 'layers.0' in key and 'embed' in key:
                codebook_0_weight = state_dict[key]
            elif 'layers.1' in key and 'embed' in key:
                codebook_1_weight = state_dict[key]
        
        if codebook_0_weight is None or codebook_1_weight is None:
            raise KeyError("Cannot find both codebook layers in checkpoint")
        
        if len(codebook_0_weight.shape) == 3:
            codebook_0_weight = codebook_0_weight.squeeze(0)
        if len(codebook_1_weight.shape) == 3:
            codebook_1_weight = codebook_1_weight.squeeze(0)
        
        self.codebook_0.weight.data[:self.codebook_size].copy_(codebook_0_weight)
        self.codebook_1.weight.data[:self.codebook_size].copy_(codebook_1_weight)
        
        print(f"Loaded codebook_0: {codebook_0_weight.shape}")
        print(f"Loaded codebook_1: {codebook_1_weight.shape}")
        
        if self.freeze_codebook:
            print("Frozen RQ-VAE codes, MASK token trainable")
    
    def forward(self, codes):
        B, N, num_q = codes.shape
        assert num_q == self.num_quantizers
        
        k1 = codes[:, :, 0]
        k2 = codes[:, :, 1]
        
        e1 = self.codebook_0(k1)
        e2 = self.codebook_1(k2)
        
        if self.freeze_codebook:
            is_mask_1 = (k1 == self.mask_token_id)
            is_mask_2 = (k2 == self.mask_token_id)
            e1 = torch.where(is_mask_1.unsqueeze(-1), e1, e1.detach())
            e2 = torch.where(is_mask_2.unsqueeze(-1), e2, e2.detach())
        
        return e1 + e2