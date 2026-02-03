import torch
import torch.nn as nn


class HierarchicalEmbedder(nn.Module):

    def __init__(
        self,
        codebook_size: int = 1024,
        mask_token_id: int = 1024,
        rqvae_dim: int = 256,
        num_codes: int = 8,
        d_model: int = 256,
        freeze_codebook: bool = False
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.mask_token_id = mask_token_id
        self.rqvae_dim = rqvae_dim
        self.num_codes = num_codes
        self.freeze_codebook = freeze_codebook

        vocab_size = codebook_size + 1
        self.embedding = nn.Embedding(vocab_size, rqvae_dim)
        self.proj = nn.Linear(num_codes * rqvae_dim, d_model)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def load_rqvae_codebook(self, rqvae_checkpoint_path):
        checkpoint = torch.load(rqvae_checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        w0, w1 = None, None
        for key in state_dict.keys():
            if 'layers.0' in key and 'embed' in key:
                w0 = state_dict[key]
            elif 'layers.1' in key and 'embed' in key:
                w1 = state_dict[key]

        if w0 is None or w1 is None:
            raise KeyError("Cannot find codebook layers in checkpoint")

        if len(w0.shape) == 3: w0 = w0.squeeze(0)
        if len(w1.shape) == 3: w1 = w1.squeeze(0)

        avg_weight = (w0 + w1) / 2.0
        self.embedding.weight.data[:self.codebook_size].copy_(avg_weight)

        if self.freeze_codebook:
            self.embedding.weight.requires_grad = False

    def forward(self, codes):
        B, N, L = codes.shape
        e = self.embedding(codes) # (B, N, L, rqvae_dim)
        e = e.view(B, N, -1) # (B, N, L * rqvae_dim)
        return self.proj(e)
