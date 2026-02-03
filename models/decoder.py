import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalDecoder(nn.Module):

    def __init__(
        self,
        d_model: int = 256,
        codebook_size: int = 1024,
        num_codes: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.codebook_size = codebook_size
        self.num_codes = num_codes

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, codebook_size) for _ in range(num_codes)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x, return_logits=False):
        B, N, _ = x.shape
        h = self.mlp(x) # (B, N, hidden_dim)
        
        logits = [head(h) for head in self.heads] # List of (B, N, V)
        logits = torch.stack(logits, dim=2) # (B, N, L, V)

        if return_logits:
            return logits
        else:
            return logits.argmax(dim=-1)
