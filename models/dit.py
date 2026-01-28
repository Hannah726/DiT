import torch
import torch.nn as nn


class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config["latent_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.dropout = config["dropout"]
        
        self.time_dim = config.get("time_dim", 1)
        self.time_proj_dim = config.get("time_proj_dim", 128)
        self.time_pad_value = config.get("time_pad_value", -1.0)
        
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_dim, self.time_proj_dim),
            nn.LayerNorm(self.time_proj_dim),
            nn.GELU(),
            nn.Linear(self.time_proj_dim, self.hidden_dim)
        )
        
        self.gamma_proj = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.input_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        self.output_proj = nn.Linear(self.hidden_dim, self.latent_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x, gamma, time_gaps, mask=None):
        B, N, _ = x.shape
        
        time_gaps_clean = time_gaps.clone()

        if mask is not None:
            time_gaps_clean[~mask.bool().unsqueeze(-1).expand_as(time_gaps)] = -1.0
        
        time_condition = self.time_proj(time_gaps_clean)

        if mask is not None:
            time_condition = time_condition * mask.unsqueeze(-1)
        
        if gamma.dim() == 1:
            gamma = gamma.unsqueeze(-1)
        
        gamma_emb = self.gamma_proj(gamma)
        gamma_emb = gamma_emb.unsqueeze(1).expand(-1, N, -1)
        
        h = self.input_proj(x)
        h = h + gamma_emb + time_condition
        
        for block in self.blocks:
            h = block(h, mask)
        
        out = self.output_proj(h)
        
        return out


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x, mask=None):
        if mask is not None:
            attn_mask = ~mask.bool()
        else:
            attn_mask = None
        
        x = x + self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=attn_mask
        )[0]
        
        x = x + self.mlp(self.norm2(x))

        return x