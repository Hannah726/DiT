import torch
import torch.nn as nn


class AdaptiveLayerNorm(nn.Module):
    
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.linear = nn.Linear(cond_dim, 2 * d_model)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x, cond):
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
        
        params = self.linear(cond)
        shift, scale = params.chunk(2, dim=-1)
        
        return self.ln(x) * (1 + scale) + shift


class DiTBlock(nn.Module):
    
    def __init__(self, d_model, num_heads, cond_dim, dropout=0.1):
        super().__init__()
        
        self.adaln_attn = AdaptiveLayerNorm(d_model, cond_dim)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.adaln_mlp = AdaptiveLayerNorm(d_model, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.gate_mlp = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, cond, mask=None):
        h = self.adaln_attn(x, cond)
        h = self.self_attn(h, h, h, key_padding_mask=mask)[0]
        x = x + self.gate_attn * h
        
        h = self.adaln_mlp(x, cond)
        h = self.mlp(h)
        x = x + self.gate_mlp * h
        
        return x


class DiT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.d_model = config["d_model"]
        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.dropout = config["dropout"]
        self.use_adaln = config.get("use_adaln", True)
        
        self.time_dim = config.get("time_dim", 1)
        self.time_proj_dim = config.get("time_proj_dim", 128)
        
        cond_dim = self.d_model + self.time_proj_dim
        
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_dim, self.time_proj_dim),
            nn.GELU(),
            nn.Linear(self.time_proj_dim, self.time_proj_dim)
        )
        
        self.gamma_proj = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
        self.input_proj = nn.Linear(self.d_model, self.d_model)
        
        self.max_len = config.get("max_event_size", 256)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.blocks = nn.ModuleList([
            DiTBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                cond_dim=cond_dim,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.d_model)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x, gamma, time_gaps, mask=None):
        B, N, _ = x.shape
        
        time_cond = self.time_proj(time_gaps)
        
        if mask is not None:
            time_cond = time_cond * mask.unsqueeze(-1)
        
        if mask is not None:
            time_cond_pooled = (time_cond * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        else:
            time_cond_pooled = time_cond.mean(dim=1)
        
        if gamma.dim() == 1:
            gamma = gamma.unsqueeze(-1)
        gamma_cond = self.gamma_proj(gamma)
        
        cond = torch.cat([gamma_cond, time_cond_pooled], dim=-1)
        
        h = x + self.pos_embed[:, :N, :]
        h = self.input_proj(h)
        
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask.bool()
        
        for block in self.blocks:
            h = block(h, cond, attn_mask)
        
        h = self.output_norm(h)
        out = self.output_proj(h)
        
        return out