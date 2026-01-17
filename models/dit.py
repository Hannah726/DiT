import torch
import torch.nn as nn
import math


class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config["latent_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.dropout = config["dropout"]
        
        self.time_condition_dim = config.get("time_condition_dim", self.hidden_dim)
        self.max_time_vocab = config["max_time_vocab"]
        self.time_token_len = config["time_token_len"]
        self.max_event_size = config["max_event_size"]
        
        self.time_embedding = nn.Embedding(
            self.max_time_vocab, 
            self.time_condition_dim
        )
        
        self.time_position_embedding = nn.Embedding(
            self.time_token_len,
            self.time_condition_dim
        )
        
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_condition_dim, self.time_condition_dim),
            nn.LayerNorm(self.time_condition_dim),
            nn.GELU()
        )
        
        self.input_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )
        
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                condition_dim=self.time_condition_dim,
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
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x, t, time_ids, mask=None):
        B, N, time_len = time_ids.shape
        
        time_emb = self.time_embedding(time_ids)
        
        positions = torch.arange(time_len, device=time_ids.device)
        pos_emb = self.time_position_embedding(positions)
        time_emb = time_emb + pos_emb.unsqueeze(0).unsqueeze(0)
        
        time_condition = time_emb.mean(dim=2)
        time_condition = self.time_proj(time_condition)
        
        t_emb = self.timestep_embedding(t, self.hidden_dim)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)
        
        h = self.input_proj(x)
        h = h + t_emb
        
        for block in self.blocks:
            h = block(h, time_condition, mask)
        
        out = self.output_proj(h)
        
        return out
    
    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, condition_dim, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True, 
            kdim=condition_dim, vdim=condition_dim
        )
        
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, time_condition, mask=None):
        attn_mask = ~mask if mask is not None else None
        
        x = x + self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=attn_mask
        )[0]
        
        x = x + self.cross_attn(
            self.norm2(x), time_condition, time_condition,
            key_padding_mask=attn_mask
        )[0]
        
        x = x + self.mlp(self.norm3(x))
        
        return x