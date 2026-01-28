import torch
import torch.nn as nn
import torch.nn.functional as F


class CodeDecoder(nn.Module):
    
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        codebook_size: int = 1024,
        num_codes: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.num_codes = num_codes
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.code_heads = nn.ModuleList([
            nn.Linear(hidden_dim, codebook_size)
            for _ in range(num_codes)
        ])
        
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
    
    def forward(self, latent, return_logits=False):
        B, N, _ = latent.shape
        
        h = self.mlp(latent)
        
        logits_list = []
        for head in self.code_heads:
            logits = head(h)
            logits_list.append(logits)
        
        logits = torch.stack(logits_list, dim=2)
        
        if return_logits:
            return logits
        else:
            codes = logits.argmax(dim=-1)
            return codes
    
    def compute_loss(self, latent, target_codes, mask=None):
        B, N, num_codes = target_codes.shape
        
        logits = self.forward(latent, return_logits=True)
        
        logits_flat = logits.reshape(B * N * num_codes, self.codebook_size)
        target_flat = target_codes.reshape(B * N * num_codes)
        
        loss = F.cross_entropy(logits_flat, target_flat, reduction='none')
        loss = loss.reshape(B, N, num_codes)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, num_codes)
            loss = (loss * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        with torch.no_grad():
            pred_codes = logits.argmax(dim=-1)
            
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand(-1, -1, num_codes)
                correct = ((pred_codes == target_codes) * mask_expanded).sum()
                total = mask_expanded.sum()
            else:
                correct = (pred_codes == target_codes).sum()
                total = pred_codes.numel()
            
            accuracy = correct.float() / (total + 1e-8)
        
        loss_dict = {
            'code_loss': loss.item(),
            'code_acc': accuracy.item()
        }
        
        return loss, loss_dict