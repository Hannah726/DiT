import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalDecoder(nn.Module):

    def __init__(
        self,
        d_model: int = 256,
        codebook_size: int = 1024,
        spatial_dim: int = 4,
        num_quantizers: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.codebook_size = codebook_size
        self.spatial_dim = spatial_dim
        self.num_quantizers = num_quantizers
        self.hidden_dim = hidden_dim

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

        self.head_k1 = nn.Linear(hidden_dim, codebook_size)
        self.head_k2 = nn.Linear(hidden_dim, codebook_size)

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

    def forward(self, x, return_logits=False):
        B, N, _ = x.shape

        x = x.unsqueeze(2).expand(-1, -1, self.spatial_dim, -1)

        h = self.mlp(x)

        logits_k1 = self.head_k1(h)
        logits_k2 = self.head_k2(h)

        logits = torch.stack([logits_k1, logits_k2], dim=3)
        logits = logits.reshape(B, N, self.spatial_dim * self.num_quantizers, self.codebook_size)

        if return_logits:
            return logits
        else:
            codes = logits.argmax(dim=-1)
            return codes

    def compute_loss(self, x, target_codes, mask=None, label_smoothing=0.0):
        B, N, num_total = target_codes.shape

        logits = self.forward(x, return_logits=True)

        logits_flat = logits.reshape(B * N * num_total, self.codebook_size)
        target_flat = target_codes.reshape(B * N * num_total)

        loss = F.cross_entropy(
            logits_flat,
            target_flat,
            reduction='none',
            label_smoothing=label_smoothing
        )
        loss = loss.reshape(B, N, num_total)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, num_total)
            loss = (loss * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
        else:
            loss = loss.mean()

        with torch.no_grad():
            pred_codes = logits.argmax(dim=-1)

            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand(-1, -1, num_total)
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
