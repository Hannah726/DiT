import torch
import torch.nn as nn
from typing import Optional, Tuple

from models.embedder import HierarchicalEmbedder
from models.decoder import HierarchicalDecoder
from models.dit import DiT
from models.mask_schedule import nested_random_mask


class EHRDiffusion(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.codebook_size = config["codebook_size"]
        self.mask_token_id = config["mask_token_id"]
        self.d_model = config["d_model"]
        self.num_codes = config["num_codes"]
        self.num_quantizers = config["num_quantizers"]
        self.label_smoothing = config.get("label_smoothing", 0.1)
        self.k2_ratio = config.get("nested_mask_k2_ratio", 0.3)
        
        self.embedder = HierarchicalEmbedder(
            codebook_size=config["codebook_size"],
            mask_token_id=config["mask_token_id"],
            rqvae_dim=config["rqvae_dim"],
            num_quantizers=config["num_quantizers"],
            freeze_codebook=config.get("freeze_codebook", False)
        )
        
        self.dit = DiT(config)
        
        self.decoder = HierarchicalDecoder(
            d_model=config["d_model"],
            codebook_size=config["codebook_size"],
            num_quantizers=config["num_quantizers"],
            hidden_dim=config.get("hidden_dim", 512),
            dropout=config["dropout"]
        )
    
    def load_rqvae_codebook(self, rqvae_checkpoint_path):
        self.embedder.load_rqvae_codebook(rqvae_checkpoint_path)
    
    def encode(self, codes):
        return self.embedder(codes)
    
    def decode(self, x, return_logits=False):
        return self.decoder(x, return_logits=return_logits)
    
    def forward_with_mask(
        self,
        codes: torch.Tensor,
        time_gaps: torch.Tensor,
        mask_ratio: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masked_codes, masked_positions = nested_random_mask(
            codes, mask_ratio, self.mask_token_id, valid_mask, self.k2_ratio
        )
        
        x = self.encode(masked_codes)
        x = self.dit(x, mask_ratio, time_gaps, valid_mask)
        logits = self.decode(x, return_logits=True)
        
        return logits, masked_positions
    
    def forward(self, codes, time_gaps, gamma, mask=None):
        x = self.encode(codes)
        x = self.dit(x, gamma, time_gaps, mask)
        logits = self.decode(x, return_logits=True)
        return logits
    
    def compute_loss(
        self,
        codes: torch.Tensor,
        time_gaps: torch.Tensor,
        mask_ratio: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        import torch.nn.functional as F
        
        logits, masked_positions = self.forward_with_mask(
            codes, time_gaps, mask_ratio, valid_mask
        )
        
        B, N, num_q, vocab_size = logits.shape
        
        if valid_mask is not None:
            valid_mask_expanded = valid_mask.bool().unsqueeze(-1).expand_as(masked_positions)
            loss_mask = masked_positions & valid_mask_expanded
        else:
            loss_mask = masked_positions
        
        if loss_mask.sum() == 0:
            return torch.tensor(0.0, device=codes.device), {
                'loss': 0.0,
                'mask_acc': 0.0,
                'num_masked': 0
            }
        
        logits_for_loss = logits[loss_mask].view(-1, self.codebook_size)
        targets_for_loss = codes[loss_mask].view(-1)
        
        loss = F.cross_entropy(
            logits_for_loss,
            targets_for_loss,
            reduction='mean',
            label_smoothing=self.label_smoothing
        )
        
        with torch.no_grad():
            pred_codes = logits[loss_mask].argmax(dim=-1)
            target_codes = codes[loss_mask]
            accuracy = (pred_codes == target_codes).float().mean()
            num_masked = loss_mask.sum().item()
        
        loss_dict = {
            'loss': loss.item(),
            'mask_acc': accuracy.item(),
            'num_masked': num_masked
        }
        
        return loss, loss_dict