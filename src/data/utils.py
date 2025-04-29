import torch
from torch import Tensor
from typing import Tuple

# Must match data_loader's PAD_IDX
PAD_IDX = 1

def generate_square_subsequent_mask(sz: int, device: torch.device = torch.device('cpu')) -> Tensor:
    """Square mask for causal decoding (upper-triangular)."""
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
    return mask

def create_mask(
    src: Tensor, tgt: Tensor, pad_idx: int = PAD_IDX, 
    device: torch.device = torch.device('cpu')
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Returns src_mask, tgt_mask, src_padding_mask, tgt_padding_mask.
    """
    src_seq_len, tgt_seq_len = src.size(0), tgt.size(0)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).bool()
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device).bool()
    src_padding_mask = (src == pad_idx).transpose(0,1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0,1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask