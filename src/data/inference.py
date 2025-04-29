import torch
from torch import Tensor
from typing import List

from model import Seq2SeqTransformer
from utils import generate_square_subsequent_mask
from data_loader import text_transform, vocab_transform, BOS_IDX, EOS_IDX

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def greedy_decode(
    model: Seq2SeqTransformer,
    src: Tensor,
    src_mask: Tensor,
    max_len: int,
    start_symbol: int
) -> Tensor:
    memory = model.encode(src.to(device), src_mask.to(device))
    ys = torch.tensor([[start_symbol]], device=device)
    for _ in range(max_len-1):
        tgt_mask = generate_square_subsequent_mask(ys.size(0), device).bool()
        out = model.decode(ys, memory, tgt_mask)
        prob = model.generator(out[-1, :])
        next_word = torch.argmax(prob, dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def translate_sentence(model: Seq2SeqTransformer, sentence: str) -> str:
    model.eval()
    tokens = text_transform['de'](sentence)
    src = torch.tensor(tokens).unsqueeze(1)
    src_mask = torch.zeros(src.size(0), src.size(0), dtype=torch.bool)
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=src.size(0)+5, start_symbol=BOS_IDX).flatten()
    words = vocab_transform['en'].lookup_tokens(tgt_tokens.cpu().tolist())
    return " ".join(w for w in words if w not in ['<bos>','<eos>'])