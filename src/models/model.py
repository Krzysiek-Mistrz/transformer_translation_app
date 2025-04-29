import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).unsqueeze(1)
        pe = torch.zeros(maxlen, emb_size)
        pe[:, 0::2] = torch.sin(pos * den)
        pe[:, 1::2] = torch.cos(pos * den)
        self.pos_embedding = pe.unsqueeze(1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pos_embedding[:x.size(0), :]
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens) * math.sqrt(self.emb_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(
        self, num_encoder_layers: int, num_decoder_layers: int,
        emb_size: int, nhead: int, src_vocab_size: int, tgt_vocab_size: int,
        dim_feedforward: int = 512, dropout: float = 0.1
    ):
        super().__init__()
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.pos_enc = PositionalEncoding(emb_size, dropout)
        self.transformer = Transformer(
            d_model=emb_size, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask,
                src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.pos_enc(self.src_tok_emb(src))
        tgt_emb = self.pos_enc(self.tgt_tok_emb(tgt))
        outs = self.transformer(
            src_emb, tgt_emb, src_mask, tgt_mask, None,
            src_padding_mask, tgt_padding_mask, memory_key_padding_mask
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        return self.transformer.encoder(self.pos_enc(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor) -> Tensor:
        return self.transformer.decoder(self.pos_enc(self.tgt_tok_emb(tgt)), memory, tgt_mask)