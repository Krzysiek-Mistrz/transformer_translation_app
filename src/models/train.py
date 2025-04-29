import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from typing import Tuple

from src.models.model import Seq2SeqTransformer
from data.data_loader import get_translation_dataloaders, vocab_transform
from data.utils import create_mask, PAD_IDX

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model: Seq2SeqTransformer, optimizer, dataloader) -> float:
    model.train()
    total_loss = 0.0
    loss_fn = CrossEntropyLoss(ignore_index=PAD_IDX)
    for src, tgt in tqdm(dataloader, desc="Train", leave=False):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt_input, device=device)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask, src_pad_mask)
        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model: Seq2SeqTransformer, dataloader) -> float:
    model.eval()
    total_loss = 0.0
    loss_fn = CrossEntropyLoss(ignore_index=PAD_IDX)
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt_input, device=device)
            logits = model(src, tgt_input, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask, src_pad_mask)
            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_model(
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-4
) -> None:
    train_loader, val_loader = get_translation_dataloaders(batch_size=batch_size)
    src_vocab = len(vocab_transform['de'])
    tgt_vocab = len(vocab_transform['en'])
    model = Seq2SeqTransformer(
        num_encoder_layers=3, num_decoder_layers=3,
        emb_size=512, nhead=8, src_vocab_size=src_vocab, tgt_vocab_size=tgt_vocab
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9,0.98), eps=1e-9)

    for epoch in range(1, epochs+1):
        train_loss = train_epoch(model, optimizer, train_loader)
        val_loss   = evaluate(model, val_loader)
        print(f"Epoch {epoch}/{epochs} — train: {train_loss:.3f}, val: {val_loss:.3f}")
    torch.save(model.state_dict(), "transformer_de_to_en.pt")
    print("Model saved to transformer_de_to_en.pt")