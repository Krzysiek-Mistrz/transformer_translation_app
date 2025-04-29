import torch
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from typing import Tuple, Dict, Callable

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

vocab_transform: Dict[str, any] = {}
text_transform: Dict[str, Callable[[str], torch.Tensor]] = {}

def index_to_german(tokens: torch.Tensor) -> str:
    """Convert German token indices back to a string."""
    raise NotImplementedError

def index_to_english(tokens: torch.Tensor) -> str:
    """Convert English token indices back to a string."""
    raise NotImplementedError

def get_translation_dataloaders(
    batch_size: int = 128,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Download Multi30k, build vocab & text transforms,
    and return train/val DataLoaders that yield (src, tgt) tensors.
    """
    train_iter = Multi30k(split='train', language_pair=('de', 'en'))
    val_iter   = Multi30k(split='valid', language_pair=('de', 'en'))

    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)
        raise NotImplementedError

    train_loader = DataLoader(train_iter, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=num_workers)
    val_loader = DataLoader(val_iter, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=num_workers)
    return train_loader, val_loader