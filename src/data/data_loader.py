import torch
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from typing import Tuple, Dict, Callable

# Special token indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

# You will build these in your preprocessing
vocab_transform: Dict[str, any] = {}   # e.g. {'de': vocab_de, 'en': vocab_en}
text_transform: Dict[str, Callable[[str], torch.Tensor]] = {}  # tokenizers

def index_to_german(tokens: torch.Tensor) -> str:
    """Convert German token indices back to a string."""
    # TODO: use vocab_transform['de'].lookup_tokens(...)
    raise NotImplementedError

def index_to_english(tokens: torch.Tensor) -> str:
    """Convert English token indices back to a string."""
    # TODO: use vocab_transform['en'].lookup_tokens(...)
    raise NotImplementedError

def get_translation_dataloaders(
    batch_size: int = 128,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Download Multi30k, build vocab & text transforms,
    and return train/val DataLoaders that yield (src, tgt) tensors.
    """
    # TODO: download dataset, build vocab_transform and text_transform,
    #       collate func to pad & numericalize.
    train_iter = Multi30k(split='train', language_pair=('de', 'en'))
    val_iter   = Multi30k(split='valid', language_pair=('de', 'en'))

    # Placeholder collate_fn: you must implement actual tokenization + padding
    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)
        # tokenize, numericalize, pad...
        raise NotImplementedError

    train_loader = DataLoader(train_iter, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=num_workers)
    val_loader = DataLoader(val_iter, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=num_workers)
    return train_loader, val_loader