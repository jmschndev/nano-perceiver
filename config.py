import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    vocab_size: int  # Vocabulary size of tokenizer.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    max_iters: int = 1000
    eval_interval: int = 100
    eval_gen_interval: int = 1000
    eval_iters: int = 200

    learning_rate: float = 1e-3
    batch_size: int = 16
    block_size: int = 32  # Maximum context length.

    # The parameter that determines if this is a regular decoder-only transformer
    # or a Perceiver AR.
    query_size: Optional[int] = None

    n_embed: int = 64  # Embedding dimensionality.
    n_heads: int = 4  # Number of embedding heads.
    n_blocks: int = 4  # Number of self-attention + MLP blocks.
    dropout: float = 0.0

    def __post_init__(self):
        assert self.n_embed % self.n_heads == 0
        if self.query_size is None:
            self.query_size = self.block_size
        else:
            assert self.query_size <= self.block_size
