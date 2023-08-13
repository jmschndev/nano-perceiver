import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import Config


class Attention(nn.Module):
    """Multi-headed {cross, self}-attention."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_size = cfg.n_embed // cfg.n_heads

        self.query = nn.Linear(cfg.n_embed, cfg.n_embed, bias=False)
        self.key = nn.Linear(cfg.n_embed, cfg.n_embed, bias=False)
        self.value = nn.Linear(cfg.n_embed, cfg.n_embed, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, inputs_q, inputs_kv):
        q = self.query(inputs_q)
        k = self.key(inputs_kv)
        v = self.value(inputs_kv)

        batch, q_time, _ = q.shape
        _, kv_time, _ = k.shape

        # One thing that was originally confusing in the original Perceiver was how
        # one can cross attend between a Q with size (N, D) and K with size (M, C).
        # The Perceiver IO implementation handles this via a "conv_1d" (really a linear
        # layer, but I suppose the semantics of a convolution are nice when thinking
        # about resizing channel dimensions?). Aka, it picks either D or C to be the
        # latent shape, and then projects accordingly. However, in PerceiverAR, they
        # default to just using C channels for both queries and keys/values, hence
        # omitting any projections here.
        q = torch.reshape(q, (batch, q_time, self.n_heads, self.head_size))
        k = torch.reshape(k, (batch, kv_time, self.n_heads, self.head_size))
        v = torch.reshape(v, (batch, kv_time, self.n_heads, self.head_size))

        # Compute Q @ K.T for each head.
        attention = torch.einsum("bthd,bThd->bhtT", q, k)
        scale = 1.0 / math.sqrt(self.head_size)
        attention *= scale

        # Causal masking.
        mask = torch.tril(
            torch.ones(q_time, kv_time, device=attention.device),
            diagonal=kv_time - q_time,
        )
        attention = attention.masked_fill(~mask.bool(), float("-inf"))

        # Normalize and apply dropout.
        normalized = F.softmax(attention, dim=-1)
        normalized = self.dropout(normalized)

        # Compute attention scores @ values.
        summed = torch.einsum("bhtT,bThd->bthd", normalized, v)
        return torch.reshape(summed, (batch, q_time, self.n_heads * self.head_size))


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embed, 4 * cfg.n_embed),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embed, cfg.n_embed),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.query_size = cfg.query_size

        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embed)
        self.ln2 = nn.LayerNorm(cfg.n_embed)

    def forward(self, x: torch.Tensor):
        # Attend to normalized inputs, and add skip connection.
        inputs_q, inputs_kv = x[:, -self.query_size :, :], x
        normed_q, normed_kv = self.ln1(inputs_q), self.ln1(inputs_kv)
        x = inputs_q + self.attn(inputs_q=normed_q, inputs_kv=normed_kv)

        # MLP over normalized inputs.
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.block_size = cfg.block_size
        self.device = cfg.device
        self.token_embeddings = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.position_embeddings = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_blocks)])
        self.ln_f = nn.LayerNorm(cfg.n_embed)  # final layer norm
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size)

    def forward(self, idx, targets=None):
        B, N = idx.shape

        tok_emb = self.token_embeddings(idx)
        pos_emb = self.position_embeddings(torch.arange(N, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            _, N, V = logits.shape
            logits = logits.view(B * N, V)
            targets = targets.view(B * N)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
