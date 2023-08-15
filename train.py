import torch
import data
from typing import Optional

from config import Config
from torch.profiler import profile, record_function, ProfilerActivity
from model import Transformer


def _get_config(query_size: Optional[int] = None, **kwargs):
    vocab_size, _, _, _, _ = data.get_shakespeare()
    config = Config(vocab_size=vocab_size, query_size=query_size, **kwargs)
    return config


def _get_batch(split, train_data, val_data, query_size, block_size, batch_size, device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack(
        [data[i + block_size + 1 - query_size : i + block_size + 1] for i in ix]
    )
    x, y = x.to(device), y.to(device)
    return x, y


@torch.inference_mode()
def estimate_loss(eval_iters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    torch.manual_seed(1337)

    cfg = _get_config(
        max_iters=50000,
        eval_interval=100,
        eval_gen_interval=1000,
        batch_size=16,
        query_size=64,
        block_size=256,
        n_embed=64,
        n_heads=8,
        n_blocks=4,
        dropout=0.1,
    )
    model = Transformer(cfg)
    m = model.to(cfg.device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    vocab_size, train_data, val_data, encode, decode = data.get_shakespeare()

    get_batch = lambda split: _get_batch(
        split,
        train_data,
        val_data,
        cfg.query_size,
        cfg.block_size,
        cfg.batch_size,
        cfg.device,
    )

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    for it in range(cfg.max_iters):
        # every once in a while evaluate the loss on train and val sets
        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            losses = estimate_loss(cfg.eval_iters)
            print(f"loss@{it}: train {losses['train']:.4f},  loss {losses['val']:.4f}")
        if it % cfg.eval_gen_interval == 0 or it == cfg.max_iters - 1:
            # generate from the model
            context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
            print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))
            print("=" * 50)

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    print("=" * 50)
    context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
    print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

    # xb, yb = get_batch("train")
    # with profile(
    #     activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
    # ) as prof:
    #     model(xb)

    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
