# Running scoreboard

One row per curriculum module (see `transformers-from-scratch-harness-curriculum.md` §4).
Reference config for all rows unless noted: char-level Shakespeare, 4 layers, 4 heads,
d_model=128, block_size=64, batch_size=12, 2000 iters, dropout=0.0, CPU (no compile).

Reproduce with:

```
python3 data/shakespeare_char/prepare.py
python3 train.py config/train_shakespeare_char.py --device=cpu --compile=False \
  --eval_iters=20 --log_interval=100 --block_size=64 --batch_size=12 \
  --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
python3 sample.py --out_dir=out-shakespeare-char --device=cpu
```

| Module | Total params | Active params/tok | KV-cache bytes/tok | Tokens/sec | Peak mem | Val loss | Notes |
|---|---|---|---|---|---|---|---|
| 0 — baseline | 0.80M | 0.80M (dense) | n/a — no cache, full recompute per token | ~25k (768 tok/iter @ ~31 ms, CPU) | ~823 MB RSS (CPU; mostly PyTorch runtime) | **1.886** | train loss 1.765 @ iter 2000; samples are Shakespeare-shaped pseudo-English |
