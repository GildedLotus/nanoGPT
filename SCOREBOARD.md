# Scoreboard

One row per curriculum module (see `transformers-from-scratch-harness-curriculum.md` §4).
Every architectural change gets diffed against the row above it. If a change is supposed
to move a number and it doesn't, stop and find out why before continuing.

| Module | Config | Total params | Active params/tok | KV-cache bytes/tok | ms/iter (CPU) | Val loss | Notes |
|---|---|---|---|---|---|---|---|
| 0 — baseline | 4L / 4H / d128, block 64, char-Shakespeare | 0.80M | 0.80M | — (no cache until M4) | ~38 | **1.8857** | 2000 iters, README CPU recipe |

**Baseline reproduce command** (CPU, ~3–5 min):

```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py \
  --device=cpu --compile=False --eval_iters=20 --log_interval=100 \
  --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 \
  --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
python sample.py --out_dir=out-shakespeare-char --device=cpu --num_samples=1 --max_new_tokens=300
```

Throughput at this config: 12 × 64 = 768 tokens/iter ≈ 20k tokens/sec on 4 CPU cores.
