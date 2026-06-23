# Transformers From Scratch — A Hands-On Curriculum for a Coding Harness

**What this is:** a module-by-module curriculum for evolving a tiny GPT (Karpathy's nanoGPT) into a modern LLM, implementing one architectural idea at a time. Each module ties a historical milestone to a concrete, verifiable code change in a live codebase.

**Who this is for:** a coding agent (Claude Code / Codex) pair-programming with a human learner. The agent drives the mechanics; the human is here to *understand*, not to receive a finished modern transformer. Read Section 1 before touching any code — it defines how you (the agent) should behave. The whole point is lost if you one-shot the end state.

---

## 1. Operating instructions for the agent (read first)

You are a pair-programming tutor, not an autocompleter. Follow these rules for the whole curriculum:

1. **Teach, then let the human build the core.** For each module: (a) explain the concept in a few sentences and why it mattered historically, (b) point to exactly where in the code it goes, (c) have the **human write the conceptually load-bearing lines themselves** — the attention math, the router, the RoPE rotation, the latent projection. You scaffold boilerplate, imports, configs, and tests around it. If the human asks you to just write it, do so, but default to letting them type the part that teaches. (This human learns by typing the important bits directly — respect that.)

2. **One concept per step. Minimal, isolated diffs.** Never refactor broadly or bundle changes. A module is one logical change set. Don't "improve" unrelated code while you're in there.

3. **Keep a known-good baseline and commit per module.** Use git. Tag the nanoGPT baseline (`git tag baseline`). Commit after each module with a clear message (suggested messages are given per module). Everything must be revertible so a broken experiment costs nothing.

4. **Gate progress on understanding.** After each module, confirm the human follows what changed and *why* before moving on. Offer to go deeper or proceed. Do **not** dump the whole curriculum at once.

5. **Stay tiny and fast.** Use char-level Shakespeare (nanoGPT's default) or TinyStories, a small model (e.g. ~4–8 layers, d_model 128–384), and short context. The goal is fast iteration and clear signal, **not** SOTA. A single consumer GPU is overkill; iterations should take seconds to minutes. Never let the model grow to where a training run is slow enough to break the feedback loop.

6. **Make each feature a toggle.** Add new techniques behind a config flag (e.g. `norm="rmsnorm"`, `pos="rope"`, `attn="gqa"`, `ffn="moe"`) so the human can A/B old vs new on the same harness. Ablation is the best teacher.

7. **Maintain a running scoreboard.** After every module, update a metrics table (see Section 4) so the *effect* of each change is visible: parameter count, active params/token, KV-cache bytes/token, tokens/sec, peak memory, and validation loss. When a change is supposed to save memory or add capacity, **show the number**.

8. **Verify before advancing.** After each change run: a shape/forward smoke test, a short training run (loss should still descend), and a generation sample (output should still be coherent-ish for the model size). Diff metrics against the baseline. Never proceed on an unverified change.

9. **Build the minimal faithful version.** Each technique should be the smallest correct teaching implementation, not a production one. Name the tensor-shape, compute, and memory consequence explicitly, and cite the paper. Where a technique is genuinely too heavy to implement from scratch (a real FlashAttention CUDA kernel; full RL infrastructure), say so plainly and use the practical substitute noted in the module.

10. **Connect each step to the lineage.** When you introduce a change, say which real model first shipped it and what problem it solved, so the human is building a mental map of the field, not just editing code.

---

## 2. The shape of the journey

Every change below is a push along one of five axes. Keep these in view; they organize the whole field:

1. **Architecture / attention efficiency** (how attention is computed and how cheaply)
2. **Scale** (parameters and compute)
3. **Data** (how much, how clean)
4. **Post-training / alignment** (turning a predictor into an assistant)
5. **Inference-time compute / reasoning** (thinking before answering)

This curriculum is mostly axis 1, with a taste of 4–5 at the end (the parts you can feel on a small model). Module map:

| Module | You implement | Milestone | Axis |
|---|---|---|---|
| 0 | nanoGPT baseline running + metrics | — | setup |
| 1 | Read & annotate the vanilla block | *Attention Is All You Need* (2017) | 1 |
| 2 | RMSNorm + SwiGLU + pre-norm | Modern transformer toolkit | 1 |
| 3 | RoPE (rotary positions) | Su et al. RoPE; LLaMA | 1 |
| 4 | KV cache for generation | Inference fundamentals | 1 |
| 5 | GQA (grouped-query attention) | Llama 2 / 3 | 1 |
| 6 | Fused attention (SDPA / FlashAttention) | Dao et al. FlashAttention | 1 |
| 7 | Mixture-of-Experts FFN | Mixtral; DeepSeekMoE | 1,2 |
| 8 | MLA (latent KV compression) | DeepSeek-V2 | 1 |
| 9 | Multi-Token Prediction (optional) | DeepSeek-V3 | 1 |
| 10 | RLVR / GRPO in miniature | o1 / DeepSeek-R1 | 5 |
| 11 | Content-based sparse attention (optional) | DeepSeek DSA / NSA | 1 |

---

## 3. The modules

Each module has the same shape: **Goal · Concept · Change · Verify · Record · Commit.**

### Module 0 — Baseline

**Goal.** A tiny GPT that trains and samples, with baseline metrics captured.

**Change.** Clone/build nanoGPT (or start from the human's repo). Get char-level Shakespeare training. Orient the human in `model.py` (`GPT`, `Block`, `CausalSelfAttention`, `MLP`) and `train.py`. Pick a small config that trains to a reasonable val loss in a couple of minutes.

**Verify.** Training loss descends; `sample.py` produces vaguely English-looking text.

**Record.** Establish the scoreboard's first row: params, tokens/sec, peak memory, val loss. This is the reference everything diffs against.

**Commit.** `git tag baseline && git commit -m "baseline: nanoGPT char-level Shakespeare"`

---

### Module 1 — Understand the vanilla block

**Goal.** The human can explain every line of the attention path before changing any of it. Don't skip this; everything later is a modification *of* it.

**Concept.** The 2017 Transformer: token + (learned/absolute) positional embeddings, multi-head self-attention (project to Q/K/V, scaled dot-product, causal mask, softmax, weighted sum, output projection), an MLP, residual connections, and LayerNorm. Recurrence is gone; this is what made parallel training — and therefore scale — possible.

**Change (exercise, not a feature).** Have the human re-derive the attention forward pass: annotate shapes at each step `(B, T, C) → (B, nh, T, hd)`, write out the causal mask, and explain why we scale by `1/sqrt(head_dim)`. Optionally have them re-implement `CausalSelfAttention.forward` from a blank function and diff against the original.

**Verify.** Their reimplementation matches the original's output on a fixed input (allclose).

**Commit.** `docs: annotate baseline attention` (comments only).

---

### Module 2 — RMSNorm, SwiGLU, pre-norm

**Goal.** Replace the "classic" norm and activation with what every modern model uses.

**Concept.** Three standards that quietly replaced the originals: **RMSNorm** (drop the mean-centering and bias of LayerNorm — cheaper, just as stable), **SwiGLU** (a gated MLP: `(W1 x) * silu(W2 x)` then project down — better quality per param), and **pre-normalization** (norm *before* the sublayer, which trains more stably at depth). First popularized together by LLaMA.

**Change.** Add `norm` and `ffn` config flags. Implement RMSNorm (human writes the `x / rms(x) * weight` line). Convert the MLP to SwiGLU (note it has 3 weight matrices, not 2 — keep param count comparable by adjusting the hidden dim, ~⅔·4·d). Move norms to pre-norm position if not already.

**Verify.** Loss curve is at least as good as baseline; sample still coherent.

**Record.** Params (should be ~flat if you sized SwiGLU right), tokens/sec, val loss.

**Commit.** `feat: rmsnorm + swiglu + pre-norm (LLaMA-style block)`

---

### Module 3 — RoPE (rotary position embeddings)

**Goal.** Replace absolute/learned positions with relative ones applied inside attention.

**Concept.** Instead of *adding* a position vector to the embedding, **RoPE** *rotates* the Q and K vectors by an angle that depends on position, so attention scores depend on relative offset `(i − j)`. This generalizes better to lengths unseen in training and is now near-universal. (Su et al.; shipped in LLaMA and almost everything since.)

**Change.** Add `pos="rope"`. Precompute `cos`/`sin` frequency tables. Have the human write `apply_rope(q, k)` — the half-rotation trick (split last dim, rotate pairs). Apply to Q and K *before* the dot product. Remove the learned positional embedding when RoPE is on.

**Verify.** Loss matches/beats baseline. Then the payoff test: train at context length 128, evaluate generation at 256, and compare RoPE vs the old absolute embeddings — RoPE should degrade more gracefully. Make the human *see* the extrapolation difference.

**Record.** Val loss; note the length-extrapolation observation.

**Commit.** `feat: rotary position embeddings (RoPE)`

---

### Module 4 — KV cache

**Goal.** Make autoregressive generation fast, and make the KV-cache bottleneck tangible — it's the thing every later attention trick attacks.

**Concept.** During generation, every new token re-attends to all previous keys/values. Without caching you recompute all of them every step — O(T²) wasted work. A **KV cache** stores past K/V so each step is O(T). The cache's *size* (grows with context × layers × heads) becomes the dominant inference cost, which is the whole motivation for GQA and MLA next.

**Change.** Add an inference path that maintains per-layer K/V buffers and appends one token's K/V per step. Keep the training path unchanged (full parallel attention). Have the human reason through the buffer shapes and the append.

**Verify.** Cached vs uncached generation produce identical tokens (greedy). Time both — cached should be dramatically faster as sequence length grows. Plot or print the speedup.

**Record.** Add a **KV-cache bytes/token** column to the scoreboard (compute it: `2 · n_layers · n_kv_heads · head_dim · dtype_bytes`). This number is the protagonist of Modules 5 and 8.

**Commit.** `feat: kv cache for generation`

---

### Module 5 — GQA (grouped-query attention)

**Goal.** Shrink the KV cache by sharing K/V across groups of query heads.

**Concept.** MHA gives every query head its own K/V. **MQA** (the extreme) shares one K/V across all heads — tiny cache, some quality loss. **GQA** is the middle: `n_kv_heads < n_heads`, each KV head serving a group of query heads. Standard in Llama 2 (big sizes) and Llama 3.

**Change.** Add `n_kv_heads` config. Project K/V to fewer heads and repeat-interleave them to match the query heads in the dot product. Human writes the head-grouping/repeat logic. Set `n_kv_heads = n_heads` to recover MHA, `=1` for MQA.

**Verify.** Sweep `n_kv_heads ∈ {n_heads, n_heads/2, 1}`. Watch KV-cache bytes/token drop and val loss change. The human should be able to state the tradeoff in their own words.

**Record.** KV-cache bytes/token at each setting; val loss.

**Commit.** `feat: grouped-query attention (GQA)`

---

### Module 6 — Fused attention (FlashAttention via SDPA)

**Goal.** Replace the hand-rolled attention with an IO-aware fused kernel; understand what it buys.

**Concept.** **FlashAttention** computes exact attention without materializing the full `T×T` score matrix in HBM — it tiles the computation and keeps it in fast SRAM, cutting memory traffic and enabling longer context. Same math, better systems. (Dao et al.)

**Change.** Swap the manual `softmax(QKᵀ/√d)·V` for `torch.nn.functional.scaled_dot_product_attention` (which dispatches to a FlashAttention kernel where available), passing `is_causal=True`. **Note explicitly** to the human: writing a real Flash CUDA kernel is out of scope; the lesson is (a) what the kernel does and why memory traffic, not FLOPs, was the bottleneck, and (b) how to use the primitive. Keep the manual path behind a flag for comparison.

**Verify.** Outputs match the manual path (allclose). Compare peak memory and tokens/sec at a longer context length — fused should use less memory and allow longer sequences.

**Record.** Peak memory and max trainable context length, manual vs fused.

**Commit.** `perf: fused attention via scaled_dot_product_attention`

---

### Module 7 — Mixture-of-Experts FFN

**Goal.** Decouple capacity from per-token compute — the single biggest architectural shift in modern open models.

**Concept.** Replace the single MLP with **N expert MLPs + a router** that sends each token to its top-k experts. Total parameters grow (capacity), but only k experts run per token (per-token compute stays low). This is why frontier open models have huge total params but small active params. Mixtral brought it to open weights; DeepSeekMoE refined it with fine-grained + shared experts. Add a **load-balancing loss** so the router doesn't collapse onto a few experts.

**Change.** Add `ffn="moe"` with `n_experts`, `top_k`. Implement: a linear router producing per-expert logits, top-k gating with softmax weights, dispatch tokens to experts, combine weighted outputs. Add the auxiliary load-balance loss term. Human writes the router + gating; you scaffold the dispatch/combine. Start simple (a loop over experts is fine for teaching; note it's not how you'd do it at scale).

**Verify.** Forward/backward works; loss descends. Log expert utilization (it should spread out once the balance loss is on — show before/after). Compare a dense model vs an MoE with the *same active params but more total params*.

**Record.** Add an **active params/token** column distinct from total params. Show total ↑, active ≈ flat.

**Commit.** `feat: sparse mixture-of-experts FFN with load balancing`

---

### Module 8 — MLA (multi-head latent attention)

**Goal.** Compress the KV cache further than GQA via a low-rank latent — the DeepSeek-V2 efficiency move, and the direct continuation of the MHA → GQA arc.

**Concept.** Instead of caching full per-head K/V, **MLA** down-projects them into a small shared **latent** vector that's cached, and up-projects per head at attention time. The cache stores the latent (much smaller) rather than full K/V. This is the headline efficiency idea behind DeepSeek-V2/V3's cheap long-context inference. (Pair with RoPE carefully — DeepSeek uses a decoupled-RoPE scheme; for a teaching impl you can start with a simplified version and note the subtlety.)

**Change.** Add `attn="mla"`. Implement the down-projection to a latent dim `d_c ≪ d_model`, cache the latent, and up-project to per-head K/V. Human writes the down/up projection pair and reasons about what now lives in the cache.

**Verify.** Loss comparable to GQA/MHA at matched params. KV-cache bytes/token should drop below GQA. Generation still coherent.

**Record.** KV-cache bytes/token: MHA vs GQA vs MLA, side by side. This row is the whole DeepSeek efficiency story in three numbers.

**Commit.** `feat: multi-head latent attention (MLA, DeepSeek-V2 style)`

---

### Module 9 — Multi-Token Prediction (optional, advanced)

**Goal.** Denser training signal via predicting more than one next token.

**Concept.** **MTP** adds auxiliary heads that predict the next 2 (or n) tokens during training, giving more gradient signal per position. DeepSeek-V3 used it (and the extra head can later assist speculative decoding). It's a *training-objective* change, not an architecture-of-inference change.

**Change.** Add a second prediction head and an auxiliary loss for the token at offset +2, weighted into the total loss. Human writes the loss combination. Keep it behind a flag.

**Verify.** Primary next-token loss should be unharmed (often slightly better); training is a bit slower. Confirm the auxiliary loss is actually decreasing.

**Record.** Val loss with/without MTP; tokens/sec cost.

**Commit.** `feat: multi-token prediction auxiliary objective`

---

### Module 10 — RLVR / GRPO in miniature

**Goal.** Feel how the reasoning paradigm works — reward correct *final answers*, watch behavior shift — without standing up real RL infrastructure.

**Concept.** Models like o1 and DeepSeek-R1 are trained to think before answering using **RL with verifiable rewards (RLVR)**: on tasks where correctness is automatically checkable (math, code), reward the model for right answers (and a clean format), and long chain-of-thought *emerges*. **GRPO** is the lightweight policy-gradient variant DeepSeek popularized (it scores a group of sampled answers relative to each other, skipping a separate value network). This opened a *second* scaling axis — spend compute at inference time.

**Change (scope down — be honest about limits).** Pick a tiny verifiable task: e.g. multi-digit addition, or "count the letters," where a checker can score outputs exactly. Then either:
- **(a) Minimal GRPO loop:** sample G completions per prompt from the current model, score each with the verifier, compute group-relative advantages (reward minus group mean, over std), and do a policy-gradient update with a KL penalty to a frozen reference. Have the human write the advantage computation and the loss; you scaffold sampling and the verifier.
- **(b) If full RL is too heavy for the setup:** implement just the verifier + sampling + reward bookkeeping and *walk through* the GRPO update math against that data, leaving the optimizer step as a guided read.

Tell the human up front this is the most infrastructure-heavy module and the toy task is the point.

**Verify.** Reward/accuracy on the held-out checker trends up across iterations; sampled outputs visibly change (e.g. start showing intermediate steps). Even a small, noisy upward trend demonstrates the mechanism.

**Record.** Verifier accuracy over RL steps (a small curve is enough).

**Commit.** `feat: minimal RLVR/GRPO loop on a verifiable toy task`

---

### Module 11 — Content-based sparse attention (optional, frontier)

**Goal.** Approximate the current attention frontier — near-linear long-context cost via selecting which past tokens to attend to.

**Concept.** **DeepSeek Sparse Attention (DSA)** (built on their Native Sparse Attention research) uses a fast **"lightning indexer"** to score which past tokens matter for the current query, then attends to only the top-k of them — pushing attention from quadratic toward near-linear for long contexts. It's the live frontier of attention efficiency (DeepSeek-V3.2), layered on top of MLA.

**Change.** Implement a simple scorer that, per query, ranks past positions (a cheap learned or heuristic relevance score) and restricts attention to the top-k. Human writes the selection logic. Keep it approximate — the goal is to feel the mechanism, not match the paper.

**Verify.** At long context, compare cost/memory of full vs sparse attention; check that quality holds on a task that needs only sparse long-range lookups (e.g. retrieving a key planted earlier in the sequence).

**Record.** Cost/memory vs context length, full vs sparse.

**Commit.** `feat: content-based top-k sparse attention (DSA-style)`

---

## 4. The running scoreboard

Maintain this table in the repo (e.g. `SCOREBOARD.md`), one row per module, so every change's effect is legible:

| Module | Total params | Active params/tok | KV-cache bytes/tok | Tokens/sec | Peak mem | Val loss | Notes |
|---|---|---|---|---|---|---|---|

The columns are chosen so the lessons show up as numbers: MoE moves *total* up while *active* stays flat; GQA and MLA drive *KV-cache bytes/token* down; FlashAttention moves *peak mem* down and unlocks longer context; RMSNorm/SwiGLU/RoPE mostly move *val loss*. If a change is supposed to help and the number doesn't move, stop and find out why before continuing.

---

## 5. Beyond the curriculum

**How the toys map to the real frontier.** By the end you'll have hand-built the load-bearing pieces of a current model: a LLaMA-style block (RMSNorm/SwiGLU/RoPE), GQA→MLA→sparse attention (the KV-cache compression lineage), sparse MoE (capacity-vs-compute decoupling), and a miniature reasoning loop (RLVR/GRPO). The real frontier is these same ideas at scale plus engineering you won't reproduce on a laptop: FP8 training, auxiliary-loss-free load balancing, massive data pipelines, and reasoning-as-an-adjustable-dial. DeepSeek-V3.2's DSA is Module 11 at scale on top of Module 8; every giant open MoE is Module 7 at scale.

**Reference implementations to consult (not copy wholesale).** Karpathy's nanoGPT (the baseline) and nanochat (a fuller small-scale ChatGPT-style stack); the "modded-nanogpt" speedrun forks already implement RoPE and several modern tricks and are a good cross-check for Modules 2–6.

**Papers, one per module, for when the human wants the source:** Attention Is All You Need (1); the RoPE paper (3); FlashAttention (6); the Mixtral / DeepSeekMoE reports (7); DeepSeek-V2 for MLA (8); DeepSeek-V3 for MTP and the training stack (9); the DeepSeek-R1 paper for GRPO/RLVR (10); the DeepSeek-V3.2 / Native Sparse Attention reports for DSA (11). Sebastian Raschka's "Understanding Reasoning LLMs" and his DeepSeek architecture write-ups are strong companions throughout.

**Definition of done.** The human can take any new model's tech report, point to which of these axes each change touches, and — for the architectural ones — sketch how they'd implement it as a diff against this codebase. That's the goal: not a finished model, but the ability to read the field as a series of modifications they could make themselves.
