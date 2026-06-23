# LLM Development Study Guide — Harness-First Roadmap

_Last updated: 2026-06-22._

This guide is written to be passed to a code agent such as Claude Code or OpenAI Codex. The goal is not to build a frontier model. The goal is to start from a small nanoGPT-style codebase and progressively implement faithful miniature versions of the ideas that shaped modern LLMs: Transformers, GPT-style decoder-only pretraining, LLaMA-style blocks, post-training, long context, MoE, DeepSeek-style efficiency, reasoning RL, tool use, and agentic coding workflows.

## 0. How to use this document with a code harness

Use the codebase as the learning instrument. Every paper or model family should become a small diff, a test, a metric, and a reflection note.

### Agent operating contract

The agent should follow these rules in every phase:

1. Read this guide, then inspect the repository before editing.
2. Make one conceptually small implementation change at a time.
3. Add or update tests before running longer experiments.
4. Prefer tiny, deterministic experiments over expensive runs.
5. Log the before/after metrics for loss, perplexity or bits-per-byte, tokens/sec, peak memory, and qualitative samples.
6. Never “just scale up” to hide an architectural misunderstanding.
7. Keep a `docs/phase_log.md` entry for each phase with: what changed, why, how to test, metrics, and remaining questions.
8. When blocked, write the smallest reproduction script and stop there.

### Suggested repository scaffold

```text
llm-lab/
  AGENTS.md                    # Codex-style project instructions
  CLAUDE.md                    # Claude Code-style project instructions, if used
  README.md
  docs/
    llm_development_study_guide_harness.md
    phase_log.md
    source_notes.md
  configs/
    tiny.yaml
    gpt2_like.yaml
    llama_like.yaml
    moe_tiny.yaml
    reasoning_toy.yaml
  src/llm_lab/
    __init__.py
    model.py                   # Transformer, GPT, Llama-like model variants
    attention.py               # MHA, MQA/GQA, RoPE, KV cache, optional MLA
    mlp.py                     # GELU MLP, SwiGLU, MoE FFN
    moe.py                     # routers, experts, balancing metrics
    tokenizer.py
    data.py
    train.py
    generate.py
    eval.py
    posttrain.py               # SFT, DPO, toy GRPO/RLVR
    tools.py                   # toy tool calling / agent loop
  scripts/
    train_tiny.py
    eval_tiny.py
    benchmark_attention.py
    run_phase.py
  tests/
    test_attention.py
    test_rope.py
    test_kv_cache.py
    test_moe.py
    test_training_smoke.py
    test_posttraining.py
  experiments/
    phase_01_baseline/
    phase_02_gpt2/
    phase_03_llama_block/
    ...
```

### Minimum test harness

The first code milestone is not “train a model.” It is “prove the harness can detect breakage.” Add these tests early:

```text
test_shapes:        forward pass returns [batch, time, vocab]
test_causal_mask:   token t cannot attend to future positions
test_loss_decreases: tiny model overfits 1 batch in <200 optimizer steps
test_generate:      generate() returns valid token ids and respects max_new_tokens
test_checkpoint:    save/load produces identical logits for fixed input
test_benchmark:     benchmark script records tokens/sec and peak memory
```

### Copy-paste starter prompt for Claude Code or Codex

```text
You are working in this repository as a study harness for modern LLM development.

Read docs/llm_development_study_guide_harness.md and the project AGENTS.md/CLAUDE.md first. Implement only the next incomplete phase. Do not skip ahead.

For the current phase:
1. summarize the target concept in 5-10 sentences,
2. identify files to change,
3. add or update tests,
4. implement the smallest faithful version,
5. run the relevant tests and one tiny smoke experiment,
6. update docs/phase_log.md with metrics and a short reflection.

Avoid large refactors. Avoid long training runs. If a phase requires expensive compute, implement a toy version and document what would change at scale.
```

### Suggested AGENTS.md

```markdown
# AGENTS.md

## Project purpose
This repository is a hands-on LLM development study harness. The objective is to learn concepts by implementing tiny faithful versions, not to produce a competitive model.

## Commands
- Install: `uv sync` or `pip install -e .[dev]`
- Tests: `pytest -q`
- Smoke train: `python scripts/train_tiny.py --config configs/tiny.yaml --steps 50`
- Eval: `python scripts/eval_tiny.py --checkpoint <path>`
- Attention benchmark: `python scripts/benchmark_attention.py --seq-lens 128,512,1024`

## Engineering rules
- Make small diffs.
- Add tests for every architecture change.
- Keep experiments tiny unless explicitly asked.
- Log metrics in `docs/phase_log.md`.
- Do not delete datasets, checkpoints, or experiment logs without explicit confirmation.
- Do not fetch unlicensed datasets.
- Prefer deterministic seeds for tests and smoke runs.

## Done definition
A phase is done only when tests pass, a tiny smoke run works, metrics are logged, and the reflection questions are answered.
```

### Suggested CLAUDE.md

```markdown
# CLAUDE.md

This repo is a learning harness for LLM architecture and training. Act like a careful ML engineer and tutor.

Before coding:
- Read `docs/llm_development_study_guide_harness.md`.
- State the current phase and the specific concept being implemented.
- Inspect existing tests and configs.

While coding:
- Prefer minimal, clear PyTorch over clever abstraction.
- Keep model variants switchable by config.
- Add assertions for tensor shapes and cache invariants.
- Avoid long-running jobs.

After coding:
- Run targeted tests.
- Run a tiny smoke train/eval.
- Update `docs/phase_log.md` with before/after metrics.
- Explain what changed as if teaching the next reader.
```

---

## 1. The learning threads

Track three parallel threads through the whole roadmap.

### Architecture thread

How the block changed:

```text
Transformer encoder/decoder
  -> decoder-only GPT
  -> pre-LN GPT-2/nanoGPT
  -> LLaMA block: RMSNorm + RoPE + SwiGLU + no-bias linear layers
  -> grouped-query / multi-query attention and KV cache
  -> long-context kernels and cache-aware inference
  -> MoE feed-forward layers
  -> MLA / latent KV compression
  -> multimodal adapters and native multimodal models
```

### Training/post-training thread

How the behavior changed:

```text
next-token prediction
  -> scale laws and data/compute tradeoffs
  -> instruction tuning
  -> RLHF / reward modeling
  -> DPO-style preference optimization
  -> verifiable-reward RL and GRPO/RLVR
  -> reasoning-mode models
  -> tool-use and agentic post-training
```

### Systems/evaluation thread

How the model became practical:

```text
tiny local training
  -> tokenizer/data pipeline
  -> mixed precision and distributed training concepts
  -> FlashAttention / SDPA
  -> KV cache, batching, paged attention concepts
  -> eval harnesses
  -> agent/coding benchmarks
  -> cost, latency, memory, and safety measurement
```

---

## 2. Phase roadmap

Each phase has five required artifacts:

```text
1. code diff
2. tests
3. tiny experiment
4. metrics table
5. reflection note
```

### Phase 0 — Baseline harness and tiny GPT

**Anchor:** nanoGPT / build-nanoGPT / nanochat.

**Concept:** Before studying frontier architectures, get a small decoder-only Transformer running end-to-end. nanoGPT is useful because it is intentionally simple, and nanochat is a more full-stack training/inference/chat harness.

**Implement:**

- Character-level or byte-level dataset loader.
- Minimal GPT block: token embedding, position embedding, causal self-attention, MLP, residuals, LayerNorm, language-model head.
- Training loop with AdamW, gradient clipping, checkpointing, and generation.
- `pytest` smoke tests.

**Tiny experiment:**

```bash
python scripts/train_tiny.py --config configs/tiny.yaml --steps 200 --overfit-one-batch true
python scripts/eval_tiny.py --checkpoint runs/tiny/latest.pt
```

**Metrics:**

| Metric | Baseline value | Notes |
|---|---:|---|
| train loss step 0 |  | |
| train loss final |  | should decrease on tiny overfit |
| tokens/sec |  | CPU/GPU noted |
| max memory |  | |
| sample quality |  | usually gibberish at this size |

**Reflection questions:**

- Where is causality enforced?
- What exactly is the training target at position `t`?
- What breaks if the mask is wrong?

---

### Phase 1 — “Attention Is All You Need”: vanilla Transformer mechanics

**Anchor:** Vaswani et al., 2017.

**Concept:** The Transformer replaced recurrence/convolution with attention and position-wise feed-forward layers. The original paper used encoder-decoder machine translation, but the core mechanics power decoder-only LLMs too.

**Implement:**

- Standalone scaled dot-product attention.
- Multi-head attention from Q/K/V projections.
- Sinusoidal absolute positional embeddings as an option.
- Encoder-style bidirectional attention as a teaching-only variant.
- Decoder-style causal attention as the main path.

**Tests:**

- Causal mask prevents future-token attention.
- Attention probabilities sum to 1 over valid positions.
- Multi-head and single-head shape invariants hold.

**Tiny experiment:** Compare absolute learned positions vs sinusoidal positions on a tiny dataset.

**Reflection questions:**

- Why does attention have quadratic cost in sequence length?
- What information is stored in Q, K, and V?
- Why is position information needed at all?

---

### Phase 2 — GPT-style decoder-only language modeling

**Anchors:** GPT-2/nanoGPT, GPT-3.

**Concept:** Decoder-only causal language modeling scales naturally: predict the next token over a large corpus. GPT-3 showed that scaling autoregressive LMs improves task-agnostic few-shot behavior.

**Implement:**

- Decoder-only GPT with weight tying.
- BPE/tiktoken-style tokenization path, if feasible.
- Pre-LN block option.
- Learning-rate schedule with warmup and cosine decay.
- Text generation controls: temperature, top-k, top-p.

**Tests:**

- Weight tying shares storage between embedding and LM head.
- Generation respects temperature and stops at the requested limit.
- Pre-LN and post-LN model variants both run.

**Tiny experiment:** Train two small configs and compare validation loss at equal tokens.

**Reflection questions:**

- What changed between the original Transformer and GPT-style training?
- Why is decoder-only convenient for web-scale pretraining?
- What is in-context learning, and what evidence can a tiny model show or not show?

---

### Phase 3 — Scaling laws and compute-optimal training

**Anchors:** Kaplan-style scaling laws, Chinchilla compute-optimal training.

**Concept:** Progress came not only from architecture but from scaling data, parameters, and compute. Chinchilla shifted attention toward training smaller models on more tokens for a fixed compute budget.

**Implement:**

- A simple FLOPs estimator for decoder-only Transformers.
- Experiment grid runner: `n_layer`, `n_embd`, tokens seen, loss.
- Plot or CSV output for loss vs estimated compute.
- Config validator that reports approximate parameter count and token budget before training.

**Tests:**

- Parameter count matches a manual calculation for tiny configs.
- FLOPs estimator increases with layers, width, sequence length, and tokens.

**Tiny experiment:** Train 3 tiny models for equal wall-clock or equal token budget; log loss curves.

**Reflection questions:**

- What does “compute optimal” mean?
- Why can a smaller model trained longer beat a larger undertrained model?
- Which metrics are misleading at toy scale?

---

### Phase 4 — LLaMA/Llama 2 style modern decoder block

**Anchors:** LLaMA, Llama 2.

**Concept:** LLaMA-style open models popularized an efficient modern decoder block: RMSNorm, RoPE, SwiGLU, no-bias linear layers, and later grouped-query attention. Llama 2 also made chat post-training and safety tuning central to open model practice.

**Implement:**

- RMSNorm.
- Rotary position embeddings.
- SwiGLU MLP.
- Bias-free projections as a config option.
- Optional grouped-query attention.
- LLaMA-like config preset.

**Tests:**

- RMSNorm preserves input shape and has stable gradients.
- RoPE rotates Q/K pairs without changing tensor shape.
- SwiGLU output shape equals hidden size.
- GQA correctly repeats or broadcasts K/V groups to query heads.

**Tiny experiment:** Compare baseline GPT block vs LLaMA-like block at roughly similar parameter count.

**Reflection questions:**

- Why does RoPE help with relative position behavior?
- Why does RMSNorm remove mean-centering?
- What does GQA save at inference time?

---

### Phase 5 — Instruction tuning, RLHF, and DPO

**Anchors:** InstructGPT, Llama 2 post-training, DPO.

**Concept:** Base models learn next-token prediction; assistants require post-training. The classic ladder is SFT -> reward model -> RLHF. DPO simplified preference optimization by avoiding an explicit reward model and online RL loop.

**Implement:**

- Instruction/chat data format.
- SFT loop that masks out prompt tokens and trains assistant tokens.
- Pairwise preference dataset interface.
- Tiny DPO training loop.
- Evaluation prompts for instruction following.

**Tests:**

- SFT loss only applies to intended target tokens.
- DPO loss decreases on a tiny synthetic preference set.
- Chat template round-trips messages to tokens and back where possible.

**Tiny experiment:** Create 20 synthetic instruction pairs and demonstrate the model prefers the desired format after DPO.

**Reflection questions:**

- What behavior is learned during pretraining vs SFT vs preference optimization?
- Why can preference data change style without teaching much new factual knowledge?
- What are the failure modes of toy preference training?

---

### Phase 6 — Efficient attention and long-context foundations

**Anchors:** FlashAttention, FlashAttention-2/3, MQA/GQA.

**Concept:** A lot of modern LLM progress is systems work. Attention is mathematically simple but memory-bandwidth-heavy. FlashAttention makes exact attention faster by reducing GPU memory traffic. KV caching, MQA, and GQA reduce inference cost.

**Implement:**

- Use PyTorch `scaled_dot_product_attention` where available.
- Add a manual attention path for teaching.
- Add KV cache for incremental generation.
- Benchmark naive full-prefix generation vs cached generation.
- Benchmark MHA vs GQA/MQA memory footprint.

**Tests:**

- Cached generation logits match full-prefix logits within tolerance.
- Cache grows by exactly one step per generated token.
- GQA cache bytes are lower than full MHA cache bytes for `n_kv_heads < n_heads`.

**Tiny experiment:** Generate 128 tokens with and without KV cache; record tokens/sec.

**Reflection questions:**

- Why is exact attention still expensive?
- What data lives in the KV cache?
- How do MQA/GQA change memory bandwidth during decoding?

---

### Phase 7 — Llama 3: data, multilinguality, tool use, and larger context

**Anchor:** Llama 3 Herd of Models.

**Concept:** Llama 3 is not just a block tweak. It represents the production-era stack: huge pretraining, multilinguality, code, reasoning, tool use, safety models, long context, and multimodal research extensions.

**Implement:**

- Chat template with system/developer/user/assistant roles.
- Tool-call serialization format.
- Simple tool-use training/eval set: calculator, grep, JSON extraction.
- Long-context retrieval toy eval: insert needles into synthetic documents.
- Safety/eval split in the harness: harmless refusal tests and instruction-following tests.

**Tests:**

- Tool calls parse as valid JSON or a structured object.
- Long-context synthetic retrieval checks exact string recovery.
- Prompt formatting does not leak answer tokens into the prompt mask.

**Tiny experiment:** Fine-tune a small model to emit a calculator tool call for arithmetic prompts.

**Reflection questions:**

- Why does a chat model need a template?
- Why are tool-use evals different from language modeling loss?
- What can long-context tests reveal that perplexity cannot?

---

### Phase 8 — Mixture of Experts: Mixtral and DeepSeekMoE foundations

**Anchors:** Sparsely-Gated MoE, Mixtral, DeepSeekMoE.

**Concept:** MoE increases parameter count without activating all parameters per token. Modern sparse LLMs commonly replace dense FFNs with routed expert FFNs. The hard parts are routing, load balancing, capacity, communication, and stable training.

**Implement:**

- Dense FFN baseline.
- Top-1 and top-2 router.
- Expert FFN module.
- Optional shared expert.
- Load-balancing statistics: tokens per expert, entropy, dropped tokens, expert utilization.
- Capacity factor or no-drop toy implementation.

**Tests:**

- Router probabilities sum to 1.
- Top-k expert selection returns valid expert ids.
- No token is silently lost.
- MoE output shape equals dense FFN output shape.
- Load stats detect collapsed routing in a synthetic case.

**Tiny experiment:** Train dense vs MoE FFN at similar active parameters and compare loss and tokens/sec.

**Reflection questions:**

- What is the difference between total parameters and active parameters?
- Why does routing collapse happen?
- Why is MoE harder in distributed training than a dense FFN?

---

### Phase 9 — DeepSeek-V2: MLA and economical inference

**Anchor:** DeepSeek-V2.

**Concept:** DeepSeek-V2 combined sparse MoE with Multi-head Latent Attention (MLA), where the KV cache is compressed into a latent representation. The headline lesson for the harness is cache-aware architecture: inference memory can dominate design.

**Implement:**

- A simplified MLA-inspired attention module.
- Separate config for latent KV dimension.
- Cache-byte estimator for MHA, GQA, and simplified MLA.
- Optional compatibility mode that converts between normal attention and latent attention for testing.

**Tests:**

- Simplified MLA forward pass matches expected shapes.
- Cache-byte estimator shows reduction when latent dim is smaller than full KV dim.
- Incremental decoding path works with latent cache.

**Tiny experiment:** Compare cache memory estimates across sequence lengths 1K, 8K, 32K, and 128K.

**Reflection questions:**

- Why did DeepSeek-V2 focus so heavily on KV cache reduction?
- How does MLA differ from GQA at a high level?
- What parts of full MLA are hard to reproduce faithfully in a tiny harness?

---

### Phase 10 — DeepSeek-V3: MoE at scale, auxiliary-loss-free balancing, and multi-token prediction

**Anchor:** DeepSeek-V3.

**Concept:** DeepSeek-V3 continued the DeepSeek-V2 MoE/MLA direction and added large-scale training refinements: auxiliary-loss-free load balancing, multi-token prediction, and FP8-oriented training systems. In the harness, implement the algorithmic ideas in miniature and treat FP8/distributed systems as study notes unless you have the hardware.

**Implement:**

- Multi-token prediction head: predict token `t+1` and token `t+2` with weighted losses.
- Router balancing metric without simply adding a large auxiliary loss.
- Config flag for balancing strategy: none, auxiliary loss, bias/update-based toy version.
- Training log fields for expert utilization and MTP losses.

**Tests:**

- MTP labels are shifted correctly.
- MTP loss is zero or ignored where labels are unavailable at sequence end.
- Router utilization metrics are logged.

**Tiny experiment:** Train with and without MTP for the same number of steps; compare validation loss and sample behavior.

**Reflection questions:**

- Why might predicting multiple future tokens provide denser training signal?
- What can go wrong with an auxiliary load-balancing loss?
- Which DeepSeek-V3 contributions are architecture vs training-system contributions?

---

### Phase 11 — DeepSeek-R1 and reasoning RL

**Anchors:** DeepSeekMath GRPO, DeepSeek-R1.

**Concept:** DeepSeek-R1 highlighted verifiable-reward RL for reasoning. R1-Zero demonstrated that reasoning behavior can emerge from RL on verifiable tasks without first using human-labeled reasoning traces; R1 then used a more staged recipe. The harness should implement a toy version of group-relative RL on tasks with automatic rewards.

**Implement:**

- A synthetic verifiable task generator: arithmetic, string transforms, simple code-output prediction.
- Generate `G` completions per prompt.
- Reward function: exact answer match, format reward, optional length penalty.
- GRPO-style group-normalized advantage.
- Small policy update loop, ideally with LoRA or tiny model only.
- Reasoning trace format experiments: hidden scratchpad not needed; use visible `<reasoning>` tags only for learning.

**Tests:**

- Reward function is deterministic.
- Group-normalized rewards have expected mean behavior.
- Policy update runs on a tiny batch without NaNs.
- Format reward cannot be gamed without answer reward in the toy eval.

**Tiny experiment:** Train on 100 arithmetic prompts; evaluate exact-match before and after.

**Reflection questions:**

- What makes a reward “verifiable”?
- Why does group-relative normalization avoid a separate value model?
- What are the risks of optimizing visible reasoning traces?

---

### Phase 12 — Qwen3-style thinking/non-thinking mode

**Anchor:** Qwen3.

**Concept:** Qwen3 popularized unified thinking and non-thinking modes: a model can spend more tokens on hard reasoning or answer quickly for simple tasks. In a harness, this becomes a controllable decoding and training-format problem.

**Implement:**

- Chat template flag: `mode: think` or `mode: fast`.
- Dataset examples with and without reasoning scaffolds.
- Decoding budget policy: max reasoning tokens before final answer.
- Eval split by task difficulty.

**Tests:**

- Mode flag changes template reliably.
- `fast` mode prevents long reasoning tags in outputs if trained that way.
- `think` mode allows longer budget.

**Tiny experiment:** Train or prompt-tune a tiny model on two styles and compare output length/accuracy on simple arithmetic.

**Reflection questions:**

- Is “thinking mode” an architecture feature, a post-training feature, or a product interface feature?
- How should latency be included in reasoning evals?
- When does extra reasoning hurt?

---

### Phase 13 — Agentic coding and environment interaction

**Anchors:** Claude Code, Codex, Kimi K2, SWE-bench.

**Concept:** Recent frontier development is increasingly about long-horizon tool use: editing repos, running tests, browsing files, patching bugs, and iterating. Kimi K2 is an open model report focused on agentic capabilities; Claude Code and Codex are productized coding-agent harnesses.

**Implement:**

- Toy software-engineering benchmark inside the repo: 10 broken functions with unit tests.
- Agent loop abstraction: propose patch -> run tests -> read failure -> patch again.
- Tool API: read file, write file, run command, grep.
- Transcript logger.
- Patch quality rubric: passes tests, minimal diff, no skipped tests, no hardcoded hidden answers.

**Tests:**

- Agent tools cannot write outside the workspace.
- Command runner has timeouts.
- Transcript logger records commands and outputs.
- Benchmark validator catches test deletion or cheating.

**Tiny experiment:** Use the code agent to solve one toy issue while the harness logs every action.

**Reflection questions:**

- What is the difference between a model benchmark and an agent benchmark?
- Why are tests part of the environment rather than just evaluation?
- What new safety issues arise when the model can run commands?

---

### Phase 14 — Multimodal and long-context frontier models

**Anchors:** Llama 4, Gemini 3 series, Claude Opus 4.x, GPT-5.x, Kimi K2.5.

**Concept:** The current frontier is no longer “text-only next-token model” in product form. Models are multimodal, long-context, tool-using, and increasingly agentic. The harness can study the interfaces even if it cannot train a native multimodal frontier model.

**Implement optional miniature versions:**

- Vision adapter stub: encode image metadata or small image embeddings and feed them as prefix tokens.
- Document/PDF ingestion test: convert files into chunks and run retrieval.
- Long-context summarization eval with synthetic distractors.
- Tool-use planning benchmark: multi-step calculator + file edit + unit test.

**Tests:**

- Multimodal input objects serialize deterministically.
- Long-context eval places answers at different positions.
- Tool plans are executable, not just prose.

**Tiny experiment:** Build a synthetic “visual token” classification prefix task or skip training and document interface design.

**Reflection questions:**

- What does “native multimodal” imply beyond bolting on a vision encoder?
- Why do long-context claims need retrieval and reasoning tests, not just max-token specs?
- How do product-agent capabilities change what should be evaluated?

---

### Phase 15 — Current-development watchlist

As of 2026-06-22, keep the study guide’s “latest” section mutable. Re-check official docs and model cards before using specific claims in work.

Track these buckets:

1. **Open-weight efficient frontier:** Llama 4 Scout/Maverick, DeepSeek V4 Preview, Qwen3, Kimi K2/K2.5, MiniMax, GLM/z.ai.
2. **Closed frontier agentic models:** OpenAI GPT-5.x / GPT-5.5, Claude Opus 4.x / Fable/Mythos family, Gemini 3.x / 3.5.
3. **Architecture:** MoE, GQA/MQA, MLA/sparse attention, long-context memory, multimodal adapters/native multimodal, speculative decoding, multi-token prediction.
4. **Post-training:** DPO variants, RLVR/GRPO, process/outcome rewards, synthetic data pipelines, tool-use RL, agentic environment RL.
5. **Systems:** FlashAttention variants, FP8/low precision, paged KV cache, batching/scheduling, inference optimizers, distributed filesystems and data pipelines.
6. **Evaluation:** SWE-bench variants, lm-eval, long-context retrieval, tool-use evals, contamination controls, benchmark robustness.

---

## 3. Phase task-card template for the harness

Use this YAML-like format for each phase. The agent should fill it out before implementation.

```yaml
phase_id: phase_04_llama_block
status: planned
paper_anchor:
  - LLaMA / Llama 2
  - RMSNorm
  - RoPE
  - SwiGLU
objective: >
  Replace the baseline GPT block with a switchable LLaMA-style block while keeping
  existing GPT behavior available.
allowed_files:
  - src/llm_lab/model.py
  - src/llm_lab/attention.py
  - src/llm_lab/mlp.py
  - tests/test_rope.py
  - tests/test_training_smoke.py
commands:
  - pytest -q tests/test_rope.py tests/test_training_smoke.py
  - python scripts/train_tiny.py --config configs/llama_like.yaml --steps 50
metrics:
  - train_loss_start
  - train_loss_final
  - tokens_per_sec
  - peak_memory_mb
acceptance_criteria:
  - tests pass
  - tiny train runs without NaNs
  - phase_log includes before/after metrics
  - implementation can switch between baseline and llama-like block by config
reflection_questions:
  - Why does RoPE modify Q/K rather than V?
  - How does SwiGLU change parameter count?
  - What does RMSNorm omit compared with LayerNorm?
```

---

## 4. Implementation notes by concept

### Attention variants

```text
MHA: one K/V head per Q head. Strong baseline, expensive KV cache.
MQA: many Q heads share one K/V head. Faster decoding, possible quality tradeoff.
GQA: groups of Q heads share K/V heads. Middle ground used in many modern LLMs.
MLA: compresses KV information into latent vectors; study as cache compression.
```

Harness tests should compare:

```text
- output shape
- logits equivalence where mathematically expected
- cache memory estimate
- cached vs uncached generation agreement
- tokens/sec during generation
```

### MLP variants

```text
Original Transformer FFN: Linear -> ReLU -> Linear
GPT-style FFN:           Linear -> GELU -> Linear
LLaMA-style FFN:         SwiGLU, often with larger hidden multiplier and no bias
MoE FFN:                 router selects top-k expert FFNs per token
```

Harness tests should compare:

```text
- parameter count
- active parameter count
- output shape
- train stability on tiny overfit
- expert utilization for MoE
```

### Post-training variants

```text
SFT: train on target assistant responses.
Reward modeling: learn a scalar preference model.
RLHF/PPO: optimize policy against reward while staying near reference policy.
DPO: optimize directly from chosen/rejected pairs.
GRPO/RLVR: use verifiable rewards and group-relative advantages for reasoning tasks.
```

Harness tests should compare:

```text
- loss masking correctness
- preference loss decreases on synthetic data
- exact-match improvement on verifiable tasks
- format compliance
- reward hacking cases
```

---

## 5. Suggested reading sequence

### Tier A — required spine

1. **Attention Is All You Need** — understand attention, residual blocks, position encodings.
2. **nanoGPT / build-nanoGPT** — understand minimal GPT implementation.
3. **GPT-3 / scaling laws / Chinchilla** — understand why scale and token budget matter.
4. **LLaMA + Llama 2** — understand the modern open decoder block and chat alignment.
5. **FlashAttention + GQA** — understand systems-level attention bottlenecks.
6. **Llama 3** — understand production-era data, multilinguality, code, tools, safety, long context.
7. **MoE + Mixtral + DeepSeekMoE** — understand sparse scaling.
8. **DeepSeek-V2/V3** — understand MLA, MoE efficiency, MTP, balancing, and training-system choices.
9. **DeepSeekMath + DeepSeek-R1** — understand GRPO/RLVR and reasoning model training.
10. **Qwen3 + Kimi K2** — understand hybrid thinking modes and agentic post-training.
11. **Current model docs/cards** — understand what frontier products currently optimize for.

### Tier B — optional depth

- Tokenization internals: BPE, SentencePiece, byte fallback.
- Distributed training: data/tensor/pipeline parallelism, ZeRO/FSDP.
- Inference serving: batching, paged attention, speculative decoding, quantization.
- Safety and policy: model cards, refusal behavior, jailbreak robustness, tool-permission security.
- Multimodal: vision encoders, cross-attention/adapters, native multimodal pretraining.

---

## 6. Evaluation plan

Do not rely on a single metric. Use a layered evaluation stack.

### Always-on local metrics

```text
- train loss
- validation loss
- bits per byte or perplexity
- tokens/sec train
- tokens/sec decode
- peak memory
- parameter count
- active parameter count, if MoE
- KV cache bytes/token/layer
```

### Behavioral toy evals

```text
- exact arithmetic
- JSON formatting
- instruction following
- refusal sanity checks
- long-context needle retrieval
- tool-call correctness
- simple bug-fix tasks with unit tests
```

### External evals to learn later

```text
- EleutherAI lm-evaluation-harness for common language tasks
- SWE-bench or SWE-bench Verified for software-engineering agents
- Long-context retrieval/needle tests
- Tool-use benchmarks and custom repo-specific tests
```

### Metric logging schema

```json
{
  "phase": "phase_06_kv_cache_gqa",
  "git_commit": "<sha>",
  "config": "configs/gqa_tiny.yaml",
  "seed": 1337,
  "params_total": 1234567,
  "params_active": null,
  "train_loss_start": 4.2,
  "train_loss_final": 3.1,
  "val_loss": 3.3,
  "tokens_per_sec_train": 12000,
  "tokens_per_sec_decode": 220,
  "peak_memory_mb": 900,
  "kv_cache_bytes_per_token_per_layer": 8192,
  "notes": "cached logits match full-prefix logits within 1e-4"
}
```

---

## 7. Current-source pack

Use these as starting points. Re-check all “latest model” claims before relying on them.

### Foundations and codebases

- Attention Is All You Need — https://arxiv.org/abs/1706.03762
- nanoGPT — https://github.com/karpathy/nanogpt
- build-nanoGPT — https://github.com/karpathy/build-nanogpt
- nanochat — https://github.com/karpathy/nanochat
- GPT-3: Language Models are Few-Shot Learners — https://arxiv.org/abs/2005.14165
- Chinchilla: Training Compute-Optimal Large Language Models — https://arxiv.org/abs/2203.15556

### Modern block components

- RMSNorm — https://arxiv.org/abs/1910.07467
- RoPE / RoFormer — https://arxiv.org/abs/2104.09864
- GLU/SwiGLU variants — https://arxiv.org/abs/2002.05202
- GQA — https://arxiv.org/abs/2305.13245
- FlashAttention — https://arxiv.org/abs/2205.14135
- FlashAttention-2 — https://arxiv.org/abs/2307.08691
- FlashAttention-3 — https://arxiv.org/abs/2407.08608

### Llama and open-weight model line

- LLaMA original blog — https://ai.meta.com/blog/large-language-model-llama-meta-ai/
- Llama 2 paper — https://arxiv.org/abs/2307.09288
- Llama 3 Herd paper — https://arxiv.org/abs/2407.21783
- Llama 4 official blog — https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- Llama 4 model page — https://www.llama.com/models/llama-4/

### MoE and DeepSeek

- Sparsely-Gated MoE — https://arxiv.org/abs/1701.06538
- Mixtral of Experts — https://arxiv.org/abs/2401.04088
- MegaBlocks — https://arxiv.org/abs/2211.15841
- DeepSeekMoE — https://arxiv.org/abs/2401.06066
- DeepSeekMath / GRPO — https://arxiv.org/abs/2402.03300
- DeepSeek-V2 — https://arxiv.org/abs/2405.04434
- DeepSeek-V3 — https://arxiv.org/abs/2412.19437
- DeepSeek-R1 — https://arxiv.org/abs/2501.12948
- DeepSeek official site — https://www.deepseek.com/en/

### Post-training and reasoning

- InstructGPT / RLHF — https://arxiv.org/abs/2203.02155
- DPO — https://arxiv.org/abs/2305.18290
- DeepSeek-R1 Nature article — https://www.nature.com/articles/s41586-025-09422-z
- Qwen3 technical report — https://arxiv.org/abs/2505.09388
- Qwen3 blog — https://qwenlm.github.io/blog/qwen3/

### Agentic and current frontier

- Kimi K2 — https://arxiv.org/abs/2507.20534
- Kimi K2 official page — https://moonshotai.github.io/Kimi-K2/
- Kimi K2.5 — https://arxiv.org/html/2602.02276v1
- Claude Code product page — https://claude.com/product/claude-code
- Claude Code Agent SDK — https://code.claude.com/docs/en/agent-sdk/overview
- OpenAI Codex CLI — https://github.com/openai/codex
- Codex AGENTS.md guide — https://developers.openai.com/codex/guides/agents-md
- OpenAI GPT-5.5 — https://openai.com/index/introducing-gpt-5-5/
- OpenAI models docs — https://developers.openai.com/api/docs/models
- Claude Opus 4.8 — https://www.anthropic.com/news/claude-opus-4-8
- Anthropic model overview — https://docs.anthropic.com/en/docs/about-claude/models/overview
- Gemini models docs — https://ai.google.dev/gemini-api/docs/models
- Gemini 3 developer guide — https://ai.google.dev/gemini-api/docs/gemini-3

### Evaluation

- EleutherAI lm-evaluation-harness — https://github.com/EleutherAI/lm-evaluation-harness
- Reproducible evaluation lessons — https://arxiv.org/abs/2405.14782
- SWE-bench — https://github.com/swe-bench/SWE-bench
- SWE-bench Verified — https://www.swebench.com/verified.html

---

## 8. First three sessions to run

### Session 1 — Get the baseline harness green

Prompt:

```text
Implement Phase 0 from the study guide. Keep it tiny. Add tests for shapes, causal mask, checkpoint round-trip, and one-batch overfit. Run only short smoke tests. Update docs/phase_log.md.
```

Expected output:

```text
- minimal GPT trains on tiny data
- pytest passes
- one-batch loss decreases
- phase_log has metrics
```

### Session 2 — Rebuild attention as a teaching module

Prompt:

```text
Implement Phase 1. Refactor attention into src/llm_lab/attention.py with clear MHA and causal mask code. Keep the current model behavior equivalent. Add tests proving no future-token attention leakage.
```

Expected output:

```text
- attention module is isolated
- causal-mask test fails if mask is removed
- training smoke still passes
```

### Session 3 — Convert the block toward LLaMA style

Prompt:

```text
Implement the first half of Phase 4: RMSNorm, RoPE, and SwiGLU as config-gated components. Do not implement MoE yet. Add focused tests and run a 50-step tiny training smoke.
```

Expected output:

```text
- model can switch baseline vs llama_like config
- tests cover RMSNorm/RoPE/SwiGLU
- tiny train does not NaN
```

---

## 9. North-star outcome

After this study path, you should be able to open a new LLM paper or model card and quickly answer:

1. Is the novelty architecture, data, post-training, systems, evaluation, or product interface?
2. What is the smallest faithful version I can implement in my harness?
3. What test would catch a wrong implementation?
4. What metric should move if the idea works?
5. What claims are impossible to verify at toy scale?
6. What would I need to scale the experiment responsibly?
