# Building LLMs — An Integrated Lesson Plan

**What this is.** A single learning path that combines three sources:

1. **The `claude` branch curriculum** (`transformers-from-scratch-harness-curriculum.md`) — a focused, hands-on march that evolves nanoGPT into a modern (DeepSeek-era) architecture, one verifiable diff at a time. *This is the spine.*
2. **The `codex` branch study guide** (`llm_development_study_guide_harness.md`) — a broader roadmap (post-training, tools, multimodal, eval) plus a curated reading list and per-phase task cards. *This is the map and reference.*
3. **Stanford CS336 — Language Modeling from Scratch** (Spring 2026) — the rigorous full-stack course whose lectures and 5 assignment repos fill the three pillars the branch docs skip: **systems/kernels, data curation, and scaling laws.** *These are the reference implementations to read and reverse-engineer.*

**How to use it (important — read this first).** You are not going to implement everything from scratch. The plan splits every topic into two columns:

- **BUILD** — the conceptually load-bearing parts worth typing yourself, done on the tiny nanoGPT harness (fast feedback, no GPU budget). These come from the branch curricula.
- **READ / REVERSE-ENGINEER** — the heavy systems/data/scaling work you study rather than rebuild. For these, the CS336 assignment repos are ideal: each ships **failing `pytest` unit tests** and an **`adapters.py`** that names every function a correct solution must provide. The test suite *is* a precise spec. You read the handout, read the tests, and either (a) implement just the one or two functions that teach the core idea, or (b) read a reference solution and trace it against the tests. Either way you reverse-engineer understanding from a known-good target.

---

## 1. The combined map

Each row is one learning unit. "Build" = type it on nanoGPT. "Read" = study the CS336 reference and/or watch the lecture.

| # | Topic | BUILD (hands-on, nanoGPT-scale) | READ / REVERSE-ENGINEER (CS336 + papers) |
|---|---|---|---|
| 0 | Baseline + harness | Get nanoGPT char-Shakespeare training & sampling; tag `baseline`; start a metrics scoreboard | CS336 **A1** repo layout; lecture *PyTorch & Resource Accounting* |
| 1 | Tokenization | Decode `train.bin` back to text; compare char vs BPE | **A1 BPE tokenizer** (read `tests/test_train_bpe.py` + `test_tokenizer.py`) — implement BPE train/encode/decode if you want the one deep dive; lecture *Tokenization* |
| 2 | Vanilla transformer block | Re-derive `CausalSelfAttention.forward`, annotate shapes, `allclose` vs original | **A1 model tests**; lecture *Architectures & Hyperparameters*; *Attention Is All You Need* |
| 3 | Training loop internals | Already in nanoGPT — read AdamW, LR warmup+cosine, grad clip, weight decay grouping | **A1 optimizer/training tests**; nanoGPT `train.py` |
| 4 | Modern block: RMSNorm + SwiGLU + pre-norm | **Build it** (branch Module 2) behind config flags | CS336 *Architectures* lecture; LLaMA paper; RMSNorm/SwiGLU papers |
| 5 | RoPE | **Build `apply_rope`** + length-extrapolation test (branch Module 3) | RoFormer paper; CS336 architecture lecture |
| 6 | KV cache | **Build** the incremental-decode cache (branch Module 4); add KV-bytes/token to scoreboard | CS336 *Inference* lecture; **A2** generation benchmarks |
| 7 | GQA / MQA | **Build** `n_kv_heads` grouping (branch Module 5); sweep and watch cache shrink | GQA paper; CS336 architecture lecture |
| 8 | **Systems: kernels & FlashAttention** | Swap manual attn → `scaled_dot_product_attention`; profile mem vs manual | ⭐ **A2 — FlashAttention-2 in Triton** (read the handout + Triton kernel tests; reverse-engineer the tiling/online-softmax). Lectures *GPUs*, *Kernels/Triton*. FlashAttention papers |
| 9 | **Systems: distributed training** | (Study only — nanoGPT's DDP path in `train.py` is the tiny version) | ⭐ **A2 — DDP + optimizer state sharding**; lectures *Parallelism 1 & 2*; ZeRO/FSDP notes |
| 10 | **Scaling laws** | Use nanoGPT's `estimate_mfu` + `transformer_sizing.ipynb`; run 3 tiny configs, plot loss vs compute | ⭐ **A3 — fit scaling laws on IsoFLOPs data, predict compute-optimal model under a FLOPs budget**; lectures *Scaling Laws 1 & 2*; Chinchilla |
| 11 | MoE FFN | **Build** router + top-k gating + load-balance loss (branch Module 7); add active-params/token column | CS336 *Mixture of Experts* lecture; Mixtral, DeepSeekMoE |
| 12 | MLA (latent KV) | **Build** down/up KV projection (branch Module 8); compare cache bytes MHA vs GQA vs MLA | DeepSeek-V2 paper |
| 13 | Multi-token prediction | **Build** the +2 auxiliary head (branch Module 9, optional) | DeepSeek-V3 paper |
| 14 | **Data curation** | (Study only — nanoGPT's `prepare.py` is the toy version) | ⭐ **A4 — Common Crawl HTML→text, quality/PII filtering, deduplication, measure downstream impact**; lectures *Data 1–3* |
| 15 | **Evaluation** | Add a couple of behavioral toy evals (arithmetic, needle-in-haystack) to the harness | CS336 *Evaluation* lecture; codex doc §6 eval plan; lm-eval-harness |
| 16 | Post-training: SFT | **Build** an SFT loop with prompt-token masking on tiny synthetic chat data | ⭐ **A5 — SFT on reasoning traces**; lecture *Alignment (SFT/RLHF)*; InstructGPT |
| 17 | Post-training: DPO | **Build** a tiny DPO loss on synthetic preference pairs (codex Phase 5) | DPO paper; CS336 alignment lecture |
| 18 | Reasoning RL: GRPO/RLVR | **Build** a miniature GRPO loop on a verifiable toy task (arithmetic) — group-relative advantage + KL (branch Module 10) | ⭐ **A5 — Expert Iteration + GRPO on MATH**; lecture *Alignment (RL)*; DeepSeekMath, DeepSeek-R1 |
| 19 | Frontier breadth (optional) | Tool-use / agentic toy loop; long-context needle eval (codex Phases 7, 13) | codex doc §2 phases 12–15; model cards |

⭐ = a CS336 pillar the branch curricula omit. These are the highest-value "reverse-engineer, don't rebuild" units.

---

## 2. CS336 in brief (the reference course)

**Format.** Graduate course, "from scratch but not reckless." No scaffolding — failing unit tests define correctness. Five assignments are the backbone. Lectures are on YouTube (Stanford channel) and the site posts handouts + code.

**Lecture arc (~19):** Tokenization → PyTorch & Resource Accounting → Architectures & Hyperparameters → Mixture of Experts → GPUs → Kernels/Triton → Parallelism 1–2 → Scaling Laws 1–2 → Inference → Evaluation → Data 1–3 → Alignment (SFT/RLHF, then RL/GRPO) + guest lectures.

**The 5 assignments (public GitHub repos under `github.com/stanford-cs336`):**

| Asgn | Repo | You implement | Why read it |
|---|---|---|---|
| 1 | `assignment1-basics` | BPE tokenizer, Transformer, cross-entropy, AdamW, training loop; train on TinyStories/OpenWebText | The clean from-scratch spec for everything nanoGPT already gives you — good for verifying your mental model |
| 2 | `assignment2-systems` | Profiling harness, **FlashAttention-2 Triton kernel**, **DDP**, optimizer state sharding | The systems pillar the branch docs explicitly skip — the single best thing to add |
| 3 | `assignment3-scaling` | Fit scaling laws on IsoFLOPs data; query a training API under a FLOPs budget; predict compute-optimal size & loss | Teaches the *empirical discipline* of scaling that toy models can't |
| 4 | `assignment4-data` | Common Crawl HTML→text, harmful/PII filtering, deduplication, train & measure | The data pillar — the thing that actually determines model quality |
| 5 | `assignment5-alignment` | MATH zero-shot baseline, SFT on traces, Expert Iteration, **GRPO** | A real (not toy) reasoning-RL pipeline to compare against the miniature one you build |

**How to reverse-engineer an assignment without doing it (your stated goal):**
1. Read the handout PDF (`cs336_spring2026_assignmentN_*.pdf` in each repo) for the conceptual framing.
2. Open `tests/` and `tests/adapters.py` — the adapter function signatures are an exact, named spec of every component a correct solution needs.
3. Run `uv sync && pytest` once to see the `NotImplementedError` map of the whole assignment.
4. Implement *only* the one or two functions that carry the core idea (e.g. the online-softmax step of FlashAttention, the GRPO advantage). Let the tests tell you when it's right.
5. For the rest, read a public reference solution (several exist on GitHub) and trace it against the tests — understanding-by-reading, not by typing.

---

## 3. How the three sources divide the labor

- **nanoGPT + branch curricula = the architecture & post-training core you BUILD.** Tiny, fast, every change is a flag + a number on the scoreboard. This is where typing the load-bearing lines (attention, RoPE, the router, the GRPO advantage) actually teaches you.
- **CS336 A2/A3/A4 = the three pillars you READ.** Systems/kernels, scaling-law fitting, and real data work are expensive and infrastructure-heavy; reverse-engineering the CS336 references gets you the understanding without the GPU budget or the weeks of effort.
- **CS336 A1/A5 = correctness anchors.** A1 is a clean from-scratch spec to check your nanoGPT mental model against; A5 is a real reasoning-RL pipeline to compare against your miniature GRPO.
- **codex doc = the breadth map + reading list** for everything past the architecture core (tools, agents, multimodal, eval) and the per-phase task-card template.

---

## 4. Suggested ordering

A pragmatic path that front-loads understanding and defers the heavy reading until you have the architecture in your hands:

1. **Weeks 1–2 — Core (BUILD):** map units 0–7. Get the LLaMA-style block (RMSNorm/SwiGLU/RoPE), KV cache, and GQA working on nanoGPT with a live scoreboard. Watch CS336 *Tokenization*, *Architectures*, *MoE* lectures alongside.
2. **Week 3 — Systems (READ):** unit 8–9. Watch *GPUs*, *Kernels/Triton*, *Parallelism*; reverse-engineer A2's FlashAttention kernel; implement just the online-softmax core.
3. **Week 4 — Scaling (READ + light BUILD):** unit 10. Watch *Scaling Laws 1–2*; work through A3's IsoFLOPs fitting; run your own 3-config tiny sweep.
4. **Week 5 — Modern efficiency (BUILD):** units 11–13. MoE, then MLA, then MTP. This is the DeepSeek lineage as concrete diffs.
5. **Week 6 — Data + Eval (READ):** units 14–15. Watch *Data 1–3*; reverse-engineer A4's dedup/filter pipeline; add toy evals.
6. **Weeks 7–8 — Post-training (BUILD + READ):** units 16–18. SFT → DPO → miniature GRPO on a verifiable task; compare against A5's real MATH pipeline.
7. **Optional — Frontier breadth:** unit 19 from the codex doc.

Adjust freely — the map (Section 1) is the source of truth; this ordering is just one route through it.

---

## 5. Resource pack

**Codebases**
- nanoGPT (this repo / baseline) — https://github.com/karpathy/nanoGPT
- nanochat (fuller small-scale ChatGPT stack) — https://github.com/karpathy/nanochat
- CS336 assignments — https://github.com/stanford-cs336 (`assignment1-basics` … `assignment5-alignment`)
- CS336 lectures repo — https://github.com/stanford-cs336/spring2025-lectures

**CS336**
- Course site — https://cs336.stanford.edu/
- Spring 2025 archive (schedule + materials) — https://stanford-cs336.github.io/spring2025/
- Lecture videos (Stanford YouTube) — playlist `PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_`

**Papers, one per build unit**
- Attention Is All You Need — https://arxiv.org/abs/1706.03762
- RoPE / RoFormer — https://arxiv.org/abs/2104.09864
- RMSNorm — https://arxiv.org/abs/1910.07467 · SwiGLU — https://arxiv.org/abs/2002.05202
- GQA — https://arxiv.org/abs/2305.13245 · FlashAttention — https://arxiv.org/abs/2205.14135
- Chinchilla (scaling) — https://arxiv.org/abs/2203.15556
- Mixtral — https://arxiv.org/abs/2401.04088 · DeepSeekMoE — https://arxiv.org/abs/2401.06066
- DeepSeek-V2 (MLA) — https://arxiv.org/abs/2405.04434 · V3 (MTP) — https://arxiv.org/abs/2412.19437
- InstructGPT (RLHF) — https://arxiv.org/abs/2203.02155 · DPO — https://arxiv.org/abs/2305.18290
- DeepSeekMath (GRPO) — https://arxiv.org/abs/2402.03300 · DeepSeek-R1 — https://arxiv.org/abs/2501.12948

(The `codex` branch doc has a longer, categorized source list — use it for everything past the architecture core.)

---

### Definition of done
You can open any new model's tech report and answer: is the novelty **architecture, systems, data, scaling, post-training, or product**? What's the smallest faithful version you'd build on the nanoGPT harness? Which CS336 assignment is the reference for the part you *won't* build? What metric should move if it works?
