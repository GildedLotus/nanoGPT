"""
Module 1 exercise — re-implement CausalSelfAttention.forward from a blank function.

Everything later in the curriculum (RoPE, KV cache, GQA, MLA, sparse attention)
is a modification of the ~15 lines you are about to write. The goal is that you
can explain every line of the attention path before changing any of it.

Rules of the game:
  * Don't look at model.py:52-76 while writing. Diff against it afterwards.
  * Annotate the tensor shape after EVERY line as a comment.
  * You may only use the module's own weights (attn.c_attn, attn.c_proj) and
    plain torch ops — no F.scaled_dot_product_attention. That's Module 6.

Run it:  python lessons/01_attention_exercise.py
It passes when your output matches nanoGPT's to within 1e-5.
"""
import math
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import CausalSelfAttention, GPTConfig  # noqa: E402


def my_attention_forward(attn: CausalSelfAttention, x: torch.Tensor) -> torch.Tensor:
    """Recompute attn(x) by hand.

    What you have to work with:
      attn.c_attn  : nn.Linear(n_embd, 3 * n_embd)  — packed Q,K,V projection
      attn.c_proj  : nn.Linear(n_embd, n_embd)      — output projection
      attn.n_head  : number of heads (head_dim = n_embd // n_head)

    The seven steps (write the shape after each one):
      1. Project x through c_attn, split into q, k, v      (B,T,C) -> 3 x (B,T,C)
      2. Reshape each into heads                            (B,T,C) -> (B,nh,T,hs)
      3. Attention scores q @ k^T, scaled by 1/sqrt(hs)     -> (B,nh,T,T)
         (Before moving on: what goes wrong at depth if you DON'T scale?)
      4. Causal mask: position i may not see j > i          (hint: torch.tril)
      5. Softmax over the last dim                          -> rows sum to 1
      6. Weighted sum of values, merge heads back           (B,nh,T,hs) -> (B,T,C)
      7. Output projection through c_proj

    Dropout is 0 in eval mode — ignore it.
    """
    raise NotImplementedError("Module 1: this is the part you type yourself.")


def main() -> None:
    torch.manual_seed(1337)
    cfg = GPTConfig(block_size=64, vocab_size=65, n_layer=1,
                    n_head=4, n_embd=128, dropout=0.0, bias=False)
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, 16, cfg.n_embd)

    with torch.no_grad():
        want = attn(x)
        got = my_attention_forward(attn, x)

    max_err = (want - got).abs().max().item()
    print(f"max abs error vs model.py: {max_err:.2e}")
    assert torch.allclose(want, got, atol=1e-5), "Not matching yet — keep going."
    print("PASS — your attention matches nanoGPT's. Module 1 complete.")


if __name__ == "__main__":
    main()
