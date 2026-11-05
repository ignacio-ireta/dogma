# Child et al. — Sparse Transformer (2019)

*The paper that taught Transformers to stop looking everywhere all the time.*

**← [[../_MOC|AI/ML Lineage]]**

---

## The Paper

**"Generating Long Sequences with Sparse Transformers"** — Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever, 2019.
- OpenAI
- [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)
- PDF: `papers/1904.10509.pdf`

## Key Ideas

1. **Quadratic attention was the bottleneck.**
   Vanilla self-attention from [[Vaswani Attention Is All You Need 2017|Attention Is All You Need]] scales as O(n²) in time and memory. That is fine for 512 or 2048 tokens. It gets ugly fast when you want long text, raw audio, or image sequences.

2. **Make attention sparse, not blind.**
   The paper replaces full attention with structured sparse patterns so each token attends to only a subset of tokens, while information can still propagate globally across layers.

3. **Two practical patterns matter most:**
   - **Local/fixed attention:** look nearby, preserving short-range structure
   - **Strided attention:** hop across the sequence at regular intervals, preserving long-range reach

   Together, this gives a good compromise: local detail plus global communication.

4. **The complexity drops to O(n√n).**
   That is not linear, but it is a huge practical improvement over O(n²), and it was enough to push Transformers into much longer contexts.

5. **This was not just theory.**
   The paper also adds memory-saving recomputation tricks, deeper-network stabilization, and faster kernels. Very OpenAI-coded, honestly: architecture + systems + brute practicality.

## What the Paper Showed

Sparse Transformers set state-of-the-art density modeling results on:
- **Enwik8** (character-level text)
- **CIFAR-10**
- **ImageNet-64**
- raw **audio** sequences

The broader point was bigger than those benchmarks: **self-attention could stretch far beyond the short-context NLP box people associated it with in 2017.**

## Lineage

- **Built on:** [[Vaswani Attention Is All You Need 2017|Transformers]]
- **Responds to:** the quadratic cost of dense attention
- **Leads into:** [[Brown GPT-3 2020|GPT-3]] — which explicitly uses alternating dense and locally banded sparse attention
- **Runs alongside:** long-context efficiency work that later branches into routing, recurrence hybrids, linear attention, and eventually [[Gu Goel Ré S4 2021|S4]] / [[Gu Dao Mamba 2023|Mamba]]

## Why It Matters for the Lineage

This is one of those papers people forget because it is not as culturally famous as Transformers or GPT-3. But if you skip it, the GPT lineage gets weirdly magical.

It explains a very concrete historical step:
- **2017:** attention works
- **2019:** attention learns to economize
- **2020:** GPT-3 scales that lesson into a frontier model

Sparse Transformer is part of the reason the field started treating context length and compute efficiency as architectural problems, not fixed constraints.

## The Real Historical Role

If [[Kaplan Scaling Laws 2020|Scaling Laws]] gave the field permission to go big, Sparse Transformer helped show how to do that without drowning in quadratic cost.

And if [[Gu Dao Mamba 2023|Mamba]] is the loud modern challenge to Transformer scaling, Sparse Transformer is an earlier, quieter version of the same question:

*Can we keep the good part of attention while paying less for it?*

That question never went away.

---

*The Transformer era did not only grow upward. It also learned, slowly, where it could afford not to look.*
