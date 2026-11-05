# Dao & Gu — Transformers are SSMs / Mamba-2 (2024)

*The first Mamba paper proved selective SSMs could compete. This one explains why — and makes them faster.*

**← [[../_MOC|AI/ML Lineage]]**

---

## The Paper

**"Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"** — Tri Dao, Albert Gu, 2024.
- Princeton University / Carnegie Mellon University
- ICML 2024
- [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)

## Paper Reference

This is the paper that introduces **Mamba-2**, but it does so in a slightly mischievous way: instead of titling the paper "Mamba-2," Dao and Gu frame it as a theoretical claim — **Transformers and state space models are more closely related than they look**.

That matters because the original [[Gu Dao Mamba 2023|Mamba]] note can otherwise feel like a bolt from the blue: clever, fast, powerful, but still a little magical. Mamba-2 is the cleanup pass. It gives the architecture a cleaner mathematical story, a more hardware-friendly formulation, and a much stronger answer to the question: *why should SSMs be taken seriously as a foundation-model primitive?*

## Key Ideas

1. **State Space Duality (SSD).**
   The central claim is that a certain class of selective state space models can be written in two equivalent ways:
   - a **linear recurrent / SSM form**
   - a **quadratic attention-like matrix form**

   In other words, recurrence and attention are not totally separate species. Under the right structure, they are two views of the same underlying sequence transformation.

2. **Mamba-2 is a refinement of Mamba-1, not a replacement in spirit.**
   Mamba-1 used a diagonal structured state transition. Mamba-2 adds more structure — essentially a **scalar-times-identity recurrence within each head** — which sounds like a restriction, but it unlocks much better algorithms and cleaner theory.

3. **Training gets much faster.**
   The authors report the new core layer is **2–8× faster** than Mamba-1 while staying competitive with Transformers on language modeling. This is the practical point. Elegant theory is nice; faster training is why anyone in the room pays attention.

4. **The matrix-multiplication story matters.**
   GPUs and TPUs are built to devour matrix multiplies. Mamba-1 was fast at inference but less hardware-friendly during training because its selective scan machinery did not map as naturally onto accelerator sweet spots. Mamba-2 pushes the model closer to the hardware's native language.

5. **Semiseparable matrices become the bridge concept.**
   The paper shows that SSMs can be understood as structured matrix transformations over a class of matrices called **semiseparable matrices**. This is the real intellectual bridge: control theory, recurrence, linear attention, and structured matrix algorithms are suddenly in one family photo.

## What Actually Changed from Mamba-1

Compared with [[Gu Dao Mamba 2023|Mamba]]:

- **More structure on the recurrent dynamics**
  - Mamba-1: diagonal state transition
  - Mamba-2: scalar-times-identity state transition within a head

- **Multi-head SSM design becomes explicit**
  - Mamba-2 looks more Transformer-like organizationally: multiple heads, shared dynamics inside each head, better scaling to wider models

- **Dual computation modes**
  - linear recurrent form for efficiency
  - attention-like matrix form for theory and algorithm design

- **Much larger practical state sizes**
  - the follow-up explanations from Tri Dao emphasize that Mamba-2 can use much larger state dimensions while still training faster

The vibe is: Mamba-1 proved the architecture class had teeth; Mamba-2 made it legible.

## Lineage

- **Built on:** [[Gu Goel Ré S4 2021|S4]] as the structured-SSM breakthrough, then [[Gu Dao Mamba 2023|Mamba]] as the selective-SSM language-model breakthrough
- **Responds to:** the open question left by Mamba-1 — can SSMs be both *theoretically unified* with attention and *hardware-efficient enough* to scale cleanly?
- **Connected to:** [[Vaswani Attention Is All You Need 2017|Transformers]], [[Child Sparse Transformer 2019|efficient attention]], [[Kaplan Scaling Laws 2020|scaling laws]], and structured linear algebra through semiseparable matrices
- **Leads to:** hybrid Mamba/Transformer systems, broader SSM adoption, and a more mature post-Transformer architecture conversation

## Why It Matters

### For the lineage

This is the note that closes the conceptual gap between:
- **S4:** SSMs become practical
- **Mamba:** SSMs become competitive for language
- **Mamba-2:** SSMs become mathematically unified, faster to train, and easier to compare directly against attention

Without Mamba-2, the SSM thread is interesting.
With Mamba-2, it becomes a serious alternate tradition in foundation-model design.

### The deeper shift

The title is the real provocation: **"Transformers are SSMs."**

Not literally in every engineering detail, of course. But conceptually, the paper argues that the space of efficient sequence models is more unified than the field's branding makes it seem. That is the exciting part. Once recurrence and attention start looking like coordinate systems rather than tribes, architecture research gets a lot more interesting.

## Open Questions

- Will pure SSM stacks beat Transformers broadly, or mostly on long-context / efficiency-sensitive regimes?
- Are hybrids the real destination — attention where exact token-to-token interaction matters, SSMs where compression and memory matter more?
- How much of Mamba-2's advantage will survive once mainstream frameworks and kernels catch up?

---

*Mamba made SSMs exciting. Mamba-2 made them intelligible.*
