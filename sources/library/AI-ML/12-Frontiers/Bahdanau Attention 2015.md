# Bahdanau, Cho, Bengio — Neural Machine Translation by Jointly Learning to Align and Translate (2015)

#paper #milestone #concept #read

**Authors:** Dzmitry Bahdanau, KyungHyun Cho, Yoshua Bengio
**Institution:** Jacobs University Bremen; Université de Montréal
**Published:** ICLR 2015 (arXiv:1409.0473, submitted Sep 2014)
**PDF:** `papers/1409.0473.pdf`

## The Problem This Solved

The dominant neural machine translation architecture was an **encoder-decoder**:
1. Encoder RNN reads source sentence → compresses it into a **single fixed-length vector**
2. Decoder RNN generates target sentence from that vector

The bottleneck: **one vector** to represent an entire sentence. Performance collapsed on long sentences — all the information had to fit in one compressed code.

## The Solution: Attention

Instead of compressing everything into one vector, let the decoder **look back** at all encoder states and decide which parts of the source are most relevant *right now* — at each step of generation.

At each decoder step t, compute:
1. **Alignment scores** for each encoder state hᵢ: `e(t,i) = score(sₜ₋₁, hᵢ)` (how relevant is source position i for this output step?)
2. **Attention weights** via softmax: `α(t,i) = softmax(e(t,i))`
3. **Context vector**: `cₜ = Σᵢ α(t,i) · hᵢ` — weighted sum of encoder states
4. Decoder generates next word from `(sₜ₋₁, cₜ, yₜ₋₁)`

The model **jointly learns** to translate AND to align source-target words — no separate alignment step.

## Key Ideas

- **Soft attention vs hard attention** — uses "soft" differentiable weights (sum over all positions), not a hard binary decision. Fully differentiable → trained end-to-end via backprop
- **Dynamic context** — different context vector at each decoder step, not one fixed representation
- **Interpretable alignments** — attention weights form a matrix you can visualize; they look like real linguistic alignment tables
- **BLEU score:** Matched SOTA phrase-based systems (Moses) on English→French
- **Long sentences:** Unlike encoder-decoder, performance doesn't degrade with sentence length

## Cross-Thread Connections 🔗

- **[[Luong Pham Manning Attention 2015]]** — turns this first successful learned alignment into a configurable layer: global vs. local scope, multiple score functions, and input-fed alignment history.
- **[[Vaswani Attention Is All You Need 2017]]** — takes the attention lineage beyond recurrent translation and makes attention *the whole architecture*, removing the RNN entirely
- **[[Hochreiter-Schmidhuber LSTM 1997]]** — attention was invented to fix LSTM's weakness (fixed-length bottleneck)
- **[[Hamilton Quaternions 1843]]** — the context vector computation `cₜ = Σ α · h` is a **weighted dot product** — Hamilton's dot product, 172 years later, computing attention
- **[[Linear Algebra]]** — attention = learned linear combination. The attention mechanism is matrix multiplication: all alignments computed in parallel via `QKᵀ`

## What Makes This a Milestone

Before: encoder-decoder (Cho 2014, Sutskever 2014) — compress → generate
After Bahdanau: compress + selectively attend → generate
After Vaswani: *only* attend → generate (no compress at all)

> The jump from "attention as add-on to RNN" → "attention as the whole thing" (Vaswani 2017) is the pivotal step. Bahdanau's paper made that jump imaginable.

## The Deeper Idea

Language is not processed as a single compressed memory — it's processed *relationally*. The right translation of word X depends on which source words are relevant to X. Attention formalizes this: meaning is context-dependent, and context is a learned alignment.

## References Worth Chasing

- Cho et al. (2014) — "Learning Phrase Representations using RNN Encoder-Decoder" — arXiv:1406.1078 — *predecessor to this paper*
- [[Luong Pham Manning Attention 2015|Luong et al. (2015)]] — "Effective Approaches to Attention-based Neural Machine Translation" — arXiv:1508.04025 — *simplified and generalized attention variants*
- Vaswani et al. (2017) — [[Vaswani Attention Is All You Need 2017]] — *the conclusion of this thread*

## Status

- [x] Read abstract, introduction, model description
- [x] Note created
- [ ] Read section 4 (alignment visualization figures — look at those alignment matrices!)
- [x] Confirmed and linked the attention lineage: Bahdanau → Luong → Vaswani
