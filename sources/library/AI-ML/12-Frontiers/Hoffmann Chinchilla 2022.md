# Hoffmann et al. — Training Compute-Optimal Large Language Models (Chinchilla, 2022)

#paper #milestone #read

**Authors:** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre
**Institution:** DeepMind
**Published:** arXiv:2203.15556 (March 2022), NeurIPS 2022
**PDF:** `papers/2203.15556.pdf`

---

## The Question

Kaplan et al. (2020) said: to optimally spend a compute budget, grow the model and barely touch the data. But what if they were wrong? What if current large models are **undertrained** — not too small, but data-starved?

The answer: **yes, catastrophically so.**

---

## The Claim

For compute-optimal training, **model size and training tokens should scale equally** — for every doubling of model size, you should also double the training data.

This directly contradicts [[Kaplan Scaling Laws 2020]], which recommended scaling parameters much faster than data (N ∝ C^0.73 vs D ∝ C^0.27).

The Chinchilla correction: **N ∝ C^0.50, D ∝ C^0.50.** Equal scaling.

---

## The Evidence

Trained over **400 language models** from 70M to 16B parameters, on 5B to 500B tokens. Three independent approaches to estimate the compute-optimal frontier:

1. **Approach 1:** Fix compute budget → train models of different sizes → find the size that minimizes loss. (IsoFLOP curves.)
2. **Approach 2:** Fix model size → train on different amounts of data → estimate compute-optimal data for each size.
3. **Approach 3:** Parametric fit of loss as a function of both N and D simultaneously.

All three converge on the same answer: **equal scaling.**

---

## The Experiment: Chinchilla vs. Gopher

| Model | Parameters | Training Tokens | Compute |
|-------|-----------|----------------|---------|
| **Gopher** | 280B | 300B tokens | same |
| **Chinchilla** | 70B | 1.4T tokens | same |

Same compute budget. Chinchilla is 4× smaller but trained on 4× more data.

**Result:** Chinchilla uniformly and significantly outperforms Gopher on every downstream task. And GPT-3 (175B). And Jurassic-1 (178B). And Megatron-Turing NLG (530B).

On MMLU benchmark: Chinchilla 67.5% vs. Gopher 60.0% — a 7.5% improvement from a model 4× smaller.

---

## Why Kaplan Got It Wrong

The Kaplan paper's experiments were mostly at smaller scales. Their power law fit was extrapolated — and the extrapolation overweighted model size. When you look at larger compute budgets, data becomes the equal partner.

Also: Kaplan et al. didn't fully account for learning rate schedules. At different model sizes, the optimal schedule changes, and this confounds the compute-optimal estimate.

---

## The Implication Cascade

This paper rewrote how the industry builds LLMs:

1. **GPT-3 was ~5× undertrained.** At 175B params, compute-optimal training requires ~3.5T tokens, not 300B. Every 175B model trained on ~300B tokens was leaving capability on the table.
2. **Smaller + more data > bigger + less data.** This shifted the entire industry from "make it bigger" to "feed it more."
3. **Inference cost matters.** A 70B model that beats a 280B model costs 4× less per query. Economics flip overnight.
4. **Data becomes the bottleneck.** If you need tokens proportional to model size, the internet starts to look finite. Llama (Meta, 2023) and subsequent models explicitly cite Chinchilla-optimal training.

---

## Cross-Thread Connections 🔗

- **[[Kaplan Scaling Laws 2020]]** — the direct predecessor and target of revision. Same power-law framework, different exponents. Kaplan: N ∝ C^0.73. Chinchilla: N ∝ C^0.50. The 0.23 gap has trillion-dollar consequences.
- **[[Brown GPT-3 2020]]** — designed following Kaplan's (now revised) laws. GPT-3 was 175B on 300B tokens; Chinchilla says it should have been ~30B on 1.5T tokens for the same compute. GPT-3 was a beautiful demonstration of the *wrong* scaling law.
- **[[Vaswani Attention Is All You Need 2017]]** — the architecture both Gopher and Chinchilla use. The Transformer is the constant; the question is how to feed it.
- **[[Loshchilov AdamW 2019]]** — both models train with AdamW. Learning rate scheduling interacts with the scaling law estimates — part of why Kaplan's numbers were off.

### The Scaling Law Arc (complete)

```
Kaplan (Jan 2020): "Scale the model, not the data."
    → GPT-3 (May 2020): 175B params, 300B tokens — followed Kaplan
        → Chinchilla (Mar 2022): "Scale both equally."
            → Llama 1 (Feb 2023): 65B on 1.4T tokens — followed Chinchilla
                → The current paradigm
```

---

## The Deeper Question

Chinchilla doesn't just revise an exponent. It raises a philosophical question about scaling research: **how much can you trust power law extrapolation?**

Kaplan's laws held perfectly across 7 orders of magnitude in their data range. They just didn't extrapolate correctly. The lesson: empirical power laws are descriptive, not prescriptive. Physics-derived scaling laws come from theory; ML scaling laws come from curve fitting. The gap matters.

This also connects to the broader question in LINEAGE.md's cross-thread #8: power laws in physics vs. ML. Kaplan noted the resemblance to critical exponents in statistical mechanics. But Chinchilla shows the resemblance may be superficial — real physical scaling laws don't shift exponents when you measure at different scales.

---

## References Worth Chasing

- **Touvron et al. (2023)** — Llama — first major model built explicitly on Chinchilla-optimal training
- **Rae et al. (2021)** — Gopher — the 280B baseline Chinchilla beat
- **Muennighoff et al. (2023)** — "Scaling Data-Constrained Language Models" — what happens when Chinchilla says you need more data than exists?
- **Clark et al. (2022)** — Unified Scaling Laws for Routed Language Models — extends to MoE

---

## Status

- [x] Paper read and note created (March 24, 2026)
- [ ] Compare Approach 1 vs 2 vs 3 — which extrapolates best?
- [ ] Trace: Chinchilla → Llama → open-source scaling
- [ ] The data wall: when Chinchilla-optimal training hits the limits of available text
