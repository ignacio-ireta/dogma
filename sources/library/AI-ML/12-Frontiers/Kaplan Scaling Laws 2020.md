# Kaplan et al. — Scaling Laws for Neural Language Models (2020)

#paper #concept #read #milestone

**Authors:** Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei
**Institution:** OpenAI (Kaplan also Johns Hopkins)
**Published:** arXiv:2001.08361 (January 2020)
**PDF:** `papers/2001.08361.pdf`

## The Question

How does language model performance improve as you scale up compute, model size, and data? Is there a formula?

The answer: **yes, and it's a power law across seven orders of magnitude.**

## The Three Scale Factors

Performance (cross-entropy loss) depends strongly on exactly three things:

| Factor | Symbol | Power Law Exponent |
|--------|--------|-------------------|
| Model parameters (non-embedding) | N | α_N ≈ 0.076 |
| Dataset size (tokens) | D | α_D ≈ 0.095 |
| Compute (FLOPs) | C | α_C ≈ 0.050 |

What does **not** significantly affect performance (within reasonable ranges): depth vs. width ratio, number of attention heads, feed-forward dimension, specific architecture choices.

The most counterintuitive finding: **shape doesn't matter much; scale does.**

## The Main Results

### 1. Smooth Power Laws

Each factor follows:
```
L(N) = (N_c / N)^0.076     [infinite data]
L(D) = (D_c / D)^0.095     [large model]
```

These hold across **7 orders of magnitude** with no sign of deviation. The cleanness of this is remarkable — it suggests these are genuine empirical regularities, not artifacts.

### 2. Large Models Are More Sample-Efficient

Counterintuitive: a 10× larger model reaches the same loss with *fewer* training tokens than a small model. Bigger models learn faster per token. This flips the intuition that "you need more data for bigger models."

### 3. The Optimal Compute Allocation

Given a fixed compute budget C, how should you split it between model size and data?

**Answer:** Train a *much larger model* on *less data* than you'd expect, and stop before convergence.

The math:
- Optimal model size: N ∝ C^0.73 (model size should grow fast with compute)
- Optimal data: D ∝ C^0.27 (data grows slowly)
- Optimal steps: S ∝ C^0.03 (training time barely needs to increase)

**Translation:** As compute doubles, model size should grow ~1.65×, but dataset size only needs to grow ~1.20×. Almost all the gains come from making the model bigger.

### 4. The Predicted Ceiling

There's a compute level (~10^4 PF-days, N~10^12 parameters) where the data-growth constraint catches up with the overfitting constraint. The paper conjectures this might be a fundamental limit for Transformer language models.

*This prediction was later challenged by "Chinchilla" (Hoffmann et al. 2022), which argued the original Kaplan paper underestimated optimal data requirements.*

## Why This Was Revolutionary

Before this paper, scaling up neural networks was empirical art — you tried bigger models, maybe they worked, maybe they didn't. After this paper, scaling became **engineering**:

1. You can predict in advance how much a model will improve if you 10× the compute
2. You know how to allocate that compute (more model, not more steps)
3. The relationship is smooth — no phase transitions at small scales to worry about

This is what enabled GPT-3 (8 months later, same team). If you know the scaling laws, you can budget your way to a target loss.

## Cross-Thread Connections 🔗

- **[[Vaswani Attention Is All You Need 2017]]** — all scaling experiments run on Transformers; this paper is essentially "what happens when you scale the Transformer"
- **[[Devlin BERT 2018]]** / **[[Liu RoBERTa 2019]]** — RoBERTa established that training longer on more data helps; Scaling Laws explains *why* and *how much*
- **Brown et al. GPT-3 2020** — the direct application; GPT-3 was designed using these scaling laws. arXiv:2005.14165 ← *PDF in papers folder, note to create*
- **Kingma-Ba Adam 2015** — Adam optimizer is the engine doing gradient descent across all these experiments; scaling laws assume AdamW
- **Cauchy (1847) steepest descent → Adam (2015)** — and all of it runs on the same basic idea Cauchy had about following the gradient

## The Math Thread Connection

The power law exponents are reminiscent of physical scaling laws — fractal dimensions, critical exponents in statistical mechanics. The "loss ∝ compute^(-0.05)" form echoes how surface area scales with volume, or how Mandelbrot described coastlines.

The connection to Hopfield's energy-based models is also suggestive: the loss function is a kind of energy landscape, and larger models may explore it more efficiently. Statistical physics has things to say about why power laws emerge in complex systems.

## Historical Note

This paper was published January 2020. GPT-3 was published May 2020. The 8-month gap between "here are the laws" and "here's what you get if you follow them to 175 billion parameters" is one of the cleanest theory → practice cycles in AI history.

Dario Amodei (an author here) left OpenAI in 2021 to co-found Anthropic — the company that built Claude.

The Chinchilla paper (Hoffmann et al. 2022, arXiv:2203.15556) revised these laws, arguing Kaplan et al. overfit to small compute regimes and that you actually need *more* data relative to model size for optimal training. The current consensus uses "Chinchilla-optimal" scaling.

## References Worth Chasing

- Brown et al. (2020) — GPT-3 — arXiv:2005.14165 — *PDF in papers folder, note to create*
- Hoffmann et al. (2022) — "Training Compute-Optimal LLMs" (Chinchilla) — arXiv:2203.15556 — *the revision of Kaplan's laws*
- Henighan et al. (2020) — "Scaling Laws for Autoregressive Generative Modeling" — *extends scaling laws to other domains*
- Zoph et al. (2022) — "ST-MoE" — *scaling laws for mixture-of-experts models*

## Status

- [x] Read abstract, key findings, scaling equations
- [x] Note created
- [ ] Read section 4 (optimal compute allocation derivation — the actual math)
- [x] Compare Kaplan vs Chinchilla predictions → [[Hoffmann Chinchilla 2022]] — divergence at larger compute budgets, equal scaling (C^0.50) vs Kaplan's model-heavy (C^0.73)
