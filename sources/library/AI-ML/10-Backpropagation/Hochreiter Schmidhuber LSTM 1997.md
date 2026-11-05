# Hochreiter & Schmidhuber — Long Short-Term Memory (1997)

#paper #milestone #read

**Authors:** Sepp Hochreiter, Jürgen Schmidhuber
**Institution:** Technische Universität München
**Published:** Neural Computation 9(8), pp. 1735–1780, November 15, 1997
**PDF:** `papers/hochreiter-schmidhuber-lstm-1997.pdf`

---

## The Problem: Vanishing Gradients in RNNs

[[Rumelhart Hinton Williams Backpropagation 1986]] gave us a way to train multi-layer networks via the chain rule. When applied to recurrent networks (unfolding through time), this becomes **backpropagation through time (BPTT)** — the same algorithm, but the "layers" are time steps.

The problem: each time step multiplies the gradient by the recurrent weight matrix. Over many time steps:

$$\frac{\partial L}{\partial w_t} = \frac{\partial L}{\partial w_{t+1}} \cdot \frac{\partial s_{t+1}}{\partial s_t}$$

If the spectral radius (largest eigenvalue) of the weight matrix is **less than 1**, gradients **vanish** exponentially — early time steps get no signal. If it's **greater than 1**, gradients **explode**. This isn't a bug; it's a mathematical inevitability of repeated matrix multiplication.

Hochreiter first formalized this in his 1991 dissertation (German). The 1997 paper is where he and Schmidhuber present the solution.

---

## The Solution: Constant Error Carousel (CEC)

The core innovation is deceptively simple: **a memory cell with linear self-connection whose weight is fixed at 1.0.**

When the recurrent weight equals exactly 1, the error signal neither decays nor explodes as it flows backward through time. It stays constant. Hence: **Constant Error Carousel**.

$$c_t = c_{t-1} + i_t \cdot \tilde{c}_t$$

Where:
- $c_t$ = cell state at time $t$
- $i_t$ = input gate (sigmoid: how much new information to add)
- $\tilde{c}_t$ = candidate values from current input

The CEC is wrapped by three learned gating functions:
- **Input gate** ($i_t$): controls writing to the cell
- **Forget gate** ($f_t$, added later by Gers et al. 2000): controls erasing from the cell
- **Output gate** ($o_t$): controls reading from the cell

These gates use sigmoid activations (0–1 range), so they learn *when* to remember, forget, and expose information — not just *what* to store. This is **learned temporal selectivity**, which is qualitatively different from a simple RNN's fixed recurrence.

---

## Why It Worked

The original LSTM could bridge **time lags exceeding 1000 discrete steps** — an order of magnitude improvement over vanilla RNNs. But what made LSTM dominant for nearly two decades (2006–2017) wasn't just the architecture:

1. **Gating is biologically plausible.** Neuroscientists later discovered that cortical neurons do something remarkably similar — dendritic gating of information flow.
2. **Stackable.** You can layer LSTMs on top of each other, building hierarchical representations of sequences. Deep LSTM networks were state-of-the-art for machine translation until Transformers arrived.
3. **Differentiable end-to-end.** Unlike hidden Markov models or other sequence models that required marginal inference, LSTMs train with pure backpropagation.

---

## The Lineage Position

LSTM sits at a critical junction in the AI/ML tree:

```
Rumelhart-Hinton-Williams 1986 (backpropagation)
├── LeCun 1989 (CNNs — feedforward, no sequence problem)
└── Applied to RNNs → BPTT → VANISHING GRADIENT PROBLEM
    └── Hochreiter-Schmidhuber 1997 (LSTM) ← YOU ARE HERE
        ├── Gers et al. 2000 (forget gate — the missing piece)
        ├── Cho et al. 2014 (GRU — simplified LSTM, fewer gates)
        │   └── [[Cho Encoder-Decoder GRU 2014]]
        ├── Sutskever et al. 2014 (sequence-to-sequence with LSTM)
        │   └── [[Bahdanau Attention 2015]] → attention replaces encoder-decoder bottleneck
        └── Vaswani et al. 2017 (Transformers — attention without recurrence)
```

LSTM solved vanishing gradients for 20 years. Then [[Vaswani Attention Is All You Need 2017]] asked: *what if we drop recurrence entirely?* The Transformer's self-attention mechanism provides direct access to any position in the sequence — no need for information to trickle through time steps. LSTMs went from king to legacy in ~5 years.

But the gating idea didn't die. Modern architectures still use gating mechanisms (e.g., FFN layers in Transformers, residual stream normalization). The intuition — *learn when to let information flow* — persists.

---

## Key Contributions

1. **Formal analysis of the vanishing gradient problem** — Hochreiter's framework (based on eigenvalues of the Jacobian) became the standard way to think about gradient flow in deep networks
2. **Constant Error Carousel** — the architectural fix: linear unit with weight = 1 preserves gradient flow
3. **Learned gating mechanisms** — input/output gates (and later forget gates) as differentiable information flow controllers
4. **Empirical demonstration** — learning tasks with >1000-step dependencies, previously impossible for RNNs

---

## Cross-Thread Connections 🔗

### To Mathematics
- **Eigenvalue decomposition** — the vanishing gradient analysis uses spectral radius of the weight matrix. This connects directly to MVS (eigenvalues are central to PCA, spectral theorem)
- **Matrix multiplication chain rule** — BPTT applies the multivariable chain rule repeatedly; the exponential decay/growth comes from properties of matrix powers

### Within AI/ML
- **[[Rumelhart Hinton Williams Backpropagation 1986]]** ← LSTM is impossible without backprop; BPTT is backprop unrolled in time
- **[[Hopfield Neural Networks 1982]]** ← Hopfield nets also address long-range dependencies but through energy minima, not gradient flow. Different approach to the same problem.
- **[[Hinton Sejnowski Boltzmann Machines 1983]]** ← Also addresses sequence/structure learning, but through probabilistic undirected models rather than gated deterministic RNNs
- **[[Cho Encoder-Decoder GRU 2014]]** → GRU is a simplified LSTM — merges input/forget into update gate, merges cell state and hidden state
- **[[Bahdanau Attention 2015]]** → Attention was originally designed to augment LSTM encoders (letting the decoder look at all encoded positions)
- **[[Vaswani Attention Is All You Need 2017]]** → Transformers replaced LSTMs, but inherited the "controlled information flow" idea
- **[[He ResNets 2015]]** ← ResNet skip connections also address gradient flow (in depth rather than in time). Same problem domain: how to train very deep computation graphs
- **[[Kaplan Scaling Laws 2020]]** ← Scaling laws apply to LSTMs too — they show suboptimal scaling compared to Transformers, which partly explains why Transformers won

---

## Limitations

- **Sequential processing** — can't parallelize across time steps (the killer flaw that Transformers exploited)
- **Computationally expensive per time step** (4x parameters of a standard RNN cell)
- **Still struggles with very long sequences** (>10K tokens) — CEC prevents vanishing but doesn't solve everything
- **The "constant" in CEC assumes exact arithmetic** — in practice, floating-point precision still causes slow degradation over thousands of steps

---

## Status

- [x] PDF downloaded (April 4, 2026)
- [x] Note created (April 5–6, 2026)
- [x] Gers, Schmidhuber & Cummins 2000 (Learning to Forget) — added the forget gate; critical refinement. Note created: [[Gers Learning to Forget LSTM 2000]]
- [ ] Sutskever, Vinyals & Le 2014 (Sequence to Sequence Learning with Neural Networks) — the encoder-decoder breakthrough that used LSTMs
