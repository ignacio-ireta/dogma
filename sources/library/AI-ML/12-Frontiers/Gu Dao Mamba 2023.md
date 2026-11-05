# Gu & Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (2023)

#paper #frontier #read

**Authors:** Albert Gu (Carnegie Mellon University), Tri Dao (Princeton University)
**Published:** arXiv preprint, December 2023 (v2: May 2024)
**arXiv:** [2312.00752](https://arxiv.org/abs/2312.00752)
**PDF:** `papers/2312.00752.pdf`

## The Moment

Six years after [[Vaswani et al. — Attention Is All You Need (2017)]] made Transformers the universal architecture for foundation models, this paper asked: *What if we don't need attention at all?*

Mamba introduces **selective state space models (SSMs)** — a new class of sequence models that match or exceed Transformer performance while achieving **linear-time complexity** (O(n)) instead of quadratic (O(n²)). For language modeling, Mamba-3B outperforms Transformers of the same size and matches Transformers twice its size.

This isn't incremental improvement. It's a paradigm shift — the first serious challenger to Transformer dominance since 2017.

## Key Ideas

- **State Space Models (SSMs)** as alternative to attention — continuous-time systems discretized for sequences, rooted in classical control theory
- **Selective State Spaces** — the breakthrough: make SSM parameters *input-dependent* (not fixed). The model decides what to remember/forget based on each token's content. This solves SSMs' weakness with discrete modalities like language.
- **Hardware-aware parallel algorithm** — selective parameters prevent efficient convolution, so Gu & Dao designed a custom GPU kernel that runs SSMs in parallel in recurrent mode
- **Mamba architecture** — a simple homogeneous block: selective SSM + linear projection. No attention, no MLP blocks. Just the SSM repeated.

## Architecture

```
Mamba Block:
  Input → Linear projection → Selective SSM → Linear projection → Output
          (in, out dims)     (state update)   (projection back)

Selective SSM:
  h_t = A(x_t)h_{t-1} + B(x_t)C(x_t)x_t   ← A, B, C now depend on x_t!
  y_t = D x_t + C(x_t)h_t                 ← D is learnable (skip connection)

Where:
  A(x_t) = Linear(x_t) → state transition matrix
  B(x_t) = Linear(x_t) → input projection
  C(x_t) = Linear(x_t) → output projection
```

The key innovation: **A, B, C are computed per-token, not fixed**. This makes the model "selective" — it can choose to remember or ignore based on content, like attention's strength, but without the O(n²) cost.

## Why It Matters

- **Linear scaling** — O(n) complexity vs. Transformer's O(n²) attention. Million-token sequences become practical.
- **5× faster inference** — Mamba processes 5× more tokens per second than Transformers at same model size
- **State-of-the-art performance** — language, audio, genomics. Mamba-3B beats Transformer-3B, matches Transformer-6B
- **No KV cache** — Transformers need O(n) memory for the key-value cache during inference. Mamba's state is fixed-size (h_t). Constant memory.
- **Real-world scaling** — performance *improves* on real data up to 1M token sequences (many "efficiency" architectures fail at scale)

## Theoretical Connection: Why SSMs?

State space models come from **classical control theory** (1940s-1970s). A continuous-time SSM:
```
h'(t) = Ah(t) + Bu(t)      (state derivative)
y(t) = Ch(t) + Du(t)       (output)
```

Discretized for sequences (zero-order hold):
```
h_t = Āh_{t-1} + B̄x_t
y_t = Ch_t + Dx_t
```

Gu & Dao's insight: **make Ā, B̄, C̄ functions of x_t** instead of fixed matrices. This "selectivity" is what makes SSMs work for language — the model learns to decide *when* to update state, not just *how*.

## Cross-Thread Connections 🔗

- **[[Linear Algebra]]** — SSMs are matrix recurrences. The state update is just matrix multiplication. The discretization involves matrix exponentials.
- **[[Vaswani et al. — Attention Is All You Need (2017)]]** — the architecture Mamba challenges. Mamba is "Attention Is NOT All You Need."
- **[[Hochreiter-Schmidhuber LSTM 1997]]** — both are recurrent, but LSTMs have fixed gates (σ(Wx + Wh)). Mamba has *input-dependent* state dynamics (A(x_t), B(x_t), C(x_t)).
- **[[DeepSeek — Hybrid Mamba-Transformer MoE]]** — Mamba isn't replacing Transformers everywhere. Hybrid architectures (Mamba for long context, Transformer for precise reasoning) are emerging.
- **[[Hoffmann Chinchilla 2022]]** — scaling laws apply to Mamba too. The paper shows Mamba follows similar compute-optimal scaling curves.
- **[[Kaplan Scaling Laws 2020]]** — Mamba's linear complexity changes the scaling calculus: you can afford much longer sequences for the same compute budget.

## What Came After

- **Mamba-2 (2024)** — improved SSM formulation, better hardware optimization
- **Jamba (AI21 Labs, 2024)** — hybrid Mamba-Transformer, switches between layers
- **DeepSeek-V3 (2024-2025)** — uses Mamba components in its MoE architecture
- **RWKV (2023)** — earlier linear RNN, but Mamba's selective SSMs proved more general
- **Transformers are SSMs (2024)** — theoretical paper showing Transformers can be viewed as a special case of SSMs

## Status

- [x] Read abstract and key sections
- [x] Downloaded PDF
- [ ] Read full paper (math sections on discretization, hardware implementation)
- [ ] Create note on classical control theory origins (Kalman filters, etc.)
- [ ] Create note on Mamba-2 (2024) improvements
- [ ] Explore hybrid Mamba-Transformer architectures (Jamba, DeepSeek)

## Open Questions

- **Generalization beyond language** — Mamba works great on language, audio, genomics. Does it handle vision? Structured data?
- **Training stability** — selective parameters break the nice mathematical properties of fixed SSMs. How does this affect training dynamics?
- **Hardware dependence** — Mamba's speed comes from custom CUDA kernels. Will this be adopted by major frameworks (PyTorch, JAX) or remain niche?
- **Long-term research direction** — Is Mamba a Transformer-killer, or will we see hybrid architectures become the norm?

---

*December 1, 2023 — The first serious post-Transformer paradigm. We're no longer in a one-architecture world.*
