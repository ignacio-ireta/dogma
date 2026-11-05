# Vaswani et al. — Attention Is All You Need (2017)

#paper #milestone #read

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain/Research)
**Published:** NeurIPS 2017 (31st Conference on Neural Information Processing Systems)
**arXiv:** [1706.03762](https://arxiv.org/abs/1706.03762)
**PDF:** `papers/1706.03762.pdf`

## The Moment

This paper broke the dominant paradigm in NLP. RNNs and LSTMs had ruled sequence modeling since the 1980s. This paper abolished them entirely, replacing recurrence with **self-attention** — and everything changed.

> "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."

## Key Ideas

- **Self-Attention** maps a sequence to itself — each token attends to every other token simultaneously, no time step required
- **Scaled Dot-Product Attention:** `Attention(Q, K, V) = softmax(QKᵀ/√d_k)V` — queries, keys, values all derived from the same input
- **Multi-Head Attention:** run attention h times in parallel (different projections), then concatenate — lets model attend to multiple "representation subspaces"
- **Positional Encoding:** since there's no recurrence, position info is injected as sinusoidal vectors added to embeddings
- **Residual connections + Layer Normalization** at every sub-layer — residual pathways from [[../11-CNNs/He ResNets 2015|ResNets]], with per-example stabilization from [[../10-Backpropagation/Ba Kiros Hinton Layer Normalization 2016|Layer Normalization]]

## Architecture

```
Encoder: 6 identical layers
  └── Multi-head self-attention → Add & Norm
  └── Feed-forward (2 linear + ReLU) → Add & Norm

Decoder: 6 identical layers  
  └── Masked multi-head self-attention → Add & Norm
  └── Multi-head cross-attention (over encoder output) → Add & Norm
  └── Feed-forward → Add & Norm
```

## Why It Mattered

- Parallelizable — no sequential dependency between tokens → train dramatically faster
- Long-range dependencies handled in O(1) operations (vs. O(n) for RNN)
- English→German: 28.4 BLEU (new SOTA); English→French: 41.8 BLEU
- Trained in 3.5 days on 8 GPUs — far cheaper than prior SOTA

## Cross-Thread Connections 🔗

- **[[Linear Algebra]]** — The entire attention mechanism is matrix multiplication: QKᵀ, softmax, matmul with V. Pure dot-product geometry.
- **[[Cauchy Steepest Descent → Gradient Descent]]** — optimized with [[Adam (Kingma-Ba 2015)]], itself rooted in the gradient descent lineage from Cauchy (1847)
- **[[../11-CNNs/He ResNets 2015|He et al. 2015]]** — ResNet's skip connections supply the `Add` in `Add & Norm`
- **[[../10-Backpropagation/Ba Kiros Hinton Layer Normalization 2016|Ba, Kiros & Hinton 2016]]** — per-example LayerNorm supplies the `Norm`, avoiding mini-batch-dependent sequence statistics
- **[[../10-Backpropagation/Hochreiter Schmidhuber LSTM 1997|Hochreiter & Schmidhuber 1997]]** — the recurrent sequence architecture this paper displaced
- **[[Bahdanau Attention 2015|Bahdanau et al. 2015]]** — attention first repaired the fixed-vector encoder–decoder bottleneck as an add-on to RNNs.
- **[[Luong Pham Manning Attention 2015|Luong, Pham & Manning 2015]]** — global/local scope and dot/general scoring made attention a reusable architectural layer before the Transformer parallelized it.

## What Came After

- [[BERT 2018 Devlin et al]] — encoder-only Transformer, pre-training via masked language modeling
- GPT series (Radford et al. 2018→) — decoder-only Transformer, autoregressive
- T5, XLNet, RoBERTa, BART, PaLM, GPT-4, Claude — all Transformer descendants
- Vision Transformer (ViT, 2020) — same idea applied to image patches

## References Worth Chasing

- [[Bahdanau Attention 2015|Bahdanau, Cho & Bengio (2015)]] — "Neural Machine Translation by Jointly Learning to Align and Translate" — arXiv:1409.0473 ← *origin of attention in NMT*
- [[Luong Pham Manning Attention 2015|Luong, Pham & Manning (2015)]] — "Effective Approaches to Attention-based Neural Machine Translation" — arXiv:1508.04025 ← *global/local and scoring-function bridge*
- Hochreiter & Schmidhuber (1997) — LSTM — Neural Computation 9(8) ← *already in our queue*
- Kingma & Ba (2015) — "Adam: A method for stochastic optimization" — ICLR 2015 ← *Adam optimizer*
- He et al. (2016) — Deep Residual Learning — arXiv:1512.03385 ← *we have this PDF!*

## Status

- [x] Read abstract, intro, architecture, references
- [ ] Read sections 3-5 (attention details, training, results)
- [ ] Create note for [[Bahdanau Attention 2014]]
- [ ] Create note for [[BERT 2018 Devlin et al]] (we have the PDF: 1810.04805)
