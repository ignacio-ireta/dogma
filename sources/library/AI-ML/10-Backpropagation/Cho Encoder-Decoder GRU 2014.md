# Cho et al. — Learning Phrase Representations using RNN Encoder-Decoder (2014)

#paper #milestone #concept #read

**Authors:** Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio
**Institution:** Université de Montréal (+ Bahdanau at Jacobs University)
**Published:** EMNLP 2014 (arXiv:1406.1078, submitted June 2014)
**PDF:** `papers/1406.1078.pdf`

## Two Things in One Paper

This paper does two important things simultaneously:

### 1. The Encoder-Decoder Architecture
Proposes the **RNN Encoder-Decoder**: one RNN reads an input sequence → compresses it into a fixed-length vector → second RNN generates the output sequence from that vector.

This became the dominant architecture for sequence-to-sequence tasks (translation, summarization, dialogue). It's the architecture that [[Bahdanau Attention 2015]] was invented to *fix*.

### 2. The GRU (Gated Recurrent Unit)
Introduces the **Gated Recurrent Unit** — a simpler alternative to [[Hochreiter-Schmidhuber LSTM 1997]]:

```
Update gate:  zₜ = σ(Wz·xₜ + Uz·hₜ₋₁)
Reset gate:   rₜ = σ(Wr·xₜ + Ur·hₜ₋₁)
New memory:   h̃ₜ = tanh(W·xₜ + U·(rₜ⊙hₜ₋₁))
Hidden state: hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ
```

Two gates instead of LSTM's three. Less parameters, often comparable performance. The update gate decides how much of the past to keep; the reset gate decides how much past to forget when computing new memory.

## Key Ideas

- **Fixed-length bottleneck** — the encoder compresses the entire sentence into one vector. Works for short sentences; degrades badly for long ones. This bottleneck directly motivated [[Bahdanau Attention 2015]]
- **Joint training** — encoder and decoder trained end-to-end to maximize P(target | source) — no separate alignment model
- **Phrase representation** — not just words; the model learns semantic representations of multi-word phrases
- **Results** — combined with phrase-based MT system, improved BLEU scores on English-French translation

## Cross-Thread Connections 🔗

- **[[Hochreiter-Schmidhuber LSTM 1997]]** — GRU is a simplification of LSTM. Both solve vanishing gradients with gating mechanisms
- **[[Bahdanau Attention 2015]]** — direct child: Bahdanau is literally a co-author here. This paper identified the bottleneck; the next paper solved it
- **[[Vaswani Attention Is All You Need 2017]]** — two steps removed: encoder-decoder → attention-augmented encoder-decoder → pure attention (no recurrence)
- **[[Rumelhart-Hinton-Williams Backpropagation 1986]]** — GRU gates are trained via backpropagation through time (BPTT); the vanishing gradient problem they solve was caused by backprop's multiplicative chain

## The Bottleneck Insight

The paper says it explicitly: compressing a sentence into a fixed-length vector "may make it difficult for the neural network to cope with long sentences." This observation planted the seed for attention. Bahdanau (a co-author) took this observation and built the solution a few months later.

## Note on Word Vectors

The paper also shows that the encoder's fixed-length representations cluster semantically meaningful phrases together — early evidence that RNNs learn distributed semantic representations. This connects to word2vec (Mikolov 2013) which does the same thing purely from word co-occurrence.

## References Worth Chasing

- Sutskever, Vinyals, Le (2014) — "Sequence to Sequence Learning with Neural Networks" — arXiv:1409.3215 — *parallel development, longer sequences*
- Hochreiter & Schmidhuber (1997) — [[Hochreiter-Schmidhuber LSTM 1997]] — *what GRU simplified*
- Bahdanau et al. (2015) — [[Bahdanau Attention 2015]] — *the direct sequel*
- Mikolov et al. (2013) — word2vec — arXiv:1301.3781 ← *correct ID! (we downloaded wrong paper)*

## Status

- [x] Read abstract, architecture, GRU formulation
- [x] Note created
- [ ] Read section 4 (experiments — phrase clustering visualizations)
- [x] Add correct word2vec arXiv ID to download queue: 1301.3781 — *confirmed, note created: [[Mikolov Word2Vec 2013]]*
