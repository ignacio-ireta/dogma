# Mikolov et al. — Efficient Estimation of Word Representations in Vector Space (2013)

#paper #concept #read #milestone

**Authors:** Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
**Institution:** Google Inc.
**Published:** arXiv:1301.3781 (submitted Jan 2013, updated Sep 2013)
**PDF:** `papers/1301.3781.pdf`

## The Problem

Neural network language models (NNLMs) learn rich word representations as a side effect of predicting text. But they're **expensive**. A traditional NNLM with a hidden layer applied to a 10-billion-word corpus would take weeks. Could you get the same quality word vectors *without* the expensive parts?

## The Architecture

Two radically simplified models, both log-linear (no hidden layers, no non-linearity):

**CBOW (Continuous Bag of Words):**
- Predict the center word from the surrounding context
- Average the surrounding word vectors, predict center
- Fast, good at syntactic tasks

**Skip-gram:**
- Predict the surrounding context given the center word
- One input → multiple outputs
- Better at semantic relationships, works well on smaller data

Both models strip away the non-linear hidden layer that makes NNLMs expensive. The result: train on 1.6 billion words in *less than a day* on a single machine.

## The Magic: Vector Arithmetic

The paper's most striking result — that led to a decade of research:

```
vec("King") - vec("Man") + vec("Woman") ≈ vec("Queen")
vec("Paris") - vec("France") + vec("Italy") ≈ vec("Rome")
vec("bigger") - vec("big") + vec("small") ≈ vec("smaller")
```

Word meaning has **linear algebraic structure**. Semantic and syntactic relationships — analogy, gender, plural, tense — are encoded as *directions* in the vector space.

This wasn't just qualitatively interesting. They built a benchmark of 8,869 semantic + 10,675 syntactic analogy questions and measured it precisely. Skip-gram on 6 billion words: 65.6% accuracy.

## Why This Mattered

**1. Scale became accessible.** Previous approaches (NNLM, RNNLM) were too slow. Word2Vec made large-scale word vectors cheap. This enabled pre-trained embeddings to become infrastructure — you could download 300-dimension vectors for millions of words and plug them into any NLP system.

**2. Distributed representations generalize.** The arithmetic result showed that meaning isn't stored locally in any neuron — it's distributed across a high-dimensional space, and *the geometry of that space encodes relationships*. This is the founding insight of representation learning.

**3. Transfer learning became a concept.** Before word2vec, you trained an NLP model from scratch on each task. After word2vec, you could initialize with pre-trained vectors. This was the seed of the idea that would become BERT, GPT, and the modern pretrain-then-finetune paradigm.

## Cross-Thread Connections 🔗

- **[[Socher Zero-Shot Learning 2013]]** — same year; Socher used distributional word vectors as the semantic bridge for cross-modal transfer. Directly enabled by the same insight that word vectors encode semantic structure
- **[[Radford DCGAN 2016]]** — DCGAN found the same arithmetic in *image* latent space (smiling woman - neutral woman + neutral man = smiling man). The "linear algebra in learned spaces" principle transfers across modalities
- **[[Bahdanau Attention 2015]]** — word2vec embeddings were the default input layer for seq2seq models before contextual representations existed
- **[[Vaswani Attention Is All You Need 2017]]** — static word2vec embeddings → contextual attention → eventually trained end-to-end
- **[[Devlin BERT 2018]]** — BERT replaced static word vectors with contextual embeddings. Word2vec is the first domino
- **CLIP (Radford 2021)** — the natural endpoint: language and vision share a semantic space, now at 400 million image-text pairs. The "distributed representations generalize across modalities" hypothesis, vindicated at scale

## The Representation Learning Thread

```
word2vec 2013 (words have algebra) →
GloVe 2014 (global matrix factorization) →
FastText 2017 (subword units) →
ELMo 2018 (contextual, from LSTMs) →
BERT 2018 (contextual, bidirectional, Transformer) →
GPT 2018/2019 (contextual, generative) →
CLIP 2021 (cross-modal, contrastive)
```

Each step: richer context, more modalities, bigger scale.

## Historical Note

Mikolov was 26 when he published this paper. It became one of the most cited papers in NLP history. The Google Brain connection (Jeffrey Dean, Greg Corrado) gave it the computational infrastructure to demonstrate scaling properties. Mikolov later moved to Facebook AI Research, then founded a startup. The word2vec ideas remain foundational.

## References Worth Chasing

- GloVe (Pennington, Socher, Manning 2014) — *global vs. local context for word vectors*
- ELMo (Peters et al. 2018, arXiv:1802.05365) — *contextual word vectors from LSTMs*
- Collobert & Weston (2008) — "A unified architecture for NLP" — *pre-word2vec neural word embeddings*
- Bengio et al. (2003) — "A Neural Probabilistic Language Model" — *the NNLM that word2vec simplified*

## Status

- [x] Read abstract, architecture description, key results
- [x] Note created (identified via arXiv:1301.3781 — resolves task in Cho note)
- [ ] Read section 4 (evaluation results table — compare CBOW vs Skip-gram across tasks)
- [ ] Trace how GloVe (2014) addressed word2vec's local context limitation
