# Socher et al. — Zero-Shot Learning Through Cross-Modal Transfer (2013)

#paper #concept #read

**Authors:** Richard Socher, Milind Ganjoo, Hamsa Sridhar, Osbert Bastani, Christopher D. Manning, Andrew Y. Ng
**Institution:** Stanford University
**Published:** NeurIPS 2013 (arXiv:1301.3666, submitted Jan 2013)
**PDF:** `papers/1301.3666.pdf`

## Note on Acquisition

This paper was downloaded in place of word2vec (arXiv:1301.3781 — a common mix-up). However it's worth keeping — it's directly relevant and the connection to word vectors is strong. Word2vec is now queued for download.

## The Problem

Computer vision models require labeled training data for every category. What if you encounter a category with *no* training examples?

Humans can identify a "Segway" from reading a description, even without seeing one. Can a neural network do the same?

## The Solution

**Cross-modal transfer:** Train a model that maps images into a **word vector space**, then classify unseen categories by proximity in that shared semantic space.

1. Learn word vectors from large text corpus (distributional semantics — what words appear near each other)
2. Train a neural network to map images into the *same* semantic space
3. For unseen categories: compute word vector for the category name → find nearest image representation

The key insight: **word vectors encode semantic relationships**, so visual and linguistic representations can share structure. A dog image should be close to the word "dog" in the joint space — and close to "wolf" and "cat" by semantic proximity.

## Key Ideas

- **Zero-shot learning** — recognize categories unseen at training time
- **Distributional word vectors** as the semantic bridge — meaning from text statistics, grounded by vision
- **Outlier detection** — model knows when it's being asked about something unseen vs. known
- **Works on seen AND unseen classes simultaneously** — unlike prior zero-shot models which only distinguish between unseen classes

## Cross-Thread Connections 🔗

- **word2vec (Mikolov 2013)** — this paper relies on distributed word representations. The semantic arithmetic that makes "king - man + woman = queen" is what makes this zero-shot transfer possible
- **[[BERT 2018]]** — BERT's contextual representations are a deeper version of the same idea: language encodes semantic structure that transfers across tasks
- **CLIP (Radford 2021)** — the natural endpoint of this line of research: contrastive learning that directly aligns images and text in a shared embedding space, at massive scale
- **[[Goodfellow GANs 2014]]** — same year, same NeurIPS venue, different approach to generative/representational learning

## The Bigger Picture

This paper is part of the "distributed representations generalize" movement of 2012-2014. The hypothesis: if you learn rich enough representations, they transfer. Language knows about vision. Vision can learn from language.

This hypothesis was vindicated at scale with CLIP (2021) and GPT-4V (2023): one model can understand images *and* text because the representations align in a shared geometric space.

## References Worth Chasing

- Mikolov et al. (2013) — word2vec — **arXiv:1301.3781** ← *queued for download (correct ID)*
- Farhadi et al. (2009) — "Describing Objects by their Attributes" — *earlier semantic attribute approach*
- Radford et al. (2021) — CLIP — arXiv:2103.00020 — *the modern version of this idea, at scale*
- Lampert et al. (2014) — "Attribute-Based Classification for Zero-Shot Visual Object Categorization" — TPAMI

## Status

- [x] Read abstract, introduction, model description
- [x] Note created (acquired by accident, worth keeping — strong connection to word vectors)
- [ ] Read section 4 (experiments on ImageNet zero-shot recognition)
- [x] Connect to CLIP when that note is created — *done: [[Radford CLIP 2021]]*
