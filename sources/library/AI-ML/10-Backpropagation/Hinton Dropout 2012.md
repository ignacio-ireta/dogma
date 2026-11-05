# Hinton et al. — Dropout (2012)

#paper #concept #read

**Authors:** G.E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, R.R. Salakhutdinov
**Institution:** University of Toronto
**Published:** arXiv:1207.0580 (2012) — later expanded into Srivastava et al. JMLR (2014)
**PDF:** `papers/1207.0580.pdf`

## The Problem

Large neural networks trained on small datasets **overfit** — they memorize training data but fail to generalize. The network's feature detectors become co-adapted: they only work *together*, not independently.

## The Solution: Dropout

On each training step, **randomly omit 50% of hidden units**. Each neuron must learn to work in any combinatorial subset of the network — it can't rely on its neighbors.

> "Each neuron learns to detect a feature that is generally helpful for producing the correct answer given the combinatorially large variety of internal contexts in which it must operate."

## Key Ideas

- **Model averaging made cheap:** Training with dropout ≈ training an exponentially large ensemble of networks that share weights
- **Prevents co-adaptation:** Feature detectors become more independent and generalizable
- **At test time:** Use all neurons but scale weights by 0.5 (or equivalently, scale by 1/p at training time — "inverted dropout")
- **Results:** New records for speech recognition and object recognition at time of publication
- **Used in AlexNet (2012):** Krizhevsky-Sutskever-Hinton's ImageNet paper used dropout in fully-connected layers

## Connection to AlexNet

This paper was published *alongside* the AlexNet breakthrough. Same authors (Krizhevsky, Sutskever, Hinton), same year. Dropout was the regularization trick that made deep networks train without catastrophic overfitting.

## Cross-Thread Connections 🔗

- **[[Rumelhart-Hinton-Williams Backpropagation 1986]]** — Hinton is the connecting figure; backprop → deep nets → overfitting problem → dropout solution
- **[[Bayesian perspective]]** — Dropout has a Bayesian interpretation: approximate posterior inference over model parameters (Gal & Ghahramani, 2015)
- **[[Statistical Mechanics]]** — the "energy-based" view (Hinton's training background with Boltzmann Machines) — random masking is analogous to thermal noise
- **[[Krizhevsky-Sutskever-Hinton AlexNet 2012]]** — companion paper; dropout was the key regularizer enabling AlexNet's depth

## Mathematical Note

If each hidden unit is retained with probability p during training, at test time the expected output of each unit is correct only if weights are scaled by p. This is essentially the **bias-variance tradeoff** formalized: increasing variance of individual models (dropout) reduces correlation between errors → lower ensemble variance.

## Historical Significance

Dropout is now ubiquitous. Every major neural network architecture uses it (or its descendants: DropBlock, DropConnect, Stochastic Depth). The [[Vaswani Attention Is All You Need 2017]] Transformer uses `p=0.1` dropout on every sub-layer output.

## References Worth Chasing

- Srivastava et al. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" — JMLR 15(1):1929-1958 (2014) — the full journal version
- Gal & Ghahramani (2016) — "Dropout as a Bayesian Approximation" — Bayesian interpretation

## Status

- [x] Read abstract and key results
- [ ] Read full paper (sections on experiments and analysis)
