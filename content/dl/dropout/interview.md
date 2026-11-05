# Interview notes — Dropout

## Q1. Explain inverted dropout forward and backward.

**60-second answer.** During training I sample independent keep indicators and divide survivors by the keep probability, preserving each activation in expectation. The backward pass multiplies upstream gradients by that exact same scaled mask. During evaluation both maps are identities. I treat drop probability, mode, and random-generator ownership as explicit API choices.

**Follow-ups.** Why no eval rescaling? Scaling already occurred in training. Why not draw a new backward mask? That would differentiate another realized function.

## Q2. How do you test dropout when its output is random?

**60-second answer.** I make randomness explicit with a local seed and check the exact mask convention. For gradients I hold that realization fixed across central differences. I separately check evaluation, zero drop probability, seed reproducibility, and that global randomness and input arrays stay unchanged. I avoid expecting an exact retained count in a small batch.

**Follow-ups.** Should every training step reuse the same seed? Usually use a persistent advancing generator instead. Does expectation preservation imply exact ensemble predictions? No; downstream nonlinearities prevent that conclusion.

## Common mistakes

- Confusing drop probability with keep probability or scaling evaluation again.
- Drawing a fresh mask for backward instead of caching the forward multiplier.
- Reusing a fixed seed on every training step when fresh masks are intended.

Source: [paper: Srivastava et al. 2014, Dropout: A Simple Way to Prevent Neural Networks from Overfitting (JMLR 15, 1929-1958)]. See lesson.md for the derivation and explicit conventions.
