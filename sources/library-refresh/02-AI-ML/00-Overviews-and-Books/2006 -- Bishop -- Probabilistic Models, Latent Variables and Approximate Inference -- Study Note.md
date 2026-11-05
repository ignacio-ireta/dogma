# Probabilistic Models, Latent Variables and Approximate Inference

**Source:** Christopher M. Bishop, *Pattern Recognition and Machine Learning* (Springer, 2006).
**Full text (not imported):** Complete Microsoft Research-hosted PDF. **Author resource page:** https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/
**Reading status:** Title/copyright/contents and selected graphical-model, mixture, and variational sections inspected. This is a bridge note, not a completed-book reading log.

## The problem between classical statistics and neural models

A fitted decision boundary hides several different questions. What process is modeled? What parameters must be estimated? Are there latent variables? Which uncertainty is represented? Can the required probability calculation actually be performed?

This book organizes those questions into a reusable modeling workflow. It builds a bridge from likelihood and Fisher and linear models to graphical models, latent variables, approximate inference, and neural networks. Its role here is a mathematical synthesis, rather than a primary historical source for each technique it teaches.

## Three computations that should stay distinct

Parameter estimation chooses θ from observed data. Latent-variable inference determines a distribution over unobserved z for given θ. Prediction integrates or conditions on the relevant unknowns to make a claim about new data. They may appear in one training loop, but solving one does not automatically solve the others.

For a Gaussian mixture, p(x|θ)=Σ_k π_k N(x|μ_k,Σ_k). A hidden assignment k explains which component generated x. The log of the sum prevents ordinary separate-component fitting. EM computes assignment probabilities using the current model, then updates parameters using those soft responsibilities. The [original EM paper and note](../../01-Mathematics/06-Probability-and-Statistics/1977%20--%20Dempster%2C%20Laird%20and%20Rubin%20--%20EM%20and%20Latent%20Variable%20Likelihood%20--%20Study%20Note.md) gives that algorithm its own primary-source treatment.

For arbitrary q(z), define L(q,θ)=E_q[log p_θ(x,z)]-E_q[log q(z)]. Under the usual support and finiteness conditions,

log p_θ(x)=L(q,θ)+D_KL(q(z)||p_θ(z|x)).

This identity separates the evidence lower bound from the true log evidence. Equality holds when q is the exact posterior. Restricting q to a tractable family creates an approximation gap; optimizing a bound does not imply that this gap vanishes. Exact EM and variational EM should therefore not be treated as identical algorithms.

## A small decision example

Suppose two explanations give posterior probabilities 0.7 and 0.3. A hard assignment discards the second explanation. Averaging predictions over both preserves that uncertainty. The distinction matters when their outputs differ sharply: a confident single explanation can look decisive while suppressing a real alternative. It is still necessary to test whether either explanation is plausible; posterior uncertainty inside a misspecified model is not a guarantee of real-world calibration.

## What comes after

Chapters 2–4 connect distributions and linear prediction. Chapter 8 organizes conditional independence; chapter 9 develops mixtures and EM; chapters 10–11 address variational and sampling methods. Chapter 12 connects continuous latent variables to dimensionality reduction. The probability/statistics index supplies original papers and modern uncertainty extensions.

The VAE note later makes neural networks generate parameters of an approximate posterior and uses differentiable stochastic estimation. Transformers change the representation and sequence machinery; they do not eliminate the need to distinguish modeling assumptions, fitting objectives, predictive uncertainty, and inference approximations.
