# Coding, Bayesian Inference and Learning as Compression

**Source:** David J. C. MacKay, *Information Theory, Inference, and Learning Algorithms* (Cambridge University Press, 2003). Full text (not imported): version 7.2, fourth printing, 2005-03-28.
**Full text (not imported):** Complete author-hosted PDF. **Author site:** https://www.inference.org.uk/mackay/itila/
**Reading status:** Source identity, contents, and selected inference sections inspected; substantial exercises and full-book reading remain.

## The gap after entropy and divergence

Shannon measures uncertainty in messages. Kullback and Leibler quantify the cost of a mismatched distribution. The next useful question is operational: how do coding choices, uncertain parameters, and learned models become aspects of the same prediction problem?

MacKay's book supplies a unified teaching framework linking communication, Bayesian inference, computational approximations, and neural learning. This is a modern synthesis of several branches, rather than a claim that one historical discovery mechanically forced all the others.

## Prediction has a coding interpretation

Suppose a discrete event x is assigned probability q(x)>0. Its ideal information length is -log₂q(x) bits. Real prefix codes use integer lengths and obey coding constraints; block or arithmetic coding can approach expected information lengths under appropriate assumptions.

If observations come from p, the expected ideal length under q is H(p,q)=H(p)+D_KL(p||q). Thus a mismatched predictor pays extra expected bits. Maximum likelihood on independent data minimizes the empirical sum of negative log probabilities. That connection makes log loss meaningful without proving that minimizing training log loss ensures generalization.

For a simple example, let a binary source have p(1)=0.9. Predicting q(1)=0.5 spends one ideal bit per event. Using the true probability spends H₂(0.9)≈0.469 bits on average. The difference, about 0.531 bits, is the KL penalty for the uniform predictor. Individual outcomes can still be surprising; the statement is about expectation.

## A distribution over explanations

Bayesian inference replaces a single fitted parameter with a posterior proportional to likelihood times prior. The evidence p(D)=∫p(D|θ)p(θ)dθ normalizes that posterior and averages fit over parameter uncertainty. It cannot generally be replaced with the maximum likelihood value: a model with one narrow high peak and a model that fits well over a broad region need not receive the same evidence.

Exact sums and integrals become impractical when latent states or parameters are numerous. The book then develops tools such as message passing, Monte Carlo methods, and variational approximation. Their error must be separated from sampling error and model misspecification. The [graphical-model and variational inference note](../06-Probability-and-Statistics/2008%20--%20Wainwright%20and%20Jordan%20--%20Graphical%20Models%20and%20Variational%20Inference%20--%20Study%20Note.md) develops that next layer.

## What this enables in the AI thread

For language models, next-token cross-entropy is a sequential predictive coding objective, although a capable language model is more than a practical compressor. For latent-variable models, the difficulty of marginalization motivates approximate inference; the VAE note later combines a variational objective with amortized neural inference. These are conceptual links, not claims that the later authors relied on this particular textbook.

**Read next:** chapters 1–3 for entropy and inference, chapter 20 for clustering, chapters 25–26 for exact marginalization, chapters 29–33 for computational inference, and chapter 41 for learning as inference. Keep the original information-theory papers alongside the synthesis.
