# A Map from Mathematical Foundations to Machine Learning

**Source:** Marc Peter Deisenroth, A. Aldo Faisal and Cheng Soon Ong, *Mathematics for Machine Learning* (Cambridge University Press, 2020). The author draft is dated 2024-01-15.
**Full text (not imported):** Complete author PDF. **Official site:** https://mml-book.github.io/
**Reading status:** Contents and selected foundation/application sections inspected; this is a navigation and synthesis note, not a claim to have completed the book.

## The problem this closes

A historical lineage explains where an idea came from. It does not automatically tell a learner which operations must be understood to derive a model. Knowing that Gauss studied least squares, Fisher developed likelihood, and matrices admit decompositions still leaves a gap: how do those ingredients become one trainable statistical system?

This textbook provides a useful operational spine. Chapters 2–7 develop linear algebra, geometry, decompositions, vector calculus, probability, and optimization. Chapters 8–12 reuse them in empirical risk, regression, PCA, mixture models, and support-vector machines. This is a pedagogical dependency map, not evidence that those subjects developed in this chronological order.

## One model, several mathematical views

For a design matrix X and responses y, least squares minimizes L(w)=||Xw-y||²/2. Geometry interprets Xw as a point in the column space of X; the residual at a minimizer is orthogonal to that space. Calculus gives ∇L=Xᵀ(Xw-y). Linear algebra characterizes uniqueness: a full-column-rank X makes XᵀX positive definite. Numerical analysis then asks how to find the answer accurately, usually favoring QR or SVD over explicitly forming an inverse.

Probability supplies an additional interpretation. Under independent Gaussian errors with common fixed variance, minimizing squared error also maximizes likelihood. That equivalence depends on the error model; it does not make squared error the right likelihood for arbitrary outcomes. Adding a Gaussian prior gives a quadratic penalty and connects Bayesian estimation to ridge regularization.

The same reuse occurs in PCA. A low-dimensional reconstruction problem becomes an orthogonal projection, then an eigenvalue/SVD problem. In mixture models the extra unknown is a latent assignment; EM handles uncertain assignments by alternating posterior responsibilities and parameter updates rather than treating unobserved labels as known; direct observed-likelihood optimization is another option.

## Read through the library

1. Start with chapters 2–4 and the linear algebra index. Derive projection and rank before treating PCA as a library call.
2. Use chapter 5 with Leibniz and the chain rule and the calculus index. Track vector/matrix dimensions when composing derivatives.
3. Pair chapter 6 with probability and statistics. Separate a model distribution, an estimator, and a predictive uncertainty statement.
4. Use chapter 7 and the optimization index before the application chapters. Distinguish a stationary point, a convex optimum, and numerical convergence.

## Why it matters for AI

The bridge to deep learning is the repeated composition of these operations: parameterized maps produce predictions, a probabilistic or decision-theoretic loss scores them, derivatives assign credit, and an optimizer changes parameters. Later architecture papers alter the maps and computational constraints, but their assumptions remain easier to examine when this derivation can be done for a small model first.

**Verified source locators:** §§3.8,4.5–4.6,5.6–5.8,6.3,7.1–7.3,9.4,10.2–10.4 and11.3; printed page numbers differ from PDF page numbers.
