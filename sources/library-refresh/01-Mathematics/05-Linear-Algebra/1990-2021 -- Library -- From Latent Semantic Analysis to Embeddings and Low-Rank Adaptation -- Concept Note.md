# From Latent Semantic Analysis to Embeddings and Low-Rank Adaptation

Library synthesis and study commentary

Low rank serves different purposes in language systems: approximating observed counts, explaining a training objective, or restricting a trainable weight update. These uses share matrix mathematics but preserve different quantities. Reading them together closes the gap between classical approximation and contemporary language-model adaptation without treating every embedding as an SVD.

## 1990: make a retrieval representation

Exact word matching misses documents that use different vocabulary for a similar topic. Deerwester, Dumais, Furnas, Landauer and Harshman apply a truncated SVD to a term–document association matrix:

\[
X\approx U_k\Sigma_kV_k^T.
\]

Latent semantic indexing represents terms, documents and queries using the reduced structure. Similarity can then reflect patterns of shared association rather than literal overlap alone. A convenient balanced factorization is \(T=U_k\Sigma_k^{1/2}\), \(D=V_k\Sigma_k^{1/2}\), so \(X_k=TD^T\); this is a modern algebraic restatement, not a claim that every historical scoring convention used these exact coordinates.

The best-approximation theorem optimizes matrix reconstruction under a specified norm. It does not prove semantic relevance, eliminate polysemy or choose the best task dimension. Weighting the counts and selecting \(k\) are substantive modeling decisions.

## 2014: expose an implicit matrix

Levy and Goldberg analyze skip-gram with negative sampling (SGNS). Under unigram negative-context sampling and enough dimensions to optimize each word–context dot product independently, its stationary optimum satisfies

\[
w^Tc=\operatorname{PMI}(w,c)-\log b,
\]

where \(b\) is the number of negative samples and

\[
\operatorname{PMI}(w,c)=\log{P(w,c)\over P(w)P(c)}.
\]

Thus an embedding objective can be understood through association statistics. This is an analytical relationship, not an instruction that SGNS literally computes a truncated SVD. With limited dimension, the dot products are coupled through low-rank factors; SGNS uses a weighted logistic objective, while ordinary SVD optimizes a different reconstruction loss. Changing the negative-sampling distribution changes the target expression. A zero co-occurrence count also needs care because the idealized log ratio can diverge.

This distinction explains why matrix factorization can illuminate neural representations without making every training procedure equivalent. The paper compares explicit shifted-PMI representations and SVD with SGNS empirically rather than deriving universal equivalence of downstream performance.

## 2021: constrain the update, retain the base

Hu and colleagues address the storage and optimization cost of adapting a large pretrained model. LoRA freezes \(W_0\in\mathbb R^{m\times n}\) and trains

\[
W=W_0+{\alpha\over r}BA,\quad
B\in\mathbb R^{m\times r},\quad A\in\mathbb R^{r\times n}.
\]

The trainable update has rank at most \(r\) and uses \(r(m+n)\) parameters. The full pretrained matrix need not have low rank. The factors are learned through the task objective, as described in [Matrix Differentials for Machine Learning](2012%20--%20Petersen%20and%20Pedersen%20--%20Matrix%20Differentials%20for%20Machine%20Learning%20--%20Study%20Note.md), rather than obtained by an SVD of \(W_0\). The paper motivates this restriction using prior intrinsic-dimension work and empirical adaptation behavior.

## What is preserved changes the interpretation

For approximation error \(E=W-\tilde W\) and input second moment \(S=\mathbb E[xx^T]\),

\[
\mathbb E\|Ex\|_2^2=\operatorname{tr}(ESE^T).
\]

This elementary calculation shows why small unweighted Frobenius error need not match small error on actual activations. Task accuracy introduces still more structure. The relevant test can therefore be retrieval quality, objective value, activation error or adaptation performance; a single rank number does not substitute for those checks.

The sequence above is a conceptual comparison, not a verified historical influence chain from LSA through SGNS to LoRA. [Randomized SVD as a Computational Bridge](2011%20--%20Halko%2C%20Martinsson%20and%20Tropp%20--%20Randomized%20SVD%20as%20a%20Computational%20Bridge%20--%20Study%20Note.md) supplies an efficient approximation tool when an explicit matrix is available; it is not the training mechanism of all three methods.

## Primary reading and locators

- Deerwester et al., Indexing by Latent Semantic Analysis (1990): pp. 397–398, “Technical Details” and Figure 2, explain the SVD model. [Wiley record](https://doi.org/10.1002/%28SICI%291097-4571%28199009%2941%3A6%3C391%3A%3AAID-ASI1%3E3.0.CO%3B2-9) verifies the full five-author credit and journal version.
- Levy and Goldberg, Neural Word Embedding as Implicit Matrix Factorization (2014): §3.1 derives the implicit matrix; subsequent sections compare explicit representations and experiments. [Official proceedings](https://papers.neurips.cc/paper_files/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html).
- Hu et al., LoRA, arXiv v2 (2021): §4.1 defines the update; §7 investigates rank. [arXiv version history](https://arxiv.org/abs/2106.09685) identifies v2 as 16 October 2021. The copy used is that preprint, not a claim to be the later ICLR 2022 final version.
