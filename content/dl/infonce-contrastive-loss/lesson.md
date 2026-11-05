---
title: InfoNCE contrastive loss
minutes: 12
min_viable: "Read The system and Worked example, then implement p01 with Hint 1 open."
---

## The system

We have $N$ positive pairs of embeddings $(a_i, b_i)$, $i = 1, \dots, N$, each a vector in $\mathbb{R}^D$. For example, $a_i$ is an image and $b_i$ its caption. Pairs that belong together are positives. Every other pairing in the batch is a negative. Negatives are free: each anchor $a_i$ uses the $N - 1$ other $b_j$ in its batch.

The steps:

1. **Normalise.** $\hat a_i = a_i / \lVert a_i \rVert_2$ and $\hat b_i = b_i / \lVert b_i \rVert_2$. After this, a dot product is a cosine similarity.
2. **Similarity matrix with temperature $\tau > 0$.** $S_{ij} = \hat a_i^{\top} \hat b_j / \tau$, shape $N \times N$. The diagonal $S_{ii}$ holds the positives.
3. **Cross-entropy in both directions.** Row $i$ of $S$ is a softmax over the $N$ candidates $b_j$, and the correct label is $i$. Column $j$ is a softmax over the candidates $a_i$, with label $j$.
4. **Average.** The symmetric loss is

$$\mathcal{L} = \frac{1}{2N} \sum_{i=1}^{N} \Big[ \big(\operatorname{LSE}_j S_{ij} - S_{ii}\big) + \big(\operatorname{LSE}_j S_{ji} - S_{ii}\big) \Big], \qquad \operatorname{LSE}_j x_j = \log \sum_{j} e^{x_j}.$$

It is a cross-entropy, so the stable log-sum-exp rule applies.

## Worked example

Take $N = 2$ and $D = 2$, with $a_1 = b_1 = (1, 0)$ and $a_2 = b_2 = (0, 1)$. The normalised vectors are unchanged. The similarity matrix is $\hat a^\top \hat b = I$.

- **$\tau = 1$.** $S = I$. Row 1 is $(1, 0)$, and its positive is its own column. Its loss is $\log(e^1 + e^0) - 1 = \log(1 + e^{-1}) = 0.3133$. Row 2 gives the same value, and the columns give the same value too. So $\mathcal{L} = 0.3133$.
- **$\tau = 0.1$.** $S = 10 I$. Each row loss is $\log(1 + e^{-10}) = 0.0000454$, so $\mathcal{L} \approx 4.5 \times 10^{-5}$. A smaller temperature sharpens the softmax.
- **Scale check.** Replacing $a_2$ by $5 a_2$ changes nothing, because normalisation removes the length.
- **Chance level.** If every $a_i$ is one vector and every $b_j$ is another, every entry of $S$ is equal, every softmax is uniform over $N$, and $\mathcal{L} = \log N$ exactly. A loss above $\log N$ is worse than guessing.

## Why it works

Each anchor must pick its true partner out of $N$ candidates, and the loss is the negative log of the probability given to that partner. Minimizing it raises the positive similarity relative to the $N - 1$ negatives. The name InfoNCE comes from van den Oord et al. (2018); that paper is not in this repository's corpus, so this lesson makes no claim from it beyond the name.

The symmetric form follows Radford et al. (2021). CLIP "optimize[s] a symmetric cross entropy loss over these similarity scores", using the $N$ real pairs as positives and the $N^2 - N$ incorrect pairings as negatives (p. 4). Its pseudocode computes the two cross-entropies over the rows and the columns and returns their average (p. 5). Averaging gives the two retrieval directions equal weight.

The temperature sets the logit scale. CLIP learns it as a log-parameterised multiplicative scalar (p. 4; the pseudocode uses $\exp(t)$ on p. 5). It clips the scale so the logits are multiplied by at most $100$, which is $\tau \ge 0.01$, and starts it at $0.07$ (p. 5).

## Signal vs. noise

Signal: normalisation inside the loss, the diagonal labels, the two directions, and the role of $\tau$. Noise: whether $\tau$ is fixed or learned; this exercise fixes it.

## Where it breaks

- **No normalisation.** The loss then depends on embedding norms, and large vectors dominate the softmax.
- **Small temperature without a stable log-sum-exp.** Logits have size about $1/\tau$, so $\tau = 0.001$ gives logits near $1000$, which overflow a naive $e^{x}$ and make the loss `inf` or `NaN`.
- **Large temperature.** The softmax is flat, the loss stays near $\log N$, and the gradients are weak. Check the loss against $\log N$ early in training.
- **Comparing losses across batch sizes.** The chance level is $\log N$, so a loss of $2.0$ at $N = 8$ and $N = 1024$ means different things. Report the batch size with the loss.
- **False negatives.** Two in-batch items that are both correct for an anchor (for example two captions for one image) are scored as a wrong pair. The loss penalises a correct match.

## Interview angle

Sixty-second answer: a batch of N pairs gives each anchor N-1 free negatives. I normalise both sides, build the N by N cosine matrix divided by the temperature, and take cross-entropy along rows and columns with the diagonal as labels, then average the two. The temperature controls how sharp the softmax is. Chance is log N. A larger batch means more negatives per anchor, so the task is harder, but the loss values are not comparable across batch sizes unless I report the batch size. Likely follow-up: why is it symmetric? Retrieval runs in both directions, and a one-way loss can leave one of them poor. Second follow-up: what does the temperature do at the extremes? A very small one makes the softmax nearly one-hot, so hard negatives dominate. A very large one makes it flat, and the loss stays near log N.

## Connections

- Prerequisites: [Cosine similarity](../../cml/cosine-similarity/), [Cross-entropy](../../cml/cross-entropy/), [Softmax](../../cml/softmax/).
- Next: contrastive pretraining of paired encoders, then retrieval evaluation.
