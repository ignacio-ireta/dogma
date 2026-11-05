---
title: "A complete training loop"
minutes: 12
min_viable: "Read The system and Worked example, then solve p01 using Hint 1 if needed."
---

## The system

A training loop is one step repeated over batches, grouped into epochs, with evaluation between
epochs. The contract that makes it reproducible has five parts.

1. **Epochs.** One epoch is one pass over all $n$ training rows. Run $E$ epochs.
2. **Shuffle per epoch.** Draw one permutation of $\{0,\dots,n-1\}$ at the start of each epoch,
   from a seeded generator, and use that order for the epoch.
3. **Mini-batches.** Cut the permutation into consecutive slices of size $b$. The last slice has
   $n \bmod b$ rows when that is nonzero. Keep it; do not drop it.
4. **Modes.** `model.train()` while training (dropout active, batch-norm uses batch statistics).
   `model.eval()` while measuring validation loss, under `torch.no_grad()`.
5. **Logging and checkpoints.** Record the mean training loss weighted by batch size, and the
   validation loss, for every epoch. Keep a copy of the state with the lowest validation loss.

For epoch $e$, with batches $\mathcal{B}_1,\dots,\mathcal{B}_K$ of sizes $b_k$ and batch losses $L_k$
measured before each step:

$$\bar L_e^{\text{train}} = \frac{1}{n}\sum_{k=1}^{K} b_k L_k, \qquad
\bar L_e^{\text{val}} = \text{mean loss over all } m \text{ validation rows, with the model in eval mode}.$$

Here $n$ is the training size and $m$ the validation size. Both are averages over rows, so every
row counts once.

## Worked example

Let $n=10$ and $b=4$. The batches have sizes $(4,4,2)$, so $K=3$. Suppose the batch losses are
$L_1=0.8$, $L_2=0.5$, $L_3=0.2$, with $b_3=2$.

The unweighted mean is $(0.8+0.5+0.2)/3=0.5$. The weighted mean is
$(4\cdot0.8+4\cdot0.5+2\cdot0.2)/10=(3.2+2.0+0.4)/10=0.56$. The weighted value is the average of
the ten per-row losses, each measured at the parameters of its own batch. Only the weighted form
matches that row average.

Checkpointing with validation losses $0.9, 0.6, 0.65$: epoch 2 is the best (strictly lowest). The
final model holds epoch 2's parameters, even though training ran to epoch 3. A tie, such as $0.6, 0.6$,
keeps the earlier epoch.

## Why it works

The loop is an estimate of the training objective, and each piece has a specific job. A random batch gives a noisy
gradient whose expected value is the full-data gradient. That is the setting of stochastic approximation
[library:AI-ML/03-Optimization/Robbins Monro Stochastic Approximation 1951.md], and the mini-batch view is in
[book: Goodfellow, Bengio and Courville, Deep Learning, ch. 8]. The weighted mean reports the loss over every row once, so the curve is comparable across
batch sizes. Validation is measured in eval mode because dropout and batch-norm would otherwise add
noise that is not part of the model's prediction, and under `no_grad` because no gradient is needed.

Reproducibility needs three things at once: the same initial parameters, the same generator seed,
and the same sequence of draws from that generator. The contract spends exactly one draw per epoch
on the permutation and nothing else.

## Signal vs. noise

Signal: the five parts, the weighting, eval mode for validation, and the best-state rule. Noise: the
choice of optimizer class (for plain SGD it is the update line inside each batch) and framework
loaders, which only wrap the same slicing. Learn the manual slice first, then the loader.

## Where it breaks

- **Dropped last batch.** A loop with `range(0, n - b + 1, b)` silently ignores the tail, and small
  datasets lose a sizeable fraction of their rows.
- **Unweighted batch mean.** A short last batch counts as much as a full one. The curve is then
  biased toward that batch.
- **Validation in train mode.** Dropout makes the validation loss noisy, so the checkpoint choice
  becomes random.
- **Global random state.** A shuffle drawn from the global stream changes when any other code draws
  randomness, which breaks seeded comparisons.
- **Checkpointing the last state.** Once validation loss rises (overfitting), the last state is worse
  than the best state you already measured.
- **Reusing the validation set for every decision.** Each epoch's selection makes the reported score
  optimistic. Keep a separate test set for the final number.

## Interview angle

Describe the loop in a sentence: "for each epoch, shuffle with a seeded generator, run mini-batch
steps, measure validation in eval mode without gradients, log the weighted train loss, and keep the
best validation state." Then say what makes it reproducible: seed, draw order, and fixed optimizer and
loss. Follow-ups: "why keep the last partial batch?" (its rows still count for the epoch), and "why
checkpoint the best state instead of the last?" (early stopping; see
`cml/over-and-underfitting-early-stopping`).

## Connections

Requires `dl/one-training-step` (the five lines inside every batch),
`cml/shuffling-splitting-and-batching-data` (the seeded permutation and slices), and
`cml/over-and-underfitting-early-stopping` (the best-validation rule and the patience variant). Next
are `dl/datasets-and-dataloaders`, which replaces the manual slicing with a loader, and the optimizer
topics, which change only the update step.
