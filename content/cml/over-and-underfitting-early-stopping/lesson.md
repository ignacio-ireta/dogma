---
title: Over- and underfitting, early stopping
minutes: 12
min_viable: "Read The system and Where it breaks, then do p01."
---

## The system

Read two curves: training error and validation error, both against model capacity or training time.

| Signature | Training error | Validation error | Fix |
|---|---|---|---|
| Underfitting | high | high, small gap | more capacity or features; train longer |
| Good fit | low | low, small gap | keep it; confirm on a test set |
| Overfitting | low | high, large gap | regularise, add data, or stop earlier |

These are diagnostic clues, not a proof: compare errors against the achievable noise level,
check optimization, and check whether validation comes from a different distribution.

Capacity is the size of the model class: a polynomial degree, a tree depth, or the number of weights. Regularisation
shrinks the effective capacity without changing the class. Early stopping does the same through time: the number of
epochs is a capacity dial.

**Early-stopping rule.** Keep the best validation loss so far, and the epochs since it last improved. Stop when the wait
reaches the patience. An improvement must beat the best by more than `min_delta`.

## Worked example

Averages over 30 seeds (`default_rng(seed)`, seeds 0 to 29) of least-squares polynomial fits (`np.polyfit`) to
$y = \sin(2\pi x) + \varepsilon$, with $x \sim U(0, 1)$ and $\varepsilon \sim \mathcal N(0, 0.3^2)$ on both training and
validation labels. There are 30 training rows and 200 validation rows, and validation MSE uses the noisy labels. This
is one dataset family, not a general law:

| Degree | Train MSE | Validation MSE | Signature |
|---|---|---|---|
| 1 | 0.28 | 0.31 | underfitting |
| 3 | 0.08 | 0.11 | good fit |
| 5 | 0.07 | 0.16 | starting to overfit |
| 9 | 0.06 | 1.55 | overfitting |

Early stopping on losses $1.0, 0.8, 0.7, 0.72, 0.71, 0.75, 0.74$ with patience 3: epoch 2 is the best ($0.7$). Epochs 3,
4 and 5 fail to improve, so the wait reaches 3 at epoch 5. The result is $(\text{best}, \text{stop}) = (2, 5)$.

## Why it works

Overfitting means fitting the noise in the training rows. Its signature is a validation loss that rises after a
minimum while the training loss keeps falling. A checkpoint near a useful validation minimum can limit noise fitting, although a noisy validation
minimum does not identify the true population optimum. Goodfellow et al. (2016, chapter 7) treat early stopping as a regulariser for this
reason.

The validation set does two jobs in this rule: it sets the stop, and it reports the score. Those jobs conflict. Each
choice made on it is a selection, and a selected performance score tends to be optimistic (a selected loss tends to be too low).

## Signal vs. noise

- **Signal:** the shape of the validation curve, and the gap to training. A flat validation curve says more epochs
  will not help. A rising one says stop.
- **Signal:** the validation set's size and representativeness. A small or unlike validation set makes the stop noise.
- **Noise:** the exact tenth of a unit in one epoch's loss. The `min_delta` setting absorbs it.

## Where it breaks

- **Validation reuse.** Searching many epochs or configurations selects favorable validation noise.
  Evaluate the selected procedure on an untouched test set; the validation minimum is not the final estimate.
- **A small mean gain can hide a low win rate.** In a seeded check (raw powers up to degree 20, 20 training rows,
  200 noisy validation rows, 1000 noise-free test points, full-batch gradient descent from zero weights, learning rate
  0.05, at most 4000 epochs, patience 100, seeds 0 to 19), the checkpoint's mean test MSE was 0.114 and the final
  weights' was 0.116. The checkpoint won on only 6 of 20 seeds. Report the win rate beside the mean.
- **Early stopping can cost.** A temporarily flat validation curve can precede useful learning.
  Stopping there sacrifices later gains; patience is a trade-off, not a guarantee.
- **Patience too small.** One noisy uptick stops training. Set patience from the typical plateau length.
- **Non-monotone curves.** Over-parameterised models can show a second descent, so a rising curve does not always mean
  stop. Test the claim before you rely on it.

## Interview angle

**60-second answer.** I read the training and validation curves together. Low training error with high validation error
means overfitting, and I choose between lower capacity, regularisation and early stopping. High error on both means
underfitting, so I add capacity. For stopping, I keep the best checkpoint, with a patience and a `min_delta`. Then I report
a test score that the tuning never touched, along with the spread across seeds.

**Follow-ups.**

- *Why not report the validation score?* It was used to choose the epoch, degree or penalty, so it is optimistically
  biased.
- *Why look at the win rate?* A small mean gain can hide that the checkpoint wins on only a minority of seeds.

Full answers are in `interview.md`.

## Connections

- **Requires:** `cml/k-fold-cross-validation` (choosing complexity without a single lucky split) and
  `cml/ridge-regression-and-weight-decay` (the penalty alternative).
- **Unlocks:** `cml/bias-variance-decomposition`, which explains the two errors seen in the table, and the regularisation
  topics that follow.
- **Lineage:** Bishop (2006, section 1.1) introduces polynomial over-fitting. Prechelt (1998) studies when to stop. The
[Deep Learning text, chapter 7](https://www.deeplearningbook.org/contents/regularization.html)
explains the regularization interpretation.
