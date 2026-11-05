---
title: "Batch normalization"
minutes: 14
min_viable: "Read The system and Worked example, then do the forward pass in p01 using Hint 1 if stuck."
---

## The system

Batch normalization standardizes each feature across the rows of a mini-batch,
then lets the network undo the standardization if it wants to. Let
$x\in\mathbb{R}^{N\times C}$ be a batch of $N$ rows with $C$ features. Let
$\gamma,\beta\in\mathbb{R}^{C}$ be learned scale and shift. Let
$\mu_r,\sigma^2_r\in\mathbb{R}^{C}$ be running statistics kept between batches.

The rules, stated explicitly:

| Quantity | Training ($\text{train}=\text{True}$) | Inference ($\text{train}=\text{False}$) |
|---|---|---|
| Mean | $\mu_B=\frac{1}{N}\sum_i x_{i,:}$, shape $C$ | $\mu_r$ (stored) |
| Variance for normalizing | $\sigma_B^2=\frac{1}{N}\sum_i (x_{i,:}-\mu_B)^2$ (**divide by N**) | $\sigma^2_r$ (stored) |
| Output | $y=\gamma\odot\hat x+\beta$, $\hat x=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$ | same formula with $\mu_r,\sigma^2_r$ |
| Running variance update | $\sigma^2_r\leftarrow(1-m)\sigma^2_r+m\,\tilde\sigma^2_B$, $\tilde\sigma^2_B=\frac{N}{N-1}\sigma^2_B$ (**divide by N-1**) | unchanged |
| Running mean update | $\mu_r\leftarrow(1-m)\mu_r+m\,\mu_B$ | unchanged |

Here $\epsilon>0$ is a small constant (eps), $m\in[0,1]$ is the momentum,
and $\odot$ is elementwise multiplication. Every statistic is taken **per
feature**, over the batch axis. Spatial dimensions are not covered here.

Two conventions to fix now:

- **Momentum.** PyTorch uses $m$ as the weight on the **new** batch statistic.
  The default is $m=0.1$. Some frameworks and papers write the same update with
  the weight on the old value, so their default looks like $0.9$ or $0.99$.
  Read the framework's definition before you copy a number.
- **Variance.** Normalization uses the biased variance, which divides by $N$.
  The running estimate is meant to estimate the population variance, so it uses
  the unbiased variance, which divides by $N-1$. Ioffe and Szegedy use the same split
  [library-papers/1502.03167 p.4].

## Worked example

Take $N=2$, $C=2$, and $x=\begin{bmatrix}1&2\\3&6\end{bmatrix}$. Treat $\epsilon$ as
negligible here (it must be positive in code) so the arithmetic stays exact.

Training. Column means are $\mu_B=[2,\,4]$. Deviations are $\pm1$ in column 0 and
$\pm2$ in column 1. Biased variances are $\sigma_B^2=[1,\,4]$. The normalized output
is $\hat x=\begin{bmatrix}-1&-1\\1&1\end{bmatrix}$. With $\gamma=[1,1]$ and $\beta=[0,0]$,
$y=\hat x$.

Running update with $m=0.1$, starting from $\mu_r=[0,0]$ and $\sigma^2_r=[1,1]$.
The unbiased variances are $\frac{2}{1}\cdot[1,4]=[2,8]$. So
$\mu_r'=0.1\cdot[2,4]=[0.2,\,0.4]$ and $\sigma^2_r{}'=0.9\cdot[1,1]+0.1\cdot[2,8]=[1.1,\,1.7]$.

Inference. Now use $\mu_r=[0.5,1]$, $\sigma^2_r=[4,9]$, $\gamma=[2,1]$,
$\beta=[0,1]$ on the same $x$. Then $x-\mu_r=\begin{bmatrix}0.5&1\\2.5&5\end{bmatrix}$,
divide by $\sqrt{\sigma^2_r}=[2,3]$ to get $\begin{bmatrix}0.25&0.33\\1.25&1.67\end{bmatrix}$,
then $y=\begin{bmatrix}0.5&1.33\\2.5&2.67\end{bmatrix}$. The running statistics stay as they were.

## Why it works

The paper motivates batch normalization by internal covariate shift: the
distribution of each layer's inputs keeps changing as earlier layers train
[library-papers/1502.03167 p.2]. Its stated payoff is faster training with much
higher learning rates [library-papers/1502.03167 p.1]. For this course, the
practical reason to remember is simpler: standardizing each feature gives the
next layer inputs on a stable scale. Stored running statistics make inference
independent of the other examples in the batch.

The one derivation that matters is the variance split. The mean of $\sigma_B^2$
over batches is $\frac{N-1}{N}\sigma^2$, which underestimates the population
variance $\sigma^2$. Multiplying by $\frac{N}{N-1}$ removes that bias. The
normalization itself uses the biased form, so each batch is exactly standardized.
The running estimate needs the unbiased form.

The backward pass has the same structure. With $g=\partial L/\partial y$ and
$\sigma=\sqrt{\sigma_B^2+\epsilon}$, the parameter gradients are
$\partial L/\partial\beta=\sum_i g_i$ and $\partial L/\partial\gamma=\sum_i g_i\hat x_i$.
The input gradient is $\partial L/\partial x=\frac{\gamma}{N\sigma}\left(Ng-\sum_i g_i-\hat x\sum_i g_i\hat x_i\right)$.
The two sums are the mean and variance paths: each row moves every other row's
statistics. Part p02 derives and checks this form.

## Signal vs. noise

Signal: the train-versus-eval split, per-feature statistics over the batch axis,
and the two conventions above (momentum and variance).

Noise: the exact value of eps (common defaults are $10^{-5}$ and $10^{-3}$), and
the affine parameters' initial values, which are $\gamma=1$ and $\beta=0$ in most
libraries. The backward pass is a second step. Derive it only after the forward
pass and the running-statistics update match the reference.

## Where it breaks

- **Batch size one in training.** With $N=1$, every feature's variance is zero, so
  the normalized output is $0$ before $\epsilon$ and the statistics carry no
  information. The contract in p01 raises an error. PyTorch raises one too.
- **Small batches.** Batch statistics are noisy when $N$ is small, so the output
  for one example depends on the other examples in its batch. Training and
  inference then disagree. This is why layer normalization exists (next topic).
- **Train and eval mismatch.** Forgetting `model.eval()` normalizes with batch
  statistics at inference time. Outputs then depend on batch composition. Check
  this first when predictions change between a notebook and a server.
- **Large common offsets.** Computing $\mathbb{E}[x^2]-\mathbb{E}[x]^2$ in one pass
  loses precision when the mean is large. Center first, then square.

## Interview angle

Sixty-second answer: batch normalization standardizes each feature over the mini-batch
during training and reuses running estimates at inference, so the output of one example does
not depend on its neighbours. It adds a learned scale and shift to keep the layer expressive.
The trade-offs are the batch-size dependence and the train-eval mismatch.

Likely follow-ups: why keep two variances, and why does the running variance differ from the
normalizing variance? Why does batch norm fail for recurrent or online settings? Full versions
are in `interview.md`.

## Connections

Prerequisites: [dense layer forward pass](../dense-layer-forward-pass/lesson.md) gives the
per-feature layout, and [expectation and variance](../../cml/expectation-and-variance/lesson.md)
gives the biased and unbiased estimators. This unlocks the backward pass (p02), layer
normalization, which removes the batch dependence, and RMSNorm. Lineage: Ioffe and Szegedy, 2015
[library-papers/1502.03167 p.3], introduced the transform as Algorithm 1 of the paper.
