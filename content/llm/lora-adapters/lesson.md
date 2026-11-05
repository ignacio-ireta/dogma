---
title: "LoRA adapters"
minutes: 12
min_viable: "Read The system and the Worked example, then implement p01 with Hint 1 open."
---

## The system

A linear layer computes $y = xW^\top + b$. Symbols: $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ is the
base weight, $x$ one input row of length $d_\text{in}$, $b \in \mathbb{R}^{d_\text{out}}$ the bias. LoRA
freezes $W$ and adds a low-rank update:

$$
y = xW^\top + b + \frac{\alpha}{r}\, x A^\top B^\top,
\qquad A \in \mathbb{R}^{r \times d_\text{in}},\quad B \in \mathbb{R}^{d_\text{out} \times r}.
$$

The update $\Delta W = \frac{\alpha}{r} BA$ has rank at most $r$. The rules:

- **Frozen base.** $W$ and $b$ have `requires_grad=False`. Only $A$ and $B$ receive gradients.
- **Zero-initialised $B$.** Draw $A$ from a random distribution and set $B = 0$. Then $\Delta W = 0$ at
  step zero, so the first output equals the base model. If $A = 0$ as well, both gradients are zero and
  nothing trains, so $A$ must be random.
- **Scale.** The scale is $\alpha / r$, with $r = A.\text{shape}[0]$. Holding $\alpha$ fixed, a larger $r$
  shrinks the scale of each rank-one term. Treat $\alpha$ as a hyperparameter tuned with the learning rate.
- **Merge.** $W' = W + \frac{\alpha}{r} BA$ has the shape of $W$, and $y = xW'^\top + b$ gives the same
  values. Inference then pays no extra cost. A merged weight serves one adapter, so switching adapters
  needs the unmerged path or a re-merge.

## Worked example

Take $d_\text{in} = d_\text{out} = 2$, $r = 1$, $\alpha = 2$, so the scale is $2$. Let $W = I$, $b = 0$,
$A = [1, 0]$, $B = [0, 1]^\top$ and $x = [1, 2]$.

- Base path: $xW^\top = [1, 2]$.
- Adapter path: $xA^\top = 1 \cdot 1 + 2 \cdot 0 = 1$. Then $(xA^\top)B^\top = 1 \cdot [0, 1] = [0, 1]$.
  Scaled by $2$: $[0, 2]$.
- Output: $y = [1, 2] + [0, 2] = [1, 4]$.

Merged path: $BA = \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix}$, so $W' = I + 2BA = \begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}$
and $xW'^\top = [1, 4]$. Same output, one matrix product.

Zero $B$: the adapter path is $0$, so $y = xW^\top + b$, the base layer exactly.

## Why it works

The truncated SVD is the best rank-$r$ approximation of a fixed matrix in the Frobenius norm (Eckart and Young,
1936; see [library:AI-ML/02-Linear-Algebra/Eckart-Young Best Low-Rank Approximation 1936.md]). LoRA does not apply
it, because it never approximates $W$. It constrains the update $\Delta W$ to rank at most $r$ and learns $A$ and
$B$ by gradient descent, so $BA$ has rank at most $r$ by construction. The pretrained $W$ need not be low rank.
Rank is a choice about what to train, and task results judge it. Hu et al. (2021) define the update in §4.1, with
a Gaussian $A$, a zero $B$ and the scale $\alpha/r$ [corpus:library-refresh/01-Mathematics/05-Linear-Algebra/2021 -- Hu et al -- LoRA - Low-Rank Adaptation of Large Language Models -- arXiv -- v2.pdf.md p.4], and study rank in §7 [corpus:library-refresh/01-Mathematics/05-Linear-Algebra/2021 -- Hu et al -- LoRA - Low-Rank Adaptation of Large Language Models -- arXiv -- v2.pdf.md p.9].
The Library bridge note places LoRA in the wider low-rank lineage [source: library-refresh/01-Mathematics/05-Linear-Algebra/1990-2021 -- Library -- From Latent Semantic Analysis to Embeddings and Low-Rank Adaptation -- Concept Note.md]. The LLaMA
note gives context, not equations ([library:AI-ML/13-Scaling-and-Alignment/Touvron et al. — LLaMA 2023.md]).

Gradients. With $G = \partial L / \partial y$ (shape $(N, d_\text{out})$ for $N$ rows):
$\partial L / \partial B = \frac{\alpha}{r} G^\top (xA^\top)$ uses the bottleneck activation $xA^\top$.
$\partial L / \partial A = \frac{\alpha}{r} (GB)^\top x$ uses the input and the upstream gradient through $B$.
The optimizer state scales with the trainable count $r(d_\text{in} + d_\text{out})$, not with $W$.

## Signal vs. noise

Signal: the frozen base, the scale $\alpha / r$, the zero-initialised $B$, and that merging gives the same
function. Noise: the optimizer and the random distribution for $A$. Dropout on the adapter input does change the
training-time forward function, so compare merged and unmerged outputs in eval mode.

## Where it breaks

- **Both factors zero.** With $A = 0$ and $B = 0$ both gradients vanish and nothing trains.
- **Scale by $\alpha$ alone.** Changing $r$ then silently changes the update size. Check the $\alpha / r$ rule.
- **Detaching the adapter.** A `.detach()` on $A$ or $B$, or an in-place write into $W$ during merging,
  breaks the gradient or corrupts the base model for later calls.
- **Wrong orientation.** $A$ is $(r, d_\text{in})$ and $B$ is $(d_\text{out}, r)$. A transposed $B$ can pass
  a size check and still compute the wrong function.
- **Merged and unmerged disagree.** The merged weight must be $W + \frac{\alpha}{r} BA$, in that order.
  Check that both paths agree on random inputs.

## Interview angle

Sixty-second answer: LoRA keeps the pretrained $W$ frozen and learns $\Delta W = \frac{\alpha}{r} BA$, of rank
$r$. $B$ starts at zero, so training starts at the base model, and only $A$ and $B$ get gradients. At
inference the update can be merged into $W$ at no cost, or kept separate to serve several adapters.
Likely follow-up: why zero-initialise $B$ and not $A$? Both zero gives zero gradient. Zero $B$ makes the
first output the base output, and random $A$ lets gradients reach $B$. Second follow-up: what does the rank
buy? A bottleneck: trainable parameters grow as $r(d_\text{in} + d_\text{out})$ instead of $d_\text{in} d_\text{out}$.

## Connections

- Prerequisites: [Dense layer forward pass](../../dl/dense-layer-forward-pass/) for the base product, and
  [Gradients with framework autograd](../../dl/gradients-with-framework-autograd/) for which tensors get gradients.
- Next: [LoRA parameter and memory arithmetic](../lora-parameter-and-memory-arithmetic/), which counts the
  trainable parameters and the optimizer memory this design saves.
