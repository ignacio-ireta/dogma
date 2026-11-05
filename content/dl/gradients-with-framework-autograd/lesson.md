---
title: "Gradients with framework autograd"
minutes: 12
min_viable: "Read The system and Worked example, then solve p01 using Hint 1 if needed."
---

## The system

A tensor records the operations that produced it when it is tracked. Three things matter.

1. **Tracking.** A tensor is tracked when `requires_grad=True`, or when it was computed from a
   tracked tensor while grad mode is on. A tracked tensor produced by an operation stores a
   `grad_fn`, a pointer to the backward rule and its inputs. This recorded structure is the graph.
2. **Leaf vs non-leaf.** A leaf is a tensor you created directly, such as a parameter. A non-leaf
   is the result of an operation. Only leaves keep `.grad` after `backward()`. A non-leaf needs
   `retain_grad()` to keep its gradient.
3. **Accumulation.** `backward()` adds into `.grad` across calls; it does not replace it, so every
   training step must clear the old value. Recent PyTorch sets `.grad` to `None` on `zero_grad()` by default, which skips a write and keeps an unset gradient visible.
   Within one pass, a tensor used by two operations gets the sum of both adjoints, a different
   thing that needs no clearing ([Baydin et al. 2018, §3.2](../../../sources/library-refresh/01-Mathematics/03-Calculus/2018%20--%20Baydin%20et%20al%20--%20Automatic%20Differentiation%20as%20Executable%20Chain%20Rules%20--%20Study%20Note.md)).

Gradient entry points:

| Call | What it does |
|---|---|
| `y.backward()` | `y` must be a scalar. Adds $\partial y/\partial \theta$ into every leaf's `.grad` |
| `y.backward(gradient=v)` | `v` must have the shape of `y`. It turns `y` into the scalar $v\cdot y$, and adds the vector-Jacobian product $v^\top J$ |
| `torch.autograd.grad(y, x, grad_outputs=v)` | Returns $v^\top J$ for `x` as a tuple. Touches no `.grad` field |

Turning tracking off:

- `detach()` returns a tensor sharing storage, cut from the graph. Writes through it change the original.
- `torch.no_grad()` records nothing inside the block. `torch.enable_grad()` turns it back on.
- `torch.inference_mode()` is stricter: tensors made inside are inference tensors. They cannot be saved for
  backward, so `w * z` raises when `z` is one of them, while `w + z` works.

## Worked example

Let $w=(1,2)$ be a leaf with `requires_grad=True`. Let $h=3w$ (a non-leaf) and $y=\sum_i h_i^2$.

- Chain rule: $\partial y/\partial h_i = 2h_i$, so $\partial y/\partial w_i = 2h_i\cdot 3 = 18 w_i$.
- Call `y.backward()`. Then `w.grad` is $(18, 36)$, and `h.grad` is `None`: `h` is not a leaf, so the engine did not keep its gradient.
- Rebuild `y` from `w` and call `y.backward()` again without zeroing. `w.grad` becomes $(36, 72)$. Calling backward on the old graph raises unless you pass `retain_graph=True`.
- Rebuild `y` and use `torch.autograd.grad(y, w)` instead. It returns $(18, 36)$ and leaves `w.grad` unchanged.

For a non-scalar $v=x^2$ with $x=(1,2)$, `v.backward(gradient=(1,10))` computes $J^\top g$ with $J=\operatorname{diag}(2x)$: the result is $(2, 40)$.

## Why it works

One reverse pass gives the derivative of a scalar loss with respect to every leaf, at a small constant
multiple of one forward pass. Forward mode computes $J\mathbf{v}$ for one input direction per pass, so
for $d$ parameters it needs $d$ passes. Reverse mode computes $\mathbf{v}^\top J$ for the single scalar
output ([Jacobians note](../../../sources/library-refresh/01-Mathematics/03-Calculus/2026%20--%20Library%20--%20From%20Derivatives%20to%20Jacobians%20and%20Hessians%20--%20Concept%20Note.md)). Each local rule maps the incoming
gradient $\bar{\mathbf{v}}$ to $J^\top\bar{\mathbf{v}}$, the chain rule of [book: Goodfellow, Bengio and
Courville, Deep Learning, ch. 6 (back-propagation)]. Accumulation across calls suits gradient accumulation
over several losses, and it is also the source of the stale-gradient bug.

`torch.autograd.grad` gives the same derivative with no side effects, so it suits gradient checks and
sensitivity analysis. Unlike `backward()` on a copy, it leaves `.grad` alone. Use `create_graph=True` only for second derivatives, such as a gradient penalty.

## Signal vs. noise

Signal: scalar loss versus `gradient=` for vectors, accumulation across calls and within a pass, leaf
versus non-leaf, and where the graph is cut. Noise: which backward-rule implementation runs, and the exact
warnings text.

## Where it breaks

- **Stale gradients.** Forgetting to clear `.grad` adds the previous step's gradient into the current one.
- **Graph cut by in-place or detach.** `detach()` or an in-place edit that is not tracked removes the path, so the parameter's gradient is `None` or zero, silently.
- **Gradient through a rounding step.** `round` and `floor` have zero derivative almost everywhere, so a plain forward pass stops training. The straight-through estimator is a chosen rule, not the function's derivative. Small-$h$ finite differences return zero and cannot check it. Over a span of two steps, $Q(x+s)-Q(x-s)=2s$ away from ties, a secant slope of exactly 1: the average slope the identity estimator matches.
- **Kinks and ties.** Autograd differentiates the branch that ran. At a kink such as `abs` at $0$, or at a rounding tie, the returned value is a convention. The chain rule does not repair a nonsmooth primitive ([Baydin et al. 2018, per the Library note]).
- **Inference tensors.** A `w * z` that needs an inference-mode `z` raises. Use `no_grad` to skip the graph only.
- **Clearing `requires_grad` on a non-leaf.** `requires_grad_(False)` on a non-leaf raises an error.
  Detach it instead.

## Interview angle

State the graph in one sentence: "ops on tracked tensors record a `grad_fn`, and `backward` walks it in reverse, adding into leaves' `.grad`." Then say the two traps: clear the gradients every step, and know when a tensor is not tracked. Name `autograd.grad` as the choice when you need a derivative without touching state. Follow-ups: "why does a non-scalar output need a vector?" (backward needs a scalar, so $\mathbf{v}$ turns $\mathbf{y}$ into the scalar $\mathbf{v}\cdot\mathbf{y}$), and "how do you check a gradient?" (central finite differences in float64 against autograd).

## Connections

Requires `dl/reverse-mode-autograd-engine`, which built a scalar graph by hand; this topic uses the
framework's tensor version. Next are `dl/one-training-step`, which puts
zeroing, backward and update in order, and `dl/custom-autograd-functions`. The straight-through
estimator is an example of a rule chosen on purpose, as used in the vector-quantized autoencoder of
[VQ-VAE note](../../../sources/library/AI-ML/12-Frontiers/van%20den%20Oord%20VQ-VAE%202017.md); the PDF was not inspected. The same transposed-Jacobian product drives implicit
differentiation of a solver's optimality conditions ([Blondel et al. 2022](../../../sources/library-refresh/01-Mathematics/08-Optimization/2022%20--%20Blondel%20et%20al%20--%20Implicit%20Differentiation%20and%20Optimization%20Layers%20--%20Study%20Note.md)).
