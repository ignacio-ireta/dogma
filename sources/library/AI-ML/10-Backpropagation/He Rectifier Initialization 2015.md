# He et al. 2015 — Rectifier-Aware Initialization and PReLU

*ReLU made deep networks easier to optimize, but the old initialization math still treated the activation as if it were linear. He et al. repaired that mismatch—and then exposed the next failure: a plain network could finally converge at great depth without becoming more accurate.*

**← [[Glorot Bengio Initialization 2010|Xavier initialization]] ← [[Nair Hinton ReLU 2010|ReLU]] → [[../11-CNNs/He ResNets 2015|ResNet]]**

---

## Paper / Year

**Kaiming He, Xiangyu Zhang, Shaoqing Ren & Jian Sun — “Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification” (2015)**

- Paper: `corpus/raw/library-papers/1502.01852.pdf`
- arXiv: [1502.01852](https://arxiv.org/abs/1502.01852)
- Source PDF status: verified locally as `PDF document, version 1.5` on 2026-08-18

## Key Ideas

### 1. ReLU fixed saturation, but broke the linear initialization assumption

[[Glorot Bengio Initialization 2010|Glorot and Bengio]] had shown that a deep network needs forward activations and backward gradients to keep roughly the same scale across layers. Xavier initialization therefore scales weights according to fan-in and fan-out.

That derivation, however, models the activation as linear. A ReLU is not linear across its full domain: for a symmetric zero-mean input, it sets roughly half the values—and half the local derivatives—to zero. If weights use the linear-case variance, every rectified layer silently removes part of the signal budget. Repeat that mistake through enough layers and activations or gradients shrink exponentially.

The important distinction is:

> ReLU removed one cause of vanishing gradients, but it did not make signal scale self-correcting.

### 2. Initialization must match the nonlinearity

For a rectified layer with fan-in `n`, the paper derives the sufficient forward condition

\[
\frac{1}{2}n\operatorname{Var}(w)=1.
\]

For ReLU, this gives

\[
\operatorname{Var}(w)=\frac{2}{n},
\qquad
\operatorname{Std}(w)=\sqrt{\frac{2}{n}}.
\]

This is the now-standard **He (or Kaiming) initialization**. The factor of two is not an arbitrary boost. It compensates for the second moment removed by the rectifier. The paper derives the same factor from backward propagation: about half the ReLU derivatives are zero, so gradient variance also needs rectifier-aware scaling.

For a parametric rectifier with negative slope `a`, the condition generalizes to

\[
\frac{1}{2}(1+a^2)n\operatorname{Var}(w)=1.
\]

Initialization and activation are therefore one coupled design problem, not independent knobs.

### 3. PReLU makes the negative slope learnable

The paper also introduces the **Parametric Rectified Linear Unit**:

\[
f(y_i)=
\begin{cases}
y_i, & y_i>0,\\
a_i y_i, & y_i\le 0.
\end{cases}
\]

Unlike ReLU (`a=0`) or a fixed leaky ReLU, PReLU learns `a_i` with backpropagation. It adds almost no computational cost and very few parameters. On the authors’ 14-layer comparison, channel-wise PReLU reduced ImageNet top-5 error from **13.34% to 12.75%**.

PReLU was the paper’s visible activation contribution. He initialization was the more durable infrastructure contribution: modern rectified networks still inherit it even when they do not use PReLU.

### 4. The decisive experiment is not the headline ImageNet score

With a 22-layer ReLU network, both Xavier and He initialization converged, although the rectifier-aware method began reducing error sooner. At **30 weight layers**, the difference became structural:

- He initialization made the network converge.
- Xavier initialization stalled completely with diminishing gradients.

This closes a real trainability gap. Earlier deep CNN recipes sometimes initialized a deeper model from a trained shallower one or inserted auxiliary classifiers to help gradients reach intermediate layers. Rectifier-aware initialization made it possible to train very deep rectified networks directly from scratch.

### 5. Convergence revealed a second, deeper break

The paper’s most important negative result is easy to miss. Once the 30-layer plain network could train, it performed **worse** than the 14-layer model. The authors report **38.56/16.59%** top-1/top-5 error for the 30-layer model versus **33.82/13.34%** for the 14-layer ReLU model, and note similar depth saturation or degradation in other work.

So initialization solved:

- exploding or vanishing signal scale,
- failure to begin useful learning,
- dependence on shallow-model pretraining.

It did **not** solve:

- why adding layers to a plain network could raise training error,
- why an optimizer could fail to find the identity-extended version of a shallower solution,
- how to make depth itself reliably useful.

That is precisely the problem [[../11-CNNs/He ResNets 2015|ResNet]] attacked later in 2015.

## What Came Before → What It Enabled

### What came before

- [[Glorot Bengio Initialization 2010|Xavier initialization]] made variance preservation an explicit condition for deep learning, but its derivation assumed linear activations.
- [[Nair Hinton ReLU 2010|ReLU]] avoided positive-side saturation and made end-to-end deep training much faster, but zeroed the negative half of its input distribution.
- [[../11-CNNs/Krizhevsky-Sutskever-Hinton AlexNet 2012|AlexNet]] proved the ReLU/GPU/data recipe at eight learned layers.
- VGG and GoogLeNet pushed depth further, but needed careful training devices such as shallow-network pretraining or auxiliary classifiers.

### What it enabled

- Direct training from scratch of much deeper rectified networks.
- An initialization rule matched to ReLU’s actual forward and backward statistics.
- The modern default pairing: rectified activations with Kaiming initialization.
- A clean diagnosis of the next bottleneck: even when signals survive and optimization starts, **plain depth can still degrade accuracy**.
- [[../11-CNNs/He ResNets 2015|Residual learning]], which protects an identity path so new layers learn corrections rather than having to rediscover the whole mapping.

## Why It Matters

This paper is the missing hinge between **“ReLUs make deep networks trainable”** and **“residual connections make extreme depth useful.”**

Before He initialization, a very deep ReLU network could fail for a mundane numerical reason: every layer was initialized using the wrong activation model, so useful scale disappeared before learning had a chance. After the repair, the authors could separate two problems that had previously been tangled together:

1. **Signal propagation:** do activations and gradients remain alive?
2. **Optimization geometry:** can a stack of plain nonlinear layers actually learn a solution at least as good as a shallower network?

He initialization answered the first. Its own 30-layer failure made the second impossible to ignore.

That is how the lineage advances. Xavier said initialization is part of the model. ReLU changed the model’s nonlinearity. He et al. made the initialization respect that change. Then ResNet changed the path through which the model itself learns.

The paper’s 4.94% ensemble top-5 error—reported as the first published result below the cited 5.1% human estimate on ImageNet—is historically striking. But the more lasting breakthrough is quieter: **if depth changes the statistics, the initialization must change with it.**

## Lineage

- **Built on:** [[Glorot Bengio Initialization 2010|Glorot & Bengio 2010]], [[Nair Hinton ReLU 2010|Nair & Hinton 2010]], [[../11-CNNs/Krizhevsky-Sutskever-Hinton AlexNet 2012|AlexNet 2012]]
- **Exposed:** the depth-degradation problem in plain rectified networks
- **Leads to:** [[../11-CNNs/He ResNets 2015|He et al. 2015/2016 — Deep Residual Learning]]

---

*The initialization repair did not finish the depth story. It made the next failure visible enough to name.*
