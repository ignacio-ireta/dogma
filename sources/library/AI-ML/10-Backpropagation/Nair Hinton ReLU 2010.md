# Nair & Hinton — Rectified Linear Units Improve Restricted Boltzmann Machines (2010)

*The activation-function bridge from fragile, saturating deep networks to the fast end-to-end training that made AlexNet possible.*

**← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**Vinod Nair and Geoffrey E. Hinton, "Rectified Linear Units Improve Restricted Boltzmann Machines"** — ICML, 2010.

- Paper: `corpus/raw/library-papers/nair-relu-2010.pdf`
- Original author-hosted PDF: https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf
- Venue: Proceedings of the 27th International Conference on Machine Learning, pp. 807-814

## The Question

[[Hinton Deep Belief Nets 2006|Deep Belief Nets]] had revived multilayer learning, but their success depended on a complicated rescue operation: greedily pretrain each layer as a Restricted Boltzmann Machine, then fine-tune the whole network.

Why was that scaffolding needed?

The hidden units were usually binary stochastic units or smooth sigmoid/tanh units. Binary units could not preserve how strongly a feature was present. Sigmoids compressed large positive or negative inputs into saturated regions where gradients became tiny. As signals crossed more layers, useful magnitude information and useful gradients both faded.

The field could not reach modern deep learning by stacking more of the same units. It needed a nonlinearity that remained nonlinear without choking the learning signal.

## Key Ideas

### 1. A rectifier can be derived from many binary units

Nair and Hinton begin inside the RBM tradition. Replace one binary hidden unit with infinitely many copies that share weights but have progressively more negative biases. Their summed expected activity is approximately:

$$
\log(1 + e^x)
$$

This is the softplus function, a smooth approximation to a rectifier. The paper then uses an efficient noisy approximation during generative RBM training:

$$
h = \max(0, x + \mathcal{N}(0, \sigma(x)))
$$

The familiar deterministic form appears during discriminative fine-tuning:

$$
\operatorname{ReLU}(x) = \max(0, x)
$$

This derivation mattered historically. ReLU was not introduced as an arbitrary kink in a graph; it was connected to the probabilistic machinery the deep-learning community already trusted.

### 2. The gradient stays alive on the positive side

For backpropagation, the derivative is treated as zero when $x \le 0$ and one when $x > 0$.

That simple rule changes the optimization problem. A positive activation does not enter a flat sigmoid tail, so its gradient is not repeatedly multiplied by a number far below one. ReLU does not eliminate every vanishing-gradient problem, and inactive units can still receive zero gradient, but it makes deep feedforward training substantially less resistant.

### 3. Sparse activity comes for free

Negative inputs map exactly to zero. The representation therefore becomes sparse without a separate sparsity penalty: only a subset of units responds to any example.

The paper's learned units were not merely linear units hiding behind a fashionable name. On the NORB experiment, different features were active at different frequencies, and no unit was active for every image.

### 4. Rectifiers preserve relative intensity through depth

Binary units discard magnitude: a feature is on or off. Rectifiers preserve it. With zero bias and positive rescaling, ReLU is intensity equivariant:

$$
\operatorname{ReLU}(\alpha x) = \alpha \operatorname{ReLU}(x), \qquad \alpha > 0
$$

That relationship can survive across multiple layers. The representation changes predictably when image intensity changes, allowing a later invariant comparison, such as cosine similarity, to remove the nuisance variation without throwing away feature strength.

### 5. The empirical result weakened the case for mandatory pretraining

On Jittered-Cluttered NORB, a one-hidden-layer rectified model without pretraining reached **17.8% error**, compared with **18.7%** for a pretrained binary-unit model. With two hidden layers, rectified units consistently outperformed binary units under every tested pretraining configuration.

This was the revealing result: changing the unit could help more than the elaborate pretraining ritual.

## What Came Before → What It Enabled

### What came before

- [[Hinton Contrastive Divergence 2002|Contrastive Divergence]] made RBM training practical.
- [[Hinton Deep Belief Nets 2006|Deep Belief Nets]] used RBM stacks to work around the difficulty of training deep networks from random initialization.
- [[Rumelhart Hinton Williams Backpropagation 1986|Backpropagation]] supplied credit assignment, but sigmoid saturation and weak gradients still made depth difficult.

### What broke

The DBN recipe treated poor end-to-end trainability as something to route around. Each layer had to be taught separately before the whole model could learn together. Binary and saturating units also discarded information or compressed gradients precisely when the network needed to carry both through many layers.

That was not a scalable endpoint. Larger datasets and larger models would make layer-by-layer generative pretraining increasingly cumbersome.

### What ReLU enabled

- Faster discriminative training with stronger gradient flow.
- Sparse, magnitude-preserving hidden representations.
- Less dependence on greedy generative pretraining.
- The activation function used by [[../11-CNNs/Krizhevsky-Sutskever-Hinton AlexNet 2012|AlexNet]], where GPU compute, ImageNet-scale data, dropout, and ReLUs finally made deep supervised vision explode.
- A path toward much deeper networks, followed later by [[../11-CNNs/He ResNets 2015|ResNets]] when activation choice alone could no longer prevent degradation.

## Why It Matters

This paper is easy to underrate because `max(0, x)` now looks almost embarrassingly simple. But lineage is about what a method replaced.

Deep Belief Nets showed that deep representations could work if every layer was carefully pretrained. ReLU helped make that workaround unnecessary. It changed the hidden unit so that ordinary backpropagation had a cleaner path through the network.

The resulting chain is:

**Backprop assigned credit through hidden layers → DBN pretraining made depth temporarily trainable → ReLU reduced saturation and preserved feature strength → AlexNet trained a large supervised CNN fast enough to reset computer vision.**

That is why generative AI's ancestry includes an activation function born in an RBM paper. Modern foundation models do not depend on ReLU specifically, but they inherit the lesson: scaling only works after the local numerical behavior of every layer stops fighting optimization.

## Limitations

- The paper studies noisy rectifiers within RBMs, not the exact modern recipe of very deep supervised ReLU networks.
- ReLU has a zero-gradient negative region, which can create permanently inactive or "dead" units.
- Its experiments were modest by later standards, and convolutional models still outperformed its fully connected NORB systems.
- ReLU did not solve depth by itself. Initialization, GPUs, normalization, residual connections, data scale, and regularization supplied the rest of the bridge.

## Lineage

- **Built on:** [[Hinton Contrastive Divergence 2002]], [[Hinton Deep Belief Nets 2006]], [[Rumelhart Hinton Williams Backpropagation 1986]]
- **Leads to:** [[../11-CNNs/Krizhevsky-Sutskever-Hinton AlexNet 2012]], [[../11-CNNs/He ResNets 2015]]
- **Replaced:** saturating or information-poor hidden units as the default choice for deep feedforward learning

---

*The important thing about ReLU was not that it was complicated. It was that the old complication had become the problem.*
