# He et al. — Deep Residual Learning for Image Recognition (2015/2016)

*The depth barrier was no longer that gradients could not exist. It was that ordinary deep stacks could not reliably learn even the identity solution that their extra layers should have made available.*

**← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Deep Residual Learning for Image Recognition.”** Submitted to arXiv in December 2015; published at CVPR 2016.

- Paper: `corpus/raw/library-papers/1512.03385.pdf`
- arXiv: https://arxiv.org/abs/1512.03385
- Source PDF status: local file verified as `PDF document, version 1.5` on 2026-08-16

## The Question

[[Krizhevsky-Sutskever-Hinton AlexNet 2012|AlexNet]] proved that a deeper CNN, enough labeled data, GPUs, ReLUs, and regularization could overwhelm the hand-engineered vision regime. VGG and GoogLeNet then pushed depth further.

So why not keep stacking ordinary layers?

Because depth exposed a new failure. A 34-layer plain network had **higher training error** than an otherwise comparable 18-layer network. On CIFAR-10, a 56-layer plain network likewise trained worse than a 20-layer one. This was not ordinary overfitting: the deeper model was already worse on the training set.

It was also not simply the old vanishing-gradient story. He et al. used normalized initialization and [[../10-Backpropagation/Ioffe Szegedy Batch Normalization 2015|Batch Normalization]], observed healthy forward and backward signal magnitudes, and still saw degradation.

The contradiction was sharp. A deeper network should be able to copy the shallower solution and make every added layer an identity mapping. Its best achievable training error should therefore be no worse. In practice, the optimizer could not reliably find that apparently simple construction.

The field could not just keep adding transformations. It needed to change what each stack was asked to learn.

## Key Ideas

### 1. Learn a correction, not the whole mapping

For a desired mapping $H(x)$, a plain stack directly approximates $H$. A residual block instead learns

$$
F(x) = H(x) - x,
$$

and returns

$$
y = F(x) + x.
$$

If identity is the right answer, the transformed branch only needs to drive $F(x)$ toward zero. More generally, if the desired mapping is near identity, the block learns the smaller correction rather than reconstructing the full mapping from scratch.

The representational class is not the central change: either parameterization can in principle express the same function. The change is **optimization geometry**. Residual learning gives the solver a much easier reference point.

### 2. Keep the identity path parameter-free

The shortcut passes $x$ around two or three learned layers and adds it back elementwise. When dimensions match, this path introduces no learned parameters and negligible computation.

That matters for the paper’s controlled comparison. The 34-layer plain and residual networks could have the same depth, width, parameter count, and nearly the same cost; the residual version differed mainly in how its computation was parameterized.

When spatial or channel dimensions change, a projection shortcut can match them. But the experiments show that learned projections are not required everywhere: identity shortcuts are enough to solve the core degradation problem and keep the architecture economical.

### 3. Turn depth from a liability into useful capacity

The ImageNet comparison makes the repair visible:

- **18-layer plain:** 27.94% top-1 validation error
- **34-layer plain:** 28.54% — deeper became worse
- **18-layer ResNet:** 27.88%
- **34-layer ResNet:** 25.03% — deeper became better

The residual 34-layer model was about **2.8 percentage points** better than the 18-layer residual model while its plain counterpart degraded. Increased depth finally produced the result its extra capacity had promised.

### 4. Use bottleneck blocks to make extreme depth affordable

For 50-, 101-, and 152-layer models, the paper uses a three-layer bottleneck block: `1×1 → 3×3 → 1×1`. The first projection reduces channel width, the middle convolution performs spatial processing, and the final projection restores width.

The resulting 152-layer ResNet required about **11.3 billion FLOPs**, still below the reported **15.3 billion FLOPs** of VGG-16, despite being far deeper.

### 5. Separate optimization success from generalization

The 1202-layer CIFAR-10 model could drive training error below 0.1%, showing that residual learning had removed the basic optimization barrier even at extreme depth. But its 7.93% test error was worse than the 110-layer model’s 6.43%.

That result is important precisely because it is not triumphalist. Residual connections make depth trainable; they do not guarantee that every additional layer is useful or well regularized.

## What Came Before → What It Enabled

### What came before

- [[Krizhevsky-Sutskever-Hinton AlexNet 2012|AlexNet]] established the data + GPU + CNN recipe and made depth strategically valuable.
- [[../10-Backpropagation/Nair Hinton ReLU 2010|ReLU]] reduced saturation in deep supervised networks.
- [[../10-Backpropagation/Glorot Bengio Initialization 2010|Xavier-style initialization]] and related initialization work kept signals from collapsing at the start of training.
- [[../10-Backpropagation/Ioffe Szegedy Batch Normalization 2015|BatchNorm]] stabilized intermediate activations and allowed much more aggressive optimization.

These advances repaired vanishing/exploding signals well enough to reveal the next failure: **healthy gradients did not imply that a very deep plain network was easy to optimize.**

### What broke

Plain networks asked every added stack to learn a new unreferenced mapping. Even though a deeper network contained an identity-extended version of the shallower solution in principle, SGD struggled to reach it in feasible time. More depth could therefore raise training error instead of lowering it.

### What residual learning enabled

- Reliable training of 50-, 101-, and 152-layer vision networks.
- A 152-layer single model with **4.49% top-5 ImageNet validation error**, and a six-model ensemble with **3.57% top-5 test error**, which won ILSVRC 2015 classification.
- Transferable representations: replacing VGG-16 with ResNet-101 in the same Faster R-CNN setup improved COCO’s stricter mAP@[.5,.95] metric from 21.2 to 27.2, a **28% relative improvement**.
- A reusable residual-block grammar inherited by DenseNet, pre-activation ResNets, vision Transformers, and modern language models.
- The `Add` in the Transformer’s **Add & Norm** block, paired with [[../10-Backpropagation/Ba Kiros Hinton Layer Normalization 2016|Layer Normalization]].

## Why It Matters

ResNet matters because it changed the unit of deep learning from **“compute a new representation”** to **“preserve a representation and learn what should change.”**

That preservation path is one of the quiet structural reasons generative AI can be deep. In [[../12-Frontiers/Vaswani Attention Is All You Need 2017|the Transformer]], every attention or feed-forward sublayer contributes a residual update to a stream that can continue across the whole network. Attention decides what information to mix; the residual pathway lets each block modify the representation without having to recreate it.

The causal chain is:

**AlexNet makes depth valuable → initialization and BatchNorm keep signals alive → plain networks still degrade as depth increases → residual blocks make identity the default path and learned computation a correction → very deep networks become optimizable → Transformers inherit residual pathways around attention and feed-forward layers.**

A useful correction to the popular summary: ResNet is not merely a “gradient highway” paper. Its motivating failure remained even when gradients had healthy norms. The paper’s deeper contribution is a parameterization that makes useful solutions easier for optimization to reach.

## Limitations

- The original post-activation block was later refined by pre-activation ResNets, which made the identity path cleaner.
- Residual connections do not remove overfitting; the 1202-layer CIFAR model demonstrates that trainability and generalization are separate problems.
- When dimensions change, shortcuts need padding or projections, so the identity path is not literally untouched everywhere.
- The paper offers strong empirical evidence and an optimization intuition, not a complete theory of why residual parameterization reshapes the loss landscape.
- ResNets solved the depth degradation problem, not all efficiency problems. Memory, latency, data quality, and numerical stability remained active constraints.

## Lineage

- **Built on:** [[Krizhevsky-Sutskever-Hinton AlexNet 2012]], [[../10-Backpropagation/Nair Hinton ReLU 2010]], [[../10-Backpropagation/Glorot Bengio Initialization 2010]], [[../10-Backpropagation/Ioffe Szegedy Batch Normalization 2015]]
- **Leads to:** pre-activation ResNets, DenseNet, deep vision backbones, [[../12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]], BERT, GPT, and modern residual-stream language models
- **Core repair:** deeper plain stacks degraded even with healthy signals; residual blocks made identity the protected path and each learned block a correction

---

*Backpropagation could tell every layer how it was wrong. ResNet made sure a new layer did not have to replace what was already right.*
