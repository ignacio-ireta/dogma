# Hinton, Srivastava & Swersky — RMSProp (2012)

*AdaGrad learned per-coordinate memory; RMSProp learned to forget.*

**← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**"Neural Networks for Machine Learning, Lecture 6: Overview of Mini-Batch Gradient Descent / RMSProp"** — Geoffrey Hinton, with Nitish Srivastava and Kevin Swersky, University of Toronto / Coursera, 2012.

- Paper: `corpus/raw/library-papers/hinton-rmsprop-2012.pdf`
- Source: Toronto-hosted lecture slides for Hinton's *Neural Networks for Machine Learning*
- Note: RMSProp was not introduced as a conventional peer-reviewed paper. The source is the original lecture material; the slides credit Tijmen Tieleman for the unpublished RMSProp idea.
- Local PDF verified as a PDF document

## Key Ideas

1. **Mini-batch training made old optimizer intuitions unstable.**  
   Full-batch methods could use expensive, smooth optimization machinery. Online and mini-batch neural-network training could not: every update saw only a noisy slice of the data, but GPUs made mini-batches the practical route for large networks.

2. **A single learning rate still sloshes across ravines.**  
   Hinton frames the familiar problem: in a narrow error valley, the gradient is large in the direction where we want only a small correction and small in the direction where we need sustained movement. Too large a learning rate oscillates or diverges; too small a rate crawls.

3. **RProp had the right instinct, but the wrong batch regime.**  
   Resilient backpropagation used gradient signs and per-weight step sizes to avoid being dominated by raw gradient magnitude. That works better in deterministic full-batch settings. In mini-batches, dividing by each batch's raw gradient magnitude makes the denominator jump around with sampling noise.

4. **RMSProp smooths the denominator.**  
   The core move is to keep an exponential moving average of squared gradients for each weight:

   ```text
   MeanSquare_t(w) = 0.9 * MeanSquare_(t-1)(w) + 0.1 * (dE/dw_t)^2

   w_(t+1) = w_t - eta * (dE/dw_t) / sqrt(MeanSquare_t(w))
   ```

   Modern implementations add a small epsilon in the denominator for numerical stability, but the lineage idea is already here: scale each coordinate by a recent, smoothed root-mean-square gradient.

5. **The method fixes AdaGrad's long-memory failure.**  
   [[Duchi Hazan Singer AdaGrad 2011|AdaGrad]] accumulates all past squared gradients. That is excellent for sparse convex problems, but in long neural-network training runs the accumulator can grow until learning rates become too tiny. RMSProp replaces the lifetime sum with a moving average, so the optimizer remembers recent scale without being permanently punished by ancient gradients.

6. **RMSProp is the second-moment half of Adam.**  
   [[Polyak Heavy-Ball Momentum 1964|Momentum]] tracks the first moment: a smoothed direction. RMSProp tracks the second moment: a smoothed scale. [[Kingma Ba Adam 2015|Adam]] combines both and adds bias correction. Without RMSProp, Adam's genealogy jumps from "coordinates need different rates" to "adaptive moments" too abruptly.

## What Came Before → What It Enabled

### What came before

- [[Cauchy Steepest Descent 1847|Cauchy]] gave downhill motion.
- [[Robbins Monro Stochastic Approximation 1951|Robbins-Monro]] made noisy sample-based updates legitimate.
- [[Polyak Heavy-Ball Momentum 1964|Polyak]] showed that optimizer memory could carry useful direction.
- [[Duchi Hazan Singer AdaGrad 2011|AdaGrad]] gave every coordinate its own learning-rate history.

### What broke

AdaGrad's memory was too loyal.

For sparse online learning, a lifetime sum of squared gradients is a feature: common coordinates calm down, rare coordinates keep bigger effective steps. But deep networks are long nonconvex training processes. Their gradient scales change as representation layers organize, activations shift, and the model moves through different local geometries.

If the optimizer never forgets, early large gradients can haunt a parameter forever. The effective learning rate decays even when the current training regime needs movement again. Meanwhile, raw mini-batch RProp is too jittery because each batch can change the gradient magnitude denominator.

So the old recipe broke on **stale scale memory**: adaptive learning rates needed history, but not infinite history.

RMSProp made the memory recent.

### What it enabled

- Practical adaptive learning rates for mini-batch neural networks.
- A direct bridge from AdaGrad to [[Kingma Ba Adam 2015|Adam]]: moving averages of squared gradients become the `v_t` state.
- Better training recipes for recurrent and deep networks before Adam became the default.
- The modern optimizer habit of separating two questions: "What direction are gradients consistently pointing?" and "What is the recent scale of each coordinate?"

## Why It Matters

RMSProp is awkward historically because it is everywhere and nowhere: essential in the optimizer family tree, but born as lecture material rather than a polished paper. That actually makes it a useful lineage note. Deep learning did not advance only through grand architecture papers; it also advanced through practical fixes that made training stop falling apart.

The conceptual break is precise. AdaGrad says: "Coordinates with many large past gradients should slow down." RMSProp says: "Yes, but only recent past gradients should decide current scale." That small change is why adaptive optimization could survive long neural-network runs.

This is one of the hidden hinges under generative AI. Transformers, diffusion models, and frontier LLMs are not trained by pure Cauchy descent. They are trained by stateful optimizers that carry momentum, recent gradient scale, schedule rules, and regularization choices. RMSProp supplied the recent-scale piece that Adam and AdamW later made standard.

They could not just keep doing AdaGrad because deep networks need an optimizer that can adapt and then re-adapt. A model's loss landscape changes under its feet. RMSProp gave the optimizer a short-term memory instead of a permanent ledger.

## Lineage

- **Built on:** steepest descent, stochastic approximation, RProp, mini-batch GPU training, and AdaGrad's coordinate-wise adaptivity.
- **Leads to:** Adam, AdamW, Adadelta, Adafactor, and the broader adaptive-optimizer family used in large-scale deep learning.
- **Modern echo:** Adam's `v_t = beta_2 v_(t-1) + (1 - beta_2) g_t^2` is RMSProp's moving squared-gradient accumulator with bias correction and a first-moment companion.

---

*AdaGrad kept receipts forever. RMSProp learned which receipts were still relevant.*
