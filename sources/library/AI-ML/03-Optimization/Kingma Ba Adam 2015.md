# Kingma & Ba - Adam (2015)

*AdaGrad knew coordinates need different step sizes. RMSProp learned to forget old scale. Adam put direction and scale into one practical default.*

**<- [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**"Adam: A Method for Stochastic Optimization"** - Diederik P. Kingma and Jimmy Lei Ba, ICLR 2015.

- Paper: `corpus/raw/library-papers/1412.6980.pdf`
- arXiv: `1412.6980`
- Local PDF verified as a PDF document
- Context: submitted in late 2014, published at ICLR 2015

## Key Ideas

1. **First-order methods were the only scalable option.**  
   Deep learning objectives were stochastic, high-dimensional, and noisy. Higher-order methods were too expensive; plain SGD was cheap enough, but brittle.

2. **Adam keeps two memories of the gradient.**  
   The first moment `m_t` is an exponential moving average of gradients: a momentum-like estimate of direction. The second moment `v_t` is an exponential moving average of squared gradients: an RMSProp-like estimate of recent scale.

   ```text
   m_t = beta_1 * m_(t-1) + (1 - beta_1) * g_t
   v_t = beta_2 * v_(t-1) + (1 - beta_2) * g_t^2
   ```

3. **Bias correction fixes the cold-start problem.**  
   Both moving averages start at zero, so early estimates are biased downward, especially when `beta_1` and `beta_2` are close to 1. Adam corrects this with:

   ```text
   m_hat_t = m_t / (1 - beta_1^t)
   v_hat_t = v_t / (1 - beta_2^t)
   theta_t = theta_(t-1) - alpha * m_hat_t / (sqrt(v_hat_t) + epsilon)
   ```

   This matters most for sparse gradients. The paper explicitly contrasts Adam with RMSProp-with-momentum: without bias correction, large `beta_2` values can create unstable early steps.

4. **The effective update is roughly scale-invariant.**  
   If the gradient is rescaled, the numerator and denominator scale together. That makes Adam less sensitive to raw gradient magnitude and lets the stepsize `alpha` behave more like a bound on update size.

5. **The defaults became part of the culture.**  
   The paper's tested defaults - `alpha = 0.001`, `beta_1 = 0.9`, `beta_2 = 0.999`, `epsilon = 1e-8` - gave practitioners a starting point that worked across many neural-network settings. That social fact matters: a good optimizer is not only a theorem; it is something people can actually use without a week of tuning ritual.

## What Came Before -> What It Enabled

### What came before

- [[Cauchy Steepest Descent 1847|Cauchy]] made optimization into downhill motion on a scalar objective.
- [[Robbins Monro Stochastic Approximation 1951|Robbins-Monro]] made noisy sample-based updates mathematically legitimate.
- [[Widrow ADALINE 1960|Widrow-Hoff]] made one-example error correction look like gradient descent in a trainable machine.
- [[Polyak Heavy-Ball Momentum 1964|Polyak]] showed that optimizers can carry directional memory.
- [[Duchi Hazan Singer AdaGrad 2011|AdaGrad]] gave every coordinate its own accumulated scale.
- [[Hinton RMSProp 2012|RMSProp]] replaced AdaGrad's lifetime accumulator with recent squared-gradient memory.

### What broke

Each ancestor fixed a real problem, but none of them quite matched the training regime that deep learning was entering.

Plain SGD had one global learning rate. In high-dimensional networks, different parameters see wildly different gradient scales and frequencies. AdaGrad handled sparse coordinates, but its lifetime squared-gradient sum could shrink learning rates too aggressively over long nonconvex training runs. RMSProp fixed stale scale memory, but the common RMSProp-with-momentum recipe mixed momentum with rescaled gradients and lacked Adam's explicit bias correction.

The old recipe broke on **usable adaptivity**. Neural networks needed an optimizer that could:

- move consistently through noisy mini-batch gradients,
- adapt per-parameter step sizes,
- handle sparse and non-stationary gradients,
- avoid huge early updates caused by zero-initialized moment estimates,
- and still be simple enough to become a default.

Adam's answer was not a new architecture. It was bookkeeping: keep a smoothed direction, keep a smoothed recent scale, correct their initialization bias, and update with their ratio.

### What it enabled

- A robust default optimizer for deep neural networks.
- A direct bridge from RMSProp to [[Loshchilov AdamW 2019|AdamW]], where the remaining failure was not adaptivity but regularization.
- Practical training recipes for Transformers, language models, VAEs, GANs, and many other deep models.
- The modern habit of thinking about optimizer state as part of the model-training system, not as an afterthought.

## Why It Matters

Adam matters because it made stochastic optimization feel less like hand-tuned machinery and more like a reliable component. That does not mean it solved optimization. It did not erase learning-rate schedules, batch-size effects, generalization gaps, or weight-decay mistakes. But it moved the default.

The lineage question is precise: why couldn't they just keep using RMSProp, AdaGrad, or SGD with momentum?

Because frontier-scale learning was becoming too noisy, sparse, high-dimensional, and non-stationary for any one of those tricks alone. AdaGrad remembered too much. RMSProp had recent scale but not Adam's clean biased-moment correction story. Momentum had direction but not coordinate-wise normalization. Adam fused the pieces into one optimizer whose parameters had intuitive meanings and whose defaults often worked.

That is why Adam sits under the Transformer era. [[Vaswani Attention Is All You Need 2017|Attention Is All You Need]] trained with Adam. BERT, GPT-style pretraining, and the early foundation-model stack inherited the expectation that adaptive moment estimation was normal. [[Loshchilov AdamW 2019|AdamW]] then fixed the next break: Adam's L2 regularization was not true weight decay under adaptive scaling.

Generative AI is not only architectures and data. It is also the optimizer state quietly carrying the run across billions or trillions of noisy updates.

## Lineage

- **Built on:** stochastic approximation, momentum, AdaGrad, RMSProp, and mini-batch deep learning.
- **Leads to:** AdamW, Adam variants, adaptive optimizer defaults in Transformers, and later large-scale training recipes with decoupled weight decay and schedule-aware tuning.
- **Modern echo:** when modern LLM papers specify AdamW with custom `beta_2`, clipping, warmup, and weight decay, the base grammar is still Adam's: first moment for direction, second moment for scale.

---

*Adam's trick was not that gradients became easy. It made noisy gradients administrable.*
