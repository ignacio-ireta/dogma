# Loshchilov & Hutter — Decoupled Weight Decay Regularization (AdamW, 2019)

#paper #concept #read

**Authors:** Ilya Loshchilov, Frank Hutter
**Institution:** University of Freiburg
**Published:** arXiv:1711.05101 (November 2017), ICLR 2019
**PDF:** `papers/1711.05101.pdf`

---

## The Question

Everyone calls it "weight decay." But are L₂ regularization and weight decay actually the same thing?

For SGD: **yes.** For Adam: **no — and the difference matters a lot.**

---

## The Problem

Adam is an adaptive optimizer — it scales the gradient update for each parameter individually based on gradient history. L₂ regularization (adding λ‖w‖² to the loss) goes through this gradient scaling. But true **weight decay** should shrink weights independently of the gradient update.

In math:

**Adam + L₂ regularization** (what everyone was doing):
```
θ_t+1 = θ_t - α_t · m̂_t / (√v̂_t + ε) - α_t · λ · θ_t / (√v̂_t + ε)
```
The λ decay term gets *divided by the adaptive factor*. Parameters with large gradients get less regularization. That's not what you want.

**AdamW (decoupled weight decay):**
```
θ_t+1 = θ_t - α_t · m̂_t / (√v̂_t + ε) - α_t · λ · θ_t
```
The decay is applied directly to the weights, bypassing the adaptive scaling. Clean separation.

**Proposition 1:** For SGD with constant learning rate, L₂ and weight decay are equivalent.
**Proposition 2:** For adaptive methods, they are not equivalent, and L₂ gives suboptimal regularization.

---

## Key Results

- AdamW improves test error by **~15% relative** over Adam+L₂ on CIFAR-10 and ImageNet32×32
- AdamW reaches Adam+L₂'s test error faster and at better hyperparameter configurations
- The optimal weight decay factor becomes **more independent of the learning rate** in AdamW — this simplifies tuning
- Combined with cosine annealing + warm restarts → **AdamWR**, strong anytime performance

---

## Why This Became Standard

The practical impact was enormous. AdamW was adopted almost immediately in PyTorch, TensorFlow, Keras, and Caffe. Within two years, it was the default optimizer for training Transformer-based language models — including:

- **GPT-2** (Radford et al. 2019)
- **GPT-3** (Brown et al. 2020) — uses AdamW with β₁=0.9, β₂=0.95, weight decay=0.1
- **BERT** and its descendants
- Most large language models since

The fix was conceptually simple but empirically significant. The paper's insight — "you were doing weight decay wrong" — turned out to be true for a huge swath of the field.

---

## Normalized Weight Decay

The paper also proposes normalizing weight decay across batch sizes and training schedules:

```
λ = λ_norm · √(b / (B · T))
```

where b = batch size, B = reference batch size, T = total steps. This makes the optimal λ_norm more stable across training configurations. Useful when scaling up.

---

## Bayesian Connection

The Bayesian interpretation: treating weight decay as placing a Gaussian prior on weights. In the decoupled form, this prior is properly independent of the gradient noise structure. Aitchison (2018) showed this connects to Bayesian filtering — AdamW is performing approximate Bayesian inference with the prior respected cleanly.

---

## Cross-Thread Connections 🔗

- **[[Kingma Ba Adam 2015]]** — AdamW is a corrected Adam; you can't understand this paper without Adam
- **[[Vaswani Attention Is All You Need 2017]]** — Transformers are trained with AdamW; this optimizer is part of why they work
- **[[Brown GPT-3 2020]]** — GPT-3 explicitly uses AdamW; this is the optimizer that made 175B parameters tractable to train
- **[[He ResNets 2015]]** — ResNets were trained with SGD+momentum; the shift to AdamW happened with Transformers. The two optimizer choices (SGD for vision, AdamW for language) reflect different loss landscapes.
- **[[Widrow ADALINE 1960]]** — the conceptual ancestor: Widrow-Hoff delta rule is gradient descent with learning rate. Adam adds momentum and adaptive learning rates. AdamW adds proper regularization. It's the same idea, refined over 60 years.

---

## The Optimization Lineage

```
Cauchy 1847 (steepest descent)
  └── Widrow-Hoff 1960 (delta rule / LMS)
       └── SGD (stochastic, mini-batch)
            └── AdaGrad 2011 (adaptive learning rates)
                 └── RMSProp 2012 (Hinton, unpublished lecture notes)
                      └── Adam 2015 (Kingma & Ba) — momentum + adaptive rates
                           └── AdamW 2019 (Loshchilov & Hutter) — + proper weight decay
```

This is a clean lineage: every step fixes a specific limitation of the previous optimizer.

---

## References Worth Chasing

- **Kingma & Ba (2014)** — Adam — arXiv:1412.6980 — *the base optimizer being fixed*
- **Duchi et al. (2011)** — AdaGrad — *adaptive learning rates origin*
- **Wilson et al. (2017)** — "The Marginal Value of Momentum for Small Learning Rate SGD" — the paper showing Adam generalizes worse than SGD; the motivation for this work
- **Loshchilov & Hutter (2016)** — SGDR — arXiv:1608.03983 — *cosine annealing with warm restarts; combined with AdamW to make AdamWR*
- **Aitchison (2018)** — "L2 Regularization versus Batch and Weight Normalization" — Bayesian justification

---

## A Small Note on Timing

This paper was posted to arXiv in November 2017 — the same month as the Transformer (Vaswani et al., June 2017). The BERT paper was a year away. The authors couldn't have known that their fix for Adam regularization would become the default optimizer for the transformer era. Sometimes a small, clean insight turns out to be the right tool for the next generation of architecture.

---

## Status

- [x] Paper read and note created (March 10, 2026)
- [ ] Verify: does BERT actually use AdamW? Check Devlin et al.
- [ ] Trace Wilson et al. (2017) argument about Adam generalization — is it resolved by AdamW?
- [ ] Think about: why does SGD still win on vision tasks? Loss landscape differences?
