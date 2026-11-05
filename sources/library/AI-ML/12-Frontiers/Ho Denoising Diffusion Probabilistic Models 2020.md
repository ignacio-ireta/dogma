# Ho, Jain & Abbeel — Denoising Diffusion Probabilistic Models (2020)

*Instead of asking a generator to defeat a critic, corrupt data in a known way and learn each small step back.*

**← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**“Denoising Diffusion Probabilistic Models”** — Jonathan Ho, Ajay Jain, and Pieter Abbeel, NeurIPS 2020.

- Paper: `corpus/raw/library-papers/2006.11239.pdf`
- arXiv: [2006.11239](https://arxiv.org/abs/2006.11239)
- Source PDF status: local file verified as `PDF document, version 1.5` on 2026-09-03

## Key Ideas

1. **Generation becomes the reversal of a fixed corruption process.**
   The forward process gradually adds small amounts of Gaussian noise to a data sample `x₀` until `x_T` is nearly standard Gaussian noise. That direction is prescribed rather than learned. A neural reverse process `p_θ(x_{t-1} | x_t)` learns to undo one noisy step at a time, beginning from random noise and ending at a generated image.

2. **The many-step construction makes the reverse conditionals tractable.**
   A one-shot map from noise to the full data distribution is difficult to specify and optimize. With sufficiently small forward noise increments, each reverse transition can be modeled as a Gaussian. The forward process also permits direct sampling of any noisy state `x_t` from `x₀`, so training does not have to simulate all earlier steps.

3. **The practical target is simply the noise that was added.**
   Ho et al. parameterize the reverse mean through a network `ε_θ(x_t, t)` that predicts the Gaussian noise in a corrupted example. Their simplified objective is a mean-squared error:

   `E_{t,x₀,ε} || ε − ε_θ(√ᾱ_t x₀ + √(1−ᾱ_t) ε, t) ||²`.

   Training therefore reduces to choosing a random timestep, creating a noisy example in closed form, and teaching a time-conditioned network to recover the known noise.

4. **Variational inference and score matching meet in one recipe.**
   The full model is a latent-variable Markov chain trained with a variational bound on negative log likelihood. Under the noise-prediction parameterization, its reverse dynamics also resemble denoising score matching across noise scales and annealed Langevin dynamics. The paper turns those connections into a simple algorithm that trains with ordinary stochastic gradient descent and backpropagation.

5. **A U-Net learns denoising at every scale.**
   One shared U-Net receives the noisy image and a sinusoidal embedding of timestep `t`; the experiments use self-attention at the `16 × 16` feature-map resolution. Large structures emerge early in reverse diffusion and fine details later, matching the paper’s interpretation of generation as progressive lossy decompression.

6. **The result made diffusion competitive with the dominant image generators.**
   With `T = 1000` reverse steps, the unconditional CIFAR-10 model achieved Inception Score `9.46` and FID `3.17`; its 256×256 LSUN samples were reported as comparable to ProgressiveGAN. Crucially, the best perceptual samples came from the simplified noise-prediction objective, even though the exact variational bound produced better codelengths. Likelihood quality and visual quality were not the same objective.

## What Came Before → What It Enabled

### What came before

- [[Goodfellow GANs 2014|GANs]] could create sharp images without an explicit likelihood, but training coupled a generator and discriminator in a minimax game. Mode collapse, vanishing generator gradients, and unstable balance between the two networks made the recipe fragile.
- [[Radford DCGAN 2016|DCGAN]] showed that convolutional design could make adversarial image generation substantially more practical, but architecture did not remove the game itself.
- Sohl-Dickstein et al. (2015) had already defined diffusion probabilistic models, while denoising score matching and annealed Langevin dynamics offered a related route through gradients of noisy data densities. Neither route had yet demonstrated GAN-competitive high-quality image synthesis with a compact training recipe.

### What broke

The field could not just keep scaling adversarial training. A stronger discriminator could eliminate the generator’s useful gradient; a generator could cover only a few modes while still fooling the critic; and progress depended on maintaining a moving equilibrium between two learners.

Likelihood-based alternatives avoided that game but carried different costs: autoregressive models generated pixels sequentially, while variational models often traded sample sharpness against a tractable objective. What was missing was a stable supervised target that still learned a rich multimodal image distribution.

### What it enabled

- A non-adversarial image-generation recipe trained by regression against known synthetic noise.
- The modern diffusion lineage: improved DDPMs, classifier and classifier-free guidance, latent diffusion, Stable Diffusion, Imagen, and diffusion-based video and audio systems.
- Explicit control over the fidelity–diversity path through iterative denoising and conditioning at multiple noise levels.
- A new systems bottleneck: sample quality was excellent, but generation required hundreds or thousands of sequential network evaluations. DDIMs, learned schedules, distillation, latent-space diffusion, and later consistency/flow methods became necessary to reduce that latency.

## Why It Matters

DDPM is the hinge that explains why modern image generation did not remain a GAN story.

GANs learned through competition: the generator improved only through the discriminator’s current judgment. DDPM changed the learning problem. The corruption is known, the target noise is known, and every training example supplies a direct regression signal at a randomly selected noise level. A hard global generation problem is decomposed into many local denoising problems that share one network.

That does not make generation free. The same decomposition that stabilizes learning makes sampling serial and expensive: a model trained to reverse tiny corruption steps must execute many of them. The breakthrough therefore creates its own next necessity. Once diffusion solved the quality-and-stability problem, the frontier shifted to reducing sampling steps, moving denoising into compressed latent spaces, and adding guidance without sacrificing diversity.

The deeper lineage lesson is that generative AI advances by changing *where the difficulty lives*. GANs put it in a two-player optimization game. DDPMs put it in a long reverse trajectory. That trade produced a training objective simple and stable enough to scale—and the dominant foundation for the next wave of visual generative AI.

## Lineage

- **Built on:** diffusion probabilistic models, variational inference, denoising score matching, Langevin dynamics, U-Nets, and backpropagation
- **Repairs:** adversarial instability and mode-coverage fragility while retaining high perceptual sample quality
- **Preserves:** stochastic generation from noise and deep convolutional image priors
- **Leads toward:** DDIM, guidance, latent diffusion, Stable Diffusion, diffusion Transformers, and faster consistency/flow-based generation

---

*GANs asked whether a fake could fool a judge. Diffusion asked whether noise could remember the way home.*
