# Interview notes — AdamW decoupled weight decay

## Q1. What is the difference between L2 regularisation and weight decay in Adam?

**60-second answer.** For plain SGD they are the same once the coefficient is rescaled by the learning rate. In Adam, L2 adds $\lambda\theta$ to the gradient, and then the adaptive scale $\sqrt{\hat v}$ divides that term along with the loss gradient. So weights with large loss gradients are decayed less, and weights with small gradients are decayed more. AdamW multiplies the weights by $1-\eta\lambda$ and leaves the adaptive step on the loss gradient alone. Every weight then decays at the same rate.

**Follow-ups.** Why is the zero-gradient case useful? With zero loss gradient, Adam with L2 moves each weight by about the learning rate, so a small weight can cross zero. AdamW shrinks it by a fixed fraction. Which one does PyTorch's `AdamW` use? The decoupled one: it multiplies by $1-\text{lr}\cdot\text{weight decay}$ before the Adam update.

## Q2. Which parameters should you exclude from weight decay, and how?

**60-second answer.** I exclude biases and one-dimensional norm gains. Decaying them rarely helps and can shift the normalisation. I do it with parameter groups: one group with the weight decay for matrices and convolution kernels, and one group with zero for the rest. The optimizer then gets two groups, and I check the membership by name before training.

**Follow-ups.** Is this from the paper? No. The AdamW paper does not decide which parameters to exclude. It is a common convention, so I state it as a convention. What is the risk? A wrong group layout silently changes the regularisation, and resumed runs must keep the same layout.

## Common mistakes

- Using `weight_decay` in Adam and expecting AdamW's regularisation.
- Decaying biases and norm gains by default.
- Forgetting that the decay per step is $\eta\lambda$, so a learning-rate change also changes the decay.

Source: [corpus:library-papers/1711.05101.pdf.md p.3]. See lesson.md for the derivation and the zero-gradient example.
