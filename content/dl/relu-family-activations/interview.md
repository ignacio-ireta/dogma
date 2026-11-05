# Interview notes — ReLU-family activations

## Q1. Why might you choose a leaky rectifier?

**60-second answer.** Ordinary ReLU passes positive inputs and blocks negative inputs and their local gradients. A fixed positive negative slope keeps a gradient path on the negative side. I would consider it when inactive units are a concern, while checking scale and optimization too. It changes a particular failure mode; it does not guarantee better generalization or eliminate every source of vanishing gradients.

**Follow-ups.**

- What derivative do you use at zero? This exercise fixes zero and states the convention explicitly.
- How does PReLU differ? Its negative slope is learned, so it introduces parameter gradients.

## Q2. How would you check this activation backward pass?

**60-second answer.** I compare values and local slopes with the sign-region table, then use central differences at points away from zero. A finite difference crossing the kink mixes both slopes and cannot validate a selected derivative convention. I also check scalar and empty shapes and avoid aliasing the input because backward code may still need the original activations.

**Follow-ups.**

- What if an upstream gradient is supplied? Multiply it elementwise by the local derivative.
- What does slope one mean here? The forward map is identity, but the exercise still chooses zero derivative at zero.

## Common mistakes

- Returning the activation as its derivative.
- Forgetting the negative slope when computing local gradients.
- Using a finite difference at the kink to decide a convention.
