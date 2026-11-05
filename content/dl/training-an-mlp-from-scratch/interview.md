# Interview notes — Training an MLP from scratch

## Q1. Walk through one reproducible two-layer training step.

**60-second answer.** I fix initialization, architecture, and the exact scalar objective first. I use a local seeded generator for fan-in-scaled weights and zero biases, then tanh hidden activations and a linear output. The loss is half-mean-squared error. I derive all gradients from one checkpoint, apply simultaneous SGD, and record the new loss. Loss history includes the initial state.

**Follow-ups.** Why update simultaneously? An early output-weight update corrupts the hidden gradient. Why seed locally? Initialization remains reproducible without changing unrelated randomness.

## Q2. The loss is flat or increasing. What do you inspect first?

**60-second answer.** I reproduce one step and check target shape, loss normalization, and all parameter gradients against finite differences. Then I inspect saturation and gradient magnitudes before tuning the learning rate. A flat hidden weight can be expected for zero input; biases may still learn. A large step can increase the objective even with correct derivatives.

**Follow-ups.** What belongs in a resume checkpoint? Parameters, optimizer state if any, step count, and sampling RNG state. Is monotone loss guaranteed? No, except for specific objectives and suitable step-size conditions.

## Common mistakes

- Skipping the scalar objective or the lifecycle convention.
- Checking only a singleton example and missing shared contributions.
- Trusting a training curve instead of an independent gradient check.

Source: [book: Goodfellow, Bengio and Courville, Deep Learning, chapters 6 and 8]. See lesson.md for the derivation and explicit conventions.
