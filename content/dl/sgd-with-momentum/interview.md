# Interview notes — SGD with momentum

## Q1. Write the momentum update and explain what the buffer remembers.

**60-second answer.** The buffer is a decayed running sum of gradients. For each parameter tensor I keep v, update v = mu times v plus g, and move the parameters by lr times v. Old directions decay geometrically with mu, so components that keep their sign accumulate and oscillating components cancel. The first step seeds the buffer with the raw gradient. Effective step size with a constant gradient is lr times g over one minus mu, which is why momentum can destabilise a learning rate that worked without it.

**Follow-ups.** What is the first-step rule? The first buffer is the gradient itself, so the first move is exactly lr times g. Why does it matter? A learner who damps the first buffer by one minus mu produces a step that is too small and silently changes early training.

## Q2. Why do two momentum implementations sometimes give different training runs?

**60-second answer.** There are two conventions. PyTorch keeps the buffer free of the learning rate and applies lr at the update. The other common form stores lr inside the buffer. With a constant learning rate they produce the same parameters, because one buffer is lr times the other. When the schedule changes the learning rate, the PyTorch form applies the new rate to the whole accumulated direction, while the Sutskever form keeps old contributions at their old scale. So the two can diverge after a schedule change or when a checkpoint is resumed with a different convention.

**Follow-ups.** How do you check which convention a library uses? Inspect the buffer after one step with a known gradient. How does Nesterov differ? It applies the gradient plus mu times the new buffer, which is a look-ahead; it is not just a different buffer.

## Common mistakes

- Damping the first buffer by one minus mu, which shrinks the first step.
- Adding weight decay after the momentum buffer instead of to the gradient before it.
- Resuming training with zero buffers and expecting an identical trajectory.

Sources: the derivation is in lesson.md. The Goodfellow et al. chapter on optimisation is a general reference; its section number was not checked against the book text.
