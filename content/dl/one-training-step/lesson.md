---
title: "One training step"
minutes: 12
min_viable: "Read The system and Worked example, then solve p01 using Hint 1 if needed."
---

## The system

Every gradient-based training loop, however large, runs the same five lines for each batch.

```python
optimizer.zero_grad()  # 1. clear the gradients left by the last step
prediction = model(x)  # 2. forward pass on the current parameters
loss = loss_fn(prediction, y)  # 3. one scalar loss for this batch
loss.backward()  # 4. fill each parameter's .grad with d(loss)/d(parameter)
optimizer.step()  # 5. move each parameter by -lr * .grad
```

Notation: $\theta$ is the parameter vector, $\mathcal{B}=(x,y)$ is the batch, and $L(\theta;\mathcal{B})$
is the scalar loss. One step computes $g=\nabla_\theta L(\theta;\mathcal{B})$ and sets
$\theta\leftarrow\theta-\eta g$, where $\eta$ is the learning rate. Each line exists to make one of
these quantities correct.

| Line | Bug it prevents if it is missing or moved |
|---|---|
| 1 `zero_grad` | `.grad` keeps adding across steps, so the update uses every past gradient |
| 2 forward | the loss would describe parameters you did not just compute |
| 3 loss | a non-scalar output cannot be a gradient target without an explicit vector |
| 4 `backward` before update | the update sees no fresh gradient, or the previous step's |
| 5 update under `no_grad` | the update itself is recorded in the graph, and an in-place write to a leaf fails |

## Worked example

Take one parameter $w$ and one example $(x,y)=(1,0)$ with loss $L=(wx-y)^2$. Start at $w=2$ with
learning rate $\eta=0.1$.

1. Clear: $w.\text{grad}=\text{None}$.
2. Forward: $\hat y = 2\cdot 1 = 2$.
3. Loss: $L=(2-0)^2=4$.
4. Backward: $\partial L/\partial w = 2(\hat y-y)\,x = 4$. Now $w.\text{grad}=4$.
5. Update: $w\leftarrow 2-0.1\cdot 4=1.6$.

Return $4$, the loss computed in line 3, not the loss at $w=1.6$. The second call on the same batch
starts again at line 1. It gives $(1.6-0)^2=2.56$ at step 3, gradient $3.2$, and $w=1.28$. Without
line 1, the second gradient would be $4+3.2=7.2$ and $w$ would fall to $0.88$. That is the stale-gradient
bug, and the test that catches it is calling the step twice on the same batch.

## Why it works

Backward computes the gradient of the loss at the parameters that produced it. The update therefore
has to use the values from the same forward pass, before any parameter changes. That is why
the order is fixed: a gradient is valid only for the parameters that were current when the forward
pass ran. Plain stochastic gradient descent is the update $\theta\leftarrow\theta-\eta g$ on noisy
gradient estimates from batches; the classical stochastic-approximation result is in
[library:AI-ML/03-Optimization/Robbins Monro Stochastic Approximation 1951.md]. The update runs under
`no_grad` because it is not part of the function being differentiated. Recording it would grow the graph
on every step and would make the next backward pass differentiate through the update.

## Signal vs. noise

Signal: the five lines, their order, and which quantity each one fixes. Noise: which optimizer
class you call, since `torch.optim.SGD` performs exactly line 5 for plain SGD with no momentum. Learn
the manual version first so you can read any optimizer's `step()`. Momentum, weight decay and
schedules change line 5 and nothing else.

## Where it breaks

- **Stale gradients.** If you clear after `backward`, the update uses zeros or `None`, and the model
  silently stops learning. Clear before the forward pass.
- **Update outside `no_grad`.** On a leaf that requires grad, the in-place write raises an error. The
  wrong fix is `p.data`, which hides the problem while still allowing mistakes elsewhere.
- **Replacing parameters.** Creating a new `Parameter` breaks any optimizer or dictionary that held
  the old one. Update in place.
- **Reporting the wrong loss.** Returning the loss after the update makes the training curve lag a
  step behind. Keep the number from line 3.
- **Non-scalar loss.** `backward()` on a vector needs an explicit gradient argument. Reduce to a scalar
  first, usually with a mean.

## Interview angle

Give the five lines in order and say what each guarantees: "clear, forward, scalar loss, backward, then
step under no-grad." Name the two expensive bugs, stale gradients and tracked updates, and say how you
detect each: call the step twice on one batch and compare with a reference optimizer. Follow-up:
"why not update inside backward?" Because the gradient of the loss at the old parameters must be
complete before any parameter moves, and other parameters in the same graph still need their old values.

## Connections

This topic builds on `dl/gradients-with-framework-autograd`, which covers the `.grad` mechanics used
here, and on `dl/training-an-mlp-from-scratch`, which derived the same update by hand. The next step is
`dl/a-complete-training-loop`, which repeats this step over epochs and mini-batches, adds evaluation,
and checkpoints the best state.
