# Interview notes — Backprop through a dense layer

## Q1. Derive a dense layer backward pass and explain the batch reduction.

**60-second answer.** I start with shapes and the scalar loss convention. For Y=XW+b and upstream G, dX=GW.T, dW=X.TG, and db sums G over rows. The weight and bias are shared across examples, so their contributions add. I average only if the caller has not already averaged the upstream signal. The exercise exposes this decision explicitly.

**Follow-ups.** Why no full Jacobian? The VJP contracts the needed sensitivity directly. What about a singleton batch? The same equations apply; it just cannot expose a missing mean factor.

## Q2. How would you validate these gradients without trusting your derivation?

**60-second answer.** I build a scalar objective by taking the inner product of output and a fixed upstream matrix. I perturb every small input, weight, and bias entry in turn and compare central differences with backward results. A separate scalar-loop oracle checks orientation. I include a non-square shape so transposes cannot hide.

**Follow-ups.** Why fix G? It is the supplied sensitivity, not another function of the perturbed variables. Which tolerance? Compare with a moderate perturbation and declared absolute/relative tolerances; inspect cancellation before changing them.

## Common mistakes

- Dividing by batch size when the supplied upstream gradient was already averaged.
- Checking only a singleton example and missing shared-parameter contributions.
- Trusting a training curve instead of an independent gradient check.

Source: [book: Goodfellow, Bengio and Courville, Deep Learning, chapter 6.5]. See lesson.md for the derivation and explicit conventions.
