# Optimizer Failure Modes — Interview Bridge

*Use the optimizer lineage as a diagnostic tree, not a list of algorithms.*

**← [[../_MOC|AI/ML Lineage]]**

## The Interview Move

When asked why training is unstable or slow, do not jump straight to “try Adam.” First identify which assumption has failed:

```text
Is the objective itself wrong?
  → Cauchy: define one scalar measure of error that matches the task.

Is the update direction too noisy?
  → Robbins-Monro: minibatch gradients are estimates; control variance and step size.

Is progress oscillating across a narrow valley?
  → Polyak: preserve useful direction with momentum instead of restarting each step.
```

This gives a stronger answer than naming optimizers because it connects the observed symptom to the mechanism.

## Reading the Training Curve

| Symptom | Likely issue | First checks |
|---|---|---|
| Training loss falls, business metric does not | Objective mismatch | Loss definition, label quality, class costs, validation metric |
| Loss is erratic or diverges | Noisy updates or excessive step size | Learning rate, batch size, gradient clipping, data outliers |
| Loss improves but zig-zags slowly | Ill-conditioning | Feature scaling, normalization, momentum, adaptive methods |
| Training improves while validation worsens | Generalization, not optimization | Leakage, regularization, early stopping, split design |

The last row is the important boundary: an optimizer can reduce the training objective perfectly and still solve the wrong practical problem.

## Interview-Style Answer

> “I would diagnose the failure before changing the optimizer. First I would verify that the loss represents the business objective. Then I would inspect whether minibatch noise or the learning rate is making updates unstable. If the curve shows slow oscillation rather than random variance, I would suspect conditioning and test scaling or momentum. I would keep training and validation behavior separate, because a lower training loss does not guarantee better generalization or business value.”

## Source Paths

- `AI-ML/03-Optimization/Cauchy Steepest Descent 1847.md`
- `AI-ML/03-Optimization/Robbins Monro Stochastic Approximation 1951.md`
- `AI-ML/03-Optimization/Polyak Heavy-Ball Momentum 1964.md`
- `AI-ML/05-Classification/XGBoost Interview Bridge.md`

---

*The senior answer is not “which optimizer is best?” It is “what evidence says the update process is the problem?”*
