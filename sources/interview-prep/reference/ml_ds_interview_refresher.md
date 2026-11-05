# ML/DS Interview Refresher — Targeted Study Guide

*Targeted study guide for ML and data-science interviews. Each section ends with a one-line interview version.*

---

## 🔴 RED ZONES — Study These First

### AUC-ROC (Not the Same as Precision-Recall Curve)

**What it is:** A graph of **True Positive Rate** (recall) vs **False Positive Rate** at every possible classification threshold.

- **TPR (y-axis):** Of all actual positives, how many did we catch? = TP / (TP + FN) — this is just recall.
- **FPR (x-axis):** Of all actual negatives, how many did we accidentally flag? = FP / (FP + TN)

**AUC** = Area Under that Curve. Ranges from 0 to 1.
- **1.0** = perfect separation (model never confuses positives and negatives)
- **0.5** = coin flip (the diagonal line — model is guessing)
- **< 0.5** = worse than random (model is inverted somehow)

**The one-liner for interviews:** "AUC tells you the probability that your model ranks a random positive case higher than a random negative case."

**In practical terms:** If you pick one random non-paying customer and one random paying customer, AUC 0.92 means there's a 92% chance the model assigns a higher risk score to the non-payer.

**AUC ≠ Accuracy.** AUC is threshold-independent (it considers ALL thresholds). Accuracy is measured at ONE specific threshold. A model with AUC 0.92 could have 80% accuracy or 95% accuracy depending on where you set the cutoff.

**When someone says "so it's 92% accurate?"** → "No. AUC 0.92 means the model is excellent at *ranking* — separating risky customers from safe ones. The actual accuracy depends on where we set the decision threshold, which is a business decision about how many false alarms you can tolerate."

---

### Bagging vs Boosting

Both are **ensemble methods** — they combine multiple models to get better performance than any single model.

**Bagging (Bootstrap AGGregating):**
- Train multiple models on **random subsets** of the data (with replacement)
- Train them **in parallel** — they don't know about each other
- Combine by **averaging** (regression) or **voting** (classification)
- **Reduces variance** (overfitting)
- Classic example: **Random Forest** = bagged decision trees

Think: a committee where everyone independently studies different chapters, then votes.

**Boosting:**
- Train models **sequentially** — each one focuses on the **mistakes** of the previous
- Each new model gets a weighted dataset where previously misclassified samples weigh more
- Combine by **weighted sum** of all models
- **Reduces bias** (underfitting)
- Classic examples: **AdaBoost, Gradient Boosting, XGBoost**

Think: a student who retakes the exam, but only studies the questions they got wrong last time.

**XGBoost = eXtreme Gradient Boosting.** It's boosting. A production model is a sequential ensemble where each new tree corrects the errors of all previous trees. The "gradient" part means it uses gradient descent on the loss function to figure out what "correcting errors" means mathematically.

| | Bagging | Boosting |
|--|---------|----------|
| Training | Parallel | Sequential |
| Focus | Reduce variance | Reduce bias |
| Error handling | Each model independent | Each model fixes previous errors |
| Risk | Can still underfit | Can overfit if too many rounds |
| Example | Random Forest | XGBoost |

---

### L1 vs L2 Regularization

**First: what regularization IS.**
Regularization prevents overfitting by **adding a penalty for model complexity to the loss function**. It tells the model: "don't just minimize prediction error — also keep your weights small."

It is NOT normalization/standardization (scaling features to the same range). That's preprocessing. Regularization is a training constraint.

**L1 Regularization (Lasso):**
- Adds **sum of |weights|** to the loss function
- Pushes some weights to **exactly zero**
- Effect: **feature selection** — the model kills irrelevant features entirely
- Result: sparse model (few non-zero weights)
- Use when: you suspect many features are junk and want the model to pick winners

**L2 Regularization (Ridge):**
- Adds **sum of weights²** to the loss function
- Pushes weights **toward zero but never exactly zero**
- Effect: **dampens all features** — none dominate, none die
- Result: dense model (all features contribute, just less aggressively)
- Use when: most features are useful but you want to prevent any one from being too influential

**Memory trick:**
- **L1 = Lasso = Lassos the useless features and kicks them out** (sparse, zeros)
- **L2 = Ridge = Ridges everything down evenly** (dense, small)

**In XGBoost:** Both are available as hyperparameters:
- `reg_alpha` = L1 regularization
- `reg_lambda` = L2 regularization (default = 1, so L2 is ON by default!)

XGBoost uses L2 regularization by default, and you probably didn't even know it.

**Regularization vs Normalization vs Standardization:**

| Term | What it does | When |
|------|-------------|------|
| **Regularization** | Penalizes model complexity during training | Training time — added to loss function |
| **Normalization** | Scales features to [0, 1] range | Preprocessing — before training |
| **Standardization** | Scales features to mean=0, std=1 | Preprocessing — before training |

---

### Parameter vs Hyperparameter

**Parameter:** Values the model **learns during training**. You never set these. The algorithm discovers them.
- Tree split points in XGBoost
- Weights in a neural network
- Coefficients in linear regression

**Hyperparameter:** Values **you set before training**. They control how the model learns.
- `max_depth` (how deep can each tree grow)
- `learning_rate` (how much each new tree contributes)
- `n_estimators` (how many trees to build)
- `reg_lambda` (L2 regularization strength)

**Memory trick: Parameters = the model decides. Hyperparameters = you decide.**

In practice, when you run `xgb.train()`, XGBoost discovers all the tree split points (parameters). But YOU told it `max_depth=6, learning_rate=0.1, n_estimators=100` (hyperparameters) before it started.

**Hyperparameter Tuning** = systematically trying different hyperparameter combinations to find the best performance. This is what SageMaker HPO does.

---

### XGBoost's Default Loss Function

For binary classification: **log loss** (binary cross-entropy).

Formula intuition: it measures how far your predicted *probability* is from the true label (0 or 1). If the true label is 1 and you predict 0.99, log loss is tiny. If you predict 0.01, log loss is huge.

- Gini and AUC-ROC are **evaluation metrics** — they tell you how good the model is AFTER training
- Log loss is the **loss function** — it's what the model optimizes DURING training
- The "gradient" in gradient boosting = the gradient of the log loss function

**In practice:** A model minimizes log loss during training, but you EVALUATE it with Gini/AUC-ROC. These are different things with different purposes. The loss function is the steering wheel; the evaluation metric is the speedometer.

---

## 🟡 YELLOW ZONES — Tighten These Up

### F1 Score — The "Weakest Link" Average

F1 = 2 × (Precision × Recall) / (Precision + Recall)

It's the **harmonic mean** — dominated by the smallest value.

| Precision | Recall | Regular Average | F1 (Harmonic Mean) |
|-----------|--------|-----------------|---------------------|
| 99% | 1% | 50% | 1.98% |
| 80% | 80% | 80% | 80% |
| 90% | 70% | 80% | 78.9% |

F1 punishes imbalance. You can't hide a weakness behind a strength.

**When to use F1:** When you care about both precision AND recall and need a single number. Especially useful with imbalanced classes (like churn or default models, with about 6% positives vs 94% negatives).

**When NOT to use F1:** When one matters more than the other. Then use F-beta: Fβ = (1+β²) × (P×R) / (β²×P + R). β > 1 weights recall more, β < 1 weights precision more.

---

### Precision & Recall — A Worked Example

**Setting:** 1000 customers. 100 won't pay. Model flags 80 as "won't pay." 60 of those flags are correct.

|  | Actually won't pay | Actually will pay |
|--|-------------------|------------------|
| **Model says "won't pay"** | 60 (TP) | 20 (FP) |
| **Model says "will pay"** | 40 (FN) | 880 (TN) |

- **Precision** = 60/80 = **75%** → "Of everyone we flagged for collections, 75% were right"
- **Recall** = 60/100 = **60%** → "Of all actual deadbeats, we caught 60%"
- **Accuracy** = (60+880)/1000 = **94%** → sounds great but hides the 40 missed deadbeats
- **F1** = 2 × (0.75 × 0.60) / (0.75 + 0.60) = **66.7%** → more honest picture

**The business question:** Those 40 missed non-payers (FN) — what do they cost? If each represents $10K in lost revenue, that's $400K walking out the door. Those 20 false alarms (FP) — what do they cost? Maybe $500 each in wasted collection calls = $10K. In this case: **recall matters way more**. Tune the threshold to catch more deadbeats even if you annoy some good customers.

---

### Bias-Variance Tradeoff — Crisp Version

**Bias** = error from the model being too simple. A straight line through curved data. It systematically *misses* real patterns. **Underfitting.**

**Variance** = error from the model being too sensitive to training data. A wiggly line that hits every point perfectly but breaks on new data. **Overfitting.**

```
High Bias (Underfitting)          Sweet Spot              High Variance (Overfitting)
    ___________                   .  . .                    .
   /                             / \/ \                   /\/\/\/\/\
  /    . . .  .                 /  .    \  .             / . .  . .  \
 /  .     .                    / .    .  \              /.  .   .  .  .\
```

**The tradeoff:** Making a model more complex → reduces bias, increases variance. Simplifying → reduces variance, increases bias. You want the minimum TOTAL error.

**In practice:** If your XGBoost has `max_depth=1` (stumps) → high bias, underfitting. If `max_depth=50` → high variance, overfitting. The right `max_depth` is somewhere in between. That's what hyperparameter tuning finds.

**Interview version:** "A model with high bias is too rigid — it misses real patterns. A model with high variance memorizes noise — it breaks on new data. The goal is the complexity sweet spot where total error is minimized."

---

### Cross-Validation — Sharper Version

**Problem:** A single train/test split might be lucky or unlucky. Maybe all the easy cases ended up in test.

**Solution:** K-Fold Cross-Validation.
1. Split data into k folds (usually 5 or 10)
2. Train on k-1 folds, test on the remaining 1
3. Rotate: every fold gets to be the test set exactly once
4. Average the scores across all k runs

**Result:** A more reliable performance estimate with a confidence range.

**In practice:** With 10M records and 5-fold CV, each fold tests on 2M records. You get 5 AUC scores. If they're all ~0.91-0.93, you're confident. If they're 0.60-0.98, your model is unstable — something's wrong.

**Stratified K-Fold:** Preserves the class ratio in each fold. Critical for imbalanced data like churn or default (about 6% positives). Without stratification, some folds might have 3% and others 9%.

---

### Gradient Descent — One Level Deeper

**What it does:** Finds the model parameters that minimize the loss function by iteratively following the steepest downhill direction.

**The analogy:** You're blindfolded on a hill. You feel the slope under your feet (= compute the gradient). You take a step downhill (= update parameters). Repeat until flat ground (= minimum loss).

- **Learning rate** = how big each step is. Too big → you overshoot. Too small → takes forever.
- **Local minimum** = a valley that isn't the deepest valley. Gradient descent can get stuck here.
- **Stochastic GD** = compute gradient on random subsets (batches) instead of all data. Noisier but faster and can escape local minima.

**In XGBoost specifically:** Each new tree is built to predict the *gradient* (direction of steepest descent) of the loss function. That's why it's called gradient *boosting* — each tree takes one step downhill.

---

### Activation Functions — The Ones That Matter

| Function | Output Range | Use Case | Why |
|----------|-------------|----------|-----|
| **Sigmoid** | (0, 1) | Binary classification output layer | Squashes to probability |
| **Softmax** | (0, 1), sums to 1 | Multi-class output layer | Probability distribution across classes |
| **ReLU** | [0, ∞) | Hidden layers (default) | Fast, avoids vanishing gradient. Kills negatives. |
| **Tanh** | (-1, 1) | Hidden layers (when negatives matter) | Zero-centered, but can saturate |

**ReLU** is the workhorse. Use it as default for hidden layers. It's just `max(0, x)` — dead simple, trains fast.

**Sigmoid** is for the final layer of binary classification — it gives you a probability between 0 and 1. "This customer has a 73% chance of not paying."

**Softmax** is sigmoid's multi-class cousin. "40% cat, 35% dog, 25% hamster."

**You probably won't need activation functions for a classical-ML case interview** — they matter for neural networks, and the conversation will likely stay on classical ML and business thinking. But if asked, lead with sigmoid (binary output) and ReLU (hidden layers).

---

## 🎯 Quick-Fire Flashcards for Morning Review

**Precision?** Of everything we flagged, how many were right? TP/(TP+FP)

**Recall?** Of all actual positives, how many did we catch? TP/(TP+FN)

**F1?** Weakest-link average of precision and recall. Punishes imbalance.

**AUC-ROC?** Probability that model ranks a random positive above a random negative. Threshold-independent.

**Loss function?** What the model minimizes during training. XGBoost default: log loss.

**Evaluation metric?** How you measure model quality after training. Gini, AUC, F1.

**Bagging?** Parallel models on random subsets → average. Reduces variance. Random Forest.

**Boosting?** Sequential models that fix previous mistakes. Reduces bias. XGBoost.

**Bias?** Model too simple → misses patterns → underfitting.

**Variance?** Model too complex → memorizes noise → overfitting.

**L1 (Lasso)?** Kills features. Sets weights to exactly zero. Feature selection.

**L2 (Ridge)?** Shrinks features. Weights small but never zero. Dampens everything.

**Regularization?** Penalty on model complexity added to loss function. Prevents overfitting.

**Normalization?** Scaling features to same range. Preprocessing. NOT regularization.

**Parameter?** Model learns it. Tree splits, weights, coefficients.

**Hyperparameter?** You set it. max_depth, learning_rate, n_estimators.

**Cross-validation?** K-fold: rotate test set, average scores. Reliable performance estimate.

**Gradient descent?** Iteratively follow steepest downhill to minimize loss. Learning rate = step size.

---

