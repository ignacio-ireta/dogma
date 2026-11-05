# ML Concepts — Mnemonics & Memory Anchors

*For each concept: what it is, how to remember it, and the etymology/reasoning behind the name.*

---

## 🎯 Metrics (The Bouncer Framework)

Think of your model as a **bouncer at a club**. VIPs = actual positives. Randos = actual negatives.

| Metric | Question | Mnemonic |
|--------|----------|----------|
| **Precision** | "When I let someone in, are they actually a VIP?" | **P**recision = **P**urity of your YES predictions |
| **Recall** | "Did I let ALL the VIPs in?" | **R**ecall = did you **R**emember all the positives? |
| **F1** | "Balance of precision & recall" | **F**airly balances both — harmonic mean punishes if either sucks |
| **Accuracy** | "How often am I right overall?" | Only trustworthy when classes are balanced. 99.7% accuracy on fraud = 🤡 |
| **AUC-ROC** | "Can my model separate YES from NO at ALL thresholds?" | **ROC = Receiver Operating Characteristic** (WWII radar operators deciding "is that a plane or noise?"). AUC = how much of the chart you fill. 1.0 = perfect, 0.5 = coin flip |

**ROC plots:** TPR (y) vs FPR (x). NOT precision vs recall (that's a PR curve — different thing).

**When to choose which:**
- Missing positives is catastrophic → **Recall** (fraud, cancer)
- False alarms are expensive → **Precision** (collection calls, surgery)
- Need balance → **F1**
- Comparing models overall → **AUC-ROC**

---

## 🧠 Parameters vs Hyperparameters

**Etymology:**
- **Parameter** = Greek *para* (alongside) + *metron* (measure). Values measured FROM the data. The model discovers them.
- **Hyper** = Greek for "above, beyond." **Hyperparameter** = settings ABOVE the parameters. You're the boss, you set these before training.

**Mnemonic:** *"The model learns parameters. You choose hyperparameters. Hyper = you're the boss."*

| | Parameter | Hyperparameter |
|---|---|---|
| Who sets it? | The model (during training) | You (before training) |
| Examples | Weights, coefficients, split thresholds | learning_rate, max_depth, n_estimators |
| Changes during training? | Yes | No |

⚠️ **Feature ≠ Parameter.** A feature is a column in your dataset. A parameter is what the model learns internally.

---

## 🌲 Bagging vs Boosting

**Etymology:**
- **Bagging** = **B**ootstrap **Agg**regat**ing**. Bootstrap = random sampling with replacement. Aggregating = averaging results.
- **Boosting** = each model "boosts" the performance by fixing what the last one got wrong.

| | Bagging | Boosting |
|---|---|---|
| How | Parallel, random subsets | Sequential, learn from mistakes |
| Fixes | Variance (overfitting) | Bias (underfitting) |
| Example | Random Forest | XGBoost, AdaBoost, LightGBM |
| Risk | Can underfit | Can overfit |

**Mnemonic:** *"Bagging = bags of random samples, averaged. Boosting = boost the weak spots, one by one."*

---

## 🔒 L1 vs L2 Regularization

**Etymology:**
- **Regularization** = making the model more "regular" (less extreme weights). From Latin *regula* = rule. You're imposing rules on the model.
- **L1/L2** = L refers to the "Lp norm." L1 = sum of absolute values. L2 = sum of squared values.
- **Lasso** = **L**east **A**bsolute **S**hrinkage and **S**election **O**perator (it selects features by killing them)
- **Ridge** = named because the penalty creates a "ridge" on the loss surface that guides optimization

| | L1 (Lasso) | L2 (Ridge) |
|---|---|---|
| Penalty | \|w\| (absolute) | w² (squared) |
| Effect | Zeros out features | Shrinks all features |
| Use when | Feature selection needed | All features might matter |

**Mnemonic:** *"L**1** = **1**ess features (kills them). L**2** = shrinks them **2**wards zero (but keeps them all)."*

⚠️ **Regularization ≠ Normalization.** Regularization = penalty during TRAINING. Normalization = scaling features during PREPROCESSING. Totally different.

---

## 📉 Bias-Variance Tradeoff

**Etymology:**
- **Bias** = systematic error. From Old French *biais* = slant/deviation. The model consistently slants in the wrong direction.
- **Variance** = sensitivity to training data. How much does your model change if you swap out the training set?

```
Simple model ←————————————→ Complex model
High bias                    Low bias
Low variance                 High variance
Underfitting                 Overfitting
```

**Mnemonic:** *"Bias = consistently wrong. Variance = inconsistently wrong."*

**Detection:**

| Training score | Validation score | Diagnosis |
|---|---|---|
| Both high, close | 95% / 93% | ✅ Good fit |
| Train high, val low | 99% / 72% | 🔴 Overfitting (high variance) |
| Both low | 61% / 58% | 🔴 Underfitting (high bias) |

**Mnemonic:** *"Overfit = aces homework, bombs exam. Underfit = fails both."*

---

## 🔄 Cross-Validation

**Etymology:** *Cross* = across, cutting across. You cut across the data in different ways. Each fold takes a turn being the test set.

**Why it beats a single split:** One unlucky split could fool you. K splits averages out the luck.

**Stratified k-fold:** Preserves class ratios in each fold. Use when classes are imbalanced (fraud, churn).

---

## 📊 Loss Function vs Evaluation Metric

**Mnemonic:** *"Loss = steering wheel (guides training). Metric = speedometer (measures performance)."*

They're often different! XGBoost minimizes log loss but you report AUC-ROC.

---

## 🧮 Confusion Matrix

```
              Predicted
              +      -
Actual  +    TP     FN   ← "Actual positives row"
        -    FP     TN   ← "Actual negatives row"
```

**Mnemonic for FP vs FN:**
- **F**alse **P**ositive = "False alarm" (said yes, was no)
- **F**alse **N**egative = "Missed it" (said no, was yes)

---

## 🎲 Activation Functions (Neural Nets)

| Function | What it does | When |
|----------|-------------|------|
| **Sigmoid** | Squashes to (0,1) | Binary output. "Probability" |
| **Softmax** | Probability distribution across classes | Multi-class output |
| **ReLU** | max(0,x) — kills negatives | Hidden layers. Default choice. |

**Etymology:**
- **Sigmoid** = Greek *sigma* (S-shape). The curve looks like an S.
- **ReLU** = **Re**ctified **L**inear **U**nit. "Rectified" = made non-negative (like rectifying AC current to DC — only keeps the positive part).

---

## 📐 MSE vs MAE

- **MSE** (Mean Squared Error) = squares the errors → big errors get punished MORE. Sensitive to outliers.
- **MAE** (Mean Absolute Error) = absolute errors → all errors punished equally. Robust to outliers.

**Mnemonic:** *"Squared = scared of outliers. Absolute = absolutely chill."*

---

## 🌪️ Curse of Dimensionality

Adding more features isn't always better. More dimensions = data becomes sparse = model needs exponentially more data to generalize. Plus: overfitting risk, spurious correlations, training time.

**Mnemonic:** *"More columns, more problems."*

---

## 🔧 Overfitting Fixes Toolkit

| Fix | Why it works |
|-----|-------------|
| More training data | More examples = harder to memorize |
| Regularization (L1/L2) | Penalty on complexity |
| Feature selection | Remove noise features |
| Early stopping | Stop before memorization kicks in |
| Cross-validation | Detect it reliably |
| Reduce model complexity | Fewer trees, shallower depth |

---

## 🏗️ Class Imbalance Toolkit

| Technique | What it does |
|-----------|-------------|
| `scale_pos_weight` (XGBoost) | Weight minority class higher in loss |
| `class_weight="balanced"` (sklearn) | Same idea, sklearn style |
| SMOTE | Synthetic oversampling of minority class |
| Undersampling | Remove majority class examples |
| Adjust threshold | Move decision boundary (default 0.5 may not be optimal) |

---

*Review these the morning of the test. 5 minutes with coffee. Then trust the prep.* 🏔️
