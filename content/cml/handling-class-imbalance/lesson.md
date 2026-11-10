---
title: Handling class imbalance
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

When positives are rare, decide in this order.

1. **Metric.** Choose it from the question. Accuracy is the wrong yardstick. Use average precision compared with the prevalence, or recall at a precision floor.
2. **Threshold.** Set it from the costs of each error. With calibrated probabilities the rule is a single number (see Why it works).
3. **Training.** Change the training data or loss only if the model lacks signal on the minority class. Class weights and resampling both change the base rate the model learns, so the probabilities they produce need correcting before use.

| Method | What it changes | Probabilities afterwards |
|---|---|---|
| Threshold moving | the decision only | unchanged, still calibrated |
| Class weights | the loss | shifted toward the reweighted base rate |
| Random over- or undersampling | the data | shifted toward the resampled base rate |
| SMOTE | the data, with synthetic points | shifted, and it adds a geometric assumption |

Balanced class weights use $w_c=n/(k\,n_c)$, where $n$ is the sample size, $k$ the number of classes and $n_c$ the count of class $c$. They equalise weighted sample counts, not the realised loss, which also depends on each sample's prediction error.

## Worked example

Costs: a missed fraud costs $c_{FN}=10$, and a false alarm costs $c_{FP}=1$. The threshold is $c_{FP}/(c_{FP}+c_{FN})=1/11=0.091$.

- Calibrated $p=0.20$. Flagging costs $(1-0.20)\cdot1=0.80$, and not flagging costs $0.20\cdot10=2.00$. Flag.
- Calibrated $p=0.05$. Flagging costs $0.95$, and not flagging costs $0.50$. Do not flag.

Balanced weights for counts $(5,2,2)$ with $n=9$ and $k=3$: $w=9/15=0.6$ for the common class, and $9/6=1.5$ for each rare class. The rare classes weigh 2.5 times as much.

Prior shift: a model trained on a 50% positive sample outputs $p=0.80$. The deployment base rate is 1%. The odds become $4\times(0.01/0.99)/(0.5/0.5)=0.0404$, so $p_d=0.0404/1.0404=0.039$. The same evidence means 3.9% in deployment, not 80%.

## Why it works

**Cost threshold.** For a calibrated probability $p$, flagging a sample costs $(1-p)\,c_{FP}$ in expectation, and not flagging costs $p\,c_{FN}$. Flag when $(1-p)c_{FP}<p\,c_{FN}$, which is $p>c_{FP}/(c_{FP}+c_{FN})$ (Elkan, 2001). A threshold of 0.5 is this rule only when the two costs are equal.

**Prior shift.** Under a label shift, $P(x\mid y)$ is the same in both settings, where $x$ is the feature vector and $y\in\{0,1\}$ the label. The likelihood ratio $L(x)=P(x\mid y{=}1)/P(x\mid y{=}0)$ is therefore shared, and Bayes' rule gives posterior odds equal to prior odds times $L(x)$. Let $\pi_t$ and $\pi_d$ be the training and deployment positive priors, and $p$ the training posterior. The deployment odds are the training odds times $r=\dfrac{\pi_d/(1-\pi_d)}{\pi_t/(1-\pi_t)}$. Converting back gives $p_d=\dfrac{r\,p}{r\,p+1-p}$. King and Zeng (2001) give the equivalent intercept correction for rare-event logistic regression. Near extreme priors, add log odds instead of forming $r$ directly; preserve $p=0$ and $p=1$ separately.

**Class weights.** Multiplying each sample's loss by its class weight changes what the model learns. In the Bayes-optimal limit, a proper loss trained on weighted data estimates the posterior under a reweighted base rate. That is a prior shift, so the same correction applies to the output.

**Resampling, honestly.** Oversampling changes the training base rate in the same way, and the correction needs the resampled proportions, which you know. Duplicated points also let a flexible model memorise repeats, so it overfits. Undersampling throws away majority information. SMOTE (Chawla et al., 2002) adds a synthetic point on the segment between a minority sample and one of its $k$ nearest minority neighbours. That assumes the minority class is locally connected. In high dimension, or when the minority is multimodal, the segments cross regions with no real minority support. Resample only inside training folds. Resampling before the split puts synthetic copies of validation neighbours into training and inflates the validation score.

## Signal vs. noise

- **Signal:** the threshold is the cheapest lever, and it matches the business cost. Fix it before retraining.
- **Signal:** probabilities change whenever the training base rate changes. Check calibration against the deployment base rate.
- **Signal:** compare average precision with the prevalence baseline. That is the honest number under imbalance.
- **Noise for now:** the many named resampling variants. The metric and the threshold matter more than which variant you pick.

## Where it breaks

- **Accuracy-based selection** picks the all-majority model.
- **Resampling before splitting** leaks information, as described above.
- **Prior correction** needs the true deployment base rate. A wrong rate gives a confident, wrong probability.
- **Very large class weights** make the loss unstable and the gradients noisy.
- **Validation data** must keep the true base rate, or the threshold tuned on it will be wrong at deployment.
- **Too few minority examples** cannot be fixed by any method. Collect more data.

On the real `breast_cancer` dataset, malignant cases are 212 of 569. A classifier that predicts
every case benign still reports 62.7% accuracy while recall for malignant cases is 0%. A respectable
looking accuracy can hide a model that finds none of the cases you care about.

## Interview angle

"Your data is 1% positive. What do you do?" Answer shape: metric first (average precision against the 1% baseline), then the threshold from costs on calibrated probabilities, then training changes only if the minority signal is too weak. Resample only inside training folds, and check calibration at the deployment base rate. Follow-ups are in `interview.md`.

## Connections

- **Prerequisites:** `cml/roc-pr-curves-and-auc` (the metrics under imbalance) and `cml/bayes-theorem` (the prior correction is Bayes' rule).
- **Unlocks:** `cml/probability-calibration`, since threshold rules and prior corrections assume calibrated probabilities.
- **Lineage:** Elkan (2001) on cost-sensitive decisions, King and Zeng (2001) on prior correction for rare events, and Chawla et al. (2002) on SMOTE.
