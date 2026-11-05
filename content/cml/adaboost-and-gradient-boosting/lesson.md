---
title: AdaBoost and gradient boosting
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Boosting builds an additive model $F_M=F_0+\nu\sum_{m=1}^M h_m$ in stages. $F_0$ is a constant, each $h_m$ is a weak learner (here a depth-one tree, a stump), and $\nu$ is the learning rate.

- **Stagewise gradient step.** At stage $m$, fit $h_m$ to the **negative gradient** of the loss $L(y,F)$ at the current model, $-\partial L/\partial F$ evaluated at $F_{m-1}(\mathbf x_i)$, then update $F_m=F_{m-1}+\nu h_m$.
- **Squared error** $L=\tfrac12(y-F)^2$ has negative gradient $y-F$. Gradient boosting for regression therefore fits stumps to the residuals.
- **AdaBoost** (labels $y\in\{-1,+1\}$) minimises the exponential loss $L=e^{-yF}$. Its weights are $w_i\propto e^{-y_iF_{m-1}(\mathbf x_i)}$. A round with weighted error $\varepsilon$ gets the vote $\alpha=\tfrac12\ln\frac{1-\varepsilon}{\varepsilon}$, and the next weights are $w_i\,e^{-\alpha\,y_i h_m(\mathbf x_i)}$, normalised.
- **Shrinkage** $\nu\in(0,1]$ scales each step. Smaller $\nu$ needs more rounds.
- **Early stopping** chooses $M$ by the error on held-out data, not by the training error.

## Worked example

**One round of gradient boosting, squared error.** Take $x=(0,1,2,3)$, $y=(1,2,3,4)$, $\nu=0.5$.

- $F_0=\bar y=2.5$, so the residuals are $(-1.5,\,-0.5,\,0.5,\,1.5)$ and the training SSE is $5.0$.
- The stump that maximises the SSE reduction splits at $t=1.5$, with gain $4.0$. The other candidates give $3.0$ (at $t=0.5$ and $t=2.5$). The leaves are the child means, $-1.0$ and $+1.0$.
- $F_1=2.5+0.5\cdot(-1,-1,+1,+1)=(2,2,3,3)$, with training SSE $2.0$. The second round has a tie: thresholds $0.5$ and $2.5$ both have gain $4/3$. `boost_stumps_mse` breaks ties toward the smaller threshold, so it picks $t=0.5$, giving $(1.5,\,2.17,\,3.17,\,3.17)$ and SSE $1.0$.

**One round of AdaBoost.** Four samples have uniform weights, and the classifier mislabels one of them, so $\varepsilon=0.25$. Then $\alpha=\tfrac12\ln 3=0.549$. The mistaken sample is multiplied by $e^{\alpha}=\sqrt3$, the other three by $1/\sqrt3$, and the result is normalised to $(\tfrac16,\tfrac16,\tfrac16,\tfrac12)$. The same classifier now has weighted error exactly $\tfrac12$.

Both numbers come from `boost_stumps_mse` and `adaboost_round`, the reference implementations of this topic's problems.

## Why it works

**Boosting as gradient descent in function space.** A gradient step moves the current function $F$ in the direction of steepest descent of the loss. A stump fitted to the negative gradient is an approximation to that direction that can be evaluated on new points. Friedman (2001, §§2–3) develops this view: the raw gradient is defined only at the data points, so a fitted base learner carries it to new ones [corpus:library-refresh/02-AI-ML/02-Statistical-Learning/2001 -- Friedman -- Greedy Function Approximation - A Gradient Boosting Machine -- Annals of Statistics.pdf.md p.6].

**Why AdaBoost's weights are exponential losses.** Writing $\varepsilon$ for the weighted error and minimising $\sum_i e^{-y_i(F+\alpha h)}$ over $\alpha$ gives the closed form $\alpha=\tfrac12\ln\frac{1-\varepsilon}{\varepsilon}$. At that $\alpha$, the new weights make $h$ exactly chance-level. This is the additive logistic regression view of Friedman, Hastie & Tibshirani (2000). Freund & Schapire (1997) gave the original algorithm. Both papers are cited from memory; neither is in the corpus.

**Why shrinkage helps.** A small step $\nu$ moves $F$ less per round, so the model keeps some slack for later rounds to correct. In practice a smaller $\nu$ with more rounds usually generalises better, at a higher compute cost.

## Signal vs. noise

- **Signal:** each round fits the current residuals or exponential-loss weights. With squared error and $0<\nu\le1$, a fitted least-squares stump cannot increase training SSE; classification error need not fall every round.
- **Signal:** learning rate and number of rounds trade off. Tune them together, with early stopping.
- **Noise for now:** the second-order (Newton) step and regularised objectives. They belong in `cml/xgboost-second-order-objective`.

## Where it breaks

- **Outliers under exponential loss.** A mislabelled point is wrong in every round, so its weight grows relative to the correctly classified points by a factor $e^{2\alpha}$ per round. AdaBoost chases it. Robust losses (`cml/regression-metrics-and-robust-losses`) limit this.
- **Too many rounds.** Training error keeps falling while validation error rises. Stop by validation loss.
- **Stumps have depth one.** A sum of stumps remains additive in the features, regardless of the number of rounds. Cross-feature interactions such as XOR require deeper trees or explicit interaction features.
- **Error at or above one half.** AdaBoost has no positive vote for a classifier no better than chance, so the loop must stop there. A perfect classifier has $\varepsilon=0$ and an infinite $\alpha$, so it must stop too.

## Interview angle

*60-second answer:* Boosting fits an additive model in stages. Each weak learner approximates the negative gradient of the loss at the current model. With squared error that is the residual, so gradient boosting fits stumps to residuals and adds a shrunken step. AdaBoost is the same idea with exponential loss: the weights shift toward the samples the current learner got wrong, and the vote is $\tfrac12\ln\frac{1-\varepsilon}{\varepsilon}$. The learning rate and the number of rounds trade off, and I choose the number of rounds by validation loss. AdaBoost is sensitive to label noise because a persistently wrong point keeps gaining weight.

*Likely follow-ups:* Why use a small learning rate? (It limits how far each stage moves, so more rounds with a smaller step usually generalise better.) What goes wrong with mislabelled data? (Exponential loss gives those points growing weight, so the model overfits them.)

## Connections

- **Prerequisites:** `cml/choosing-a-tree-split` (stumps and the split search), `cml/gradient-descent` (the gradient step).
- **Unlocks:** `cml/xgboost-second-order-objective` (the second-order version of the same stagewise step).
- **Lineage:** Freund & Schapire (1997) introduced AdaBoost. Friedman (2001) cast boosting as gradient descent. Friedman, Hastie & Tibshirani (2000) gave the additive logistic view. ESL chapter 10 covers all three (from memory).
- **References:** [Friedman (2001)](https://doi.org/10.1214/aos/1013203451) develops gradient boosting in function space.
