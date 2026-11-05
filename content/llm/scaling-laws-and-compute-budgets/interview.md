# Interview notes — Scaling laws and compute budgets

## Q1. How would you split a fixed training budget between model size and data?

**60-second answer.** Training costs about $6ND$ FLOPs, where $N$ is the parameter count and $D$ the tokens. For a fixed budget, a larger $N$ leaves fewer tokens. Loss has a term for model size and a term for data, $L=E+A/N^\alpha+B/D^\beta$, and the best split balances them. Hoffmann et al. found that $N$ and $D$ should grow at nearly equal rates, so with $D\approx20N$ both scale roughly as $\sqrt{C}$, with the exponent of $N_{opt}$ between 0.46 and 0.50 across its three methods. I would treat the ratio as a starting point and check it against the data I have, since a model cannot train on more distinct text than exists.

**Follow-ups.**

- Why not just train the biggest model? For a fixed budget a larger model sees fewer tokens. Its loss can be higher than a smaller model trained on more data, and the inference cost of a large model is paid on every query.
- What would change the answer? Data quality, a data ceiling, or a different target. If inference cost matters more than training cost, teams often train beyond the compute-optimal ratio.

## Q2. How would you estimate a scaling law from experiments?

**60-second answer.** Train a grid of model sizes at several budgets, record the final loss, and fit $\log y=\log a+b\log x$ by least squares. The slope is the exponent and the intercept gives $a$. I would check the residuals for curvature, because a line on log-log axes can hide a bend, and I would not extrapolate far beyond the largest run.

**Follow-ups.**

- Why log space? It turns the power law into a line, and it weights relative errors equally across decades.
- What goes wrong with the fit? Points at zero or negative values cannot be logged, and points at very small budgets can dominate a linear-space fit. Check the range first.

## Common mistakes

- Using $C\approx 2ND$ or ignoring the factor 6.
- Reporting the rule of thumb as an exact constant.
- Fitting in linear space and calling the result a power law.
- Extrapolating a fit a decade past the data.
