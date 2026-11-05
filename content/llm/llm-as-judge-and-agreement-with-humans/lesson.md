---
title: "LLM-as-judge and agreement with humans"
minutes: 12
min_viable: "Read The system and Worked example, then implement p01 using Hint 1 if needed."
---

## The system

A judge is a function that reads an item and returns a label or a preference. Two shapes matter.

| Shape | Input | Output | Typical use |
|---|---|---|---|
| Pointwise | one answer | a score or label (for example good or bad) | filtering, dashboards |
| Pairwise | two answers $A$, $B$ | which is better: $A$, $B$ or tie | ranking models, preference data |

Human pairwise comparisons were the signal in Stiennon et al. (2020) [library:AI-ML/12-Frontiers/Stiennon et al. — Learning to Summarize from Human Feedback 2020.md]. AI-generated pairwise preferences, used as a training signal, are described in Bai et al. (2022) [library:AI-ML/13-Scaling-and-Alignment/Bai et al. — Constitutional AI 2022.md].

Three biases to design against:

- **Position bias.** The judge favours one slot, often the first. Fix it by judging both orders and scoring the pair consistently.
- **Verbosity bias.** Longer answers win even when they add nothing. Control for length, or check the win rate within length bands.
- **Self-preference.** A judge tends to favour text in its own style. Use a judge from a different model family for important comparisons.

## Worked example

Two raters label ten items as yes or no. The human $H$ and the judge $J$ are:

$H = (y,y,y,y,y,y,y,y,n,n)$ and $J = (y,y,y,y,y,y,n,n,n,y)$.

Items agree at positions 1 to 6 and at position 9, so $p_o = 7/10 = 0.7$. Marginals: $H$ says yes 8 times, so $p_H(y) = 0.8$; $J$ says yes 7 times, so $p_J(y) = 0.7$. Expected agreement by chance is

$$
p_e = 0.8 \times 0.7 + 0.2 \times 0.3 = 0.62,
$$

and Cohen's kappa is

$$
\kappa = \frac{p_o - p_e}{1 - p_e} = \frac{0.70 - 0.62}{0.38} \approx 0.21.
$$

Raw agreement of 70 percent looks respectable; kappa of 0.21 shows most of it is chance.

## Why it works

Let $p_o$ be the observed agreement, and let $p_a(\ell)$ and $p_b(\ell)$ be the two raters' label frequencies over a label set $L$. Under independence, the chance of both raters choosing $\ell$ is $p_a(\ell)\,p_b(\ell)$, so $p_e = \sum_{\ell \in L} p_a(\ell)\,p_b(\ell)$. Kappa rescales the gap between observed and chance agreement. It is 0 when agreement equals chance, 1 when agreement is perfect, and negative when the raters agree less often than chance.

The denominator $1 - p_e$ is zero only when both raters use one label for every item. Then kappa is undefined. This course's implementation raises `ValueError` and does not return a number. scikit-learn's `cohen_kappa_score` returns `nan` with an `UndefinedMetricWarning` in the same case (checked against the installed version), so a `nan` in a pipeline means this case, not a small value. The reference counts labels as integers, so $p_e = 1$ is an exact zero denominator, not a float that rounds to 0.9999999.

For pairwise judgments, judge each pair in both orders. Score $X$ as 1 when $X$ wins in both orders, 0 when $Y$ wins in both, and 0.5 otherwise. The win rate is the mean score. A tie is not a win for either side, so it scores 0.5. Two orders that disagree are the signature of position bias, so they also score 0.5. The 0.5 rule for inconsistent pairs is this course's convention, chosen so position bias cannot create a winner. It is not a standard result from the literature, and other rules, such as discarding inconsistent pairs, are also used.

## Signal vs. noise

Signal: the label set, the two raters, the order-swapped judgments, and the rule for inconsistent pairs. These decide whether a number means anything.

Noise: the exact wording of a rubric, unless you have measured its effect. Prompt tweaks help only when kappa or win rate moves on held-out human labels.

## Where it breaks

- **Skewed labels.** When most items share one label, $p_e$ is high and kappa can be low even when raw agreement is high. Report the label distribution with kappa.
- **Human ceiling.** Two humans rarely agree perfectly. Measure agreement between humans first; the judge cannot beat that ceiling by much.
- **Calibration.** A judge that says "sure" should be right more often than one that says "unsure". Bucket the judge's confidence and compare each bucket's agreement with humans. If the bucket the judge labels 90-percent confident agrees with humans only 60 percent of the time, the judge is miscalibrated, whatever its label says.
- **Small samples.** Kappa on 20 items has a wide interval. Report $n$ next to every value.

## Interview angle

Sixty-second answer: "Raw agreement hides chance, so I use Cohen's kappa over the label set and report the label distribution with it. For pairwise judging, I judge both orders and score inconsistent pairs as ties, so position bias cannot create a winner. I check calibration by confidence bucket, and I measure human-to-human agreement first as the ceiling."

Likely follow-ups:
- *Why not pointwise?* It is cheaper and simple to threshold, but scales drift and small differences are lost. Use pairwise when the comparison is close.
- *What if kappa is low but raw agreement is high?* That is the skewed-label case. Check the marginals before blaming the judge.

## Connections

This extends [confusion matrices and count-based scores](../../cml/confusion-matrix-and-count-based-scores/): kappa and win rate are both computed from counts.
