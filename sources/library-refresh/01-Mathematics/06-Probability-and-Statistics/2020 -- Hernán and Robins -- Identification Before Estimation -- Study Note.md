# Identification before estimation

A model can predict an outcome accurately while giving a wrong answer to what would happen under an intervention. Causal inference first asks whether the desired intervention quantity is determined by the observed distribution under stated assumptions. Only then does it ask how to estimate that quantity from finite data.

## Source and reading locators

Causal Inference: What If is the living online author edition, revised August 19, 2026. Its authors request the bibliographic citation Hernán and Robins (2020), *Causal Inference: What If*, Chapman & Hall/CRC. The [author landing page](https://miguelhernan.org/whatifbook) supplies both the citation and current download. Read Chapters 1–3 for counterfactuals, randomized experiments and observational studies; §2.3 for standardization and §§3.1–3.5 for identification conditions; Chapters 12–13 for inverse probability weighting and model-based standardization. Revision dates distinguish the copy used from a frozen 2020 text.

## The estimand and assumptions

For a binary intervention \(A\), let \(Y^a\) be the outcome under setting \(A=a\). The average treatment effect is \(E[Y^1-Y^0]\). Since each unit ordinarily reveals only one outcome, the observed association does not identify this difference by itself.

For discrete pretreatment covariates \(L\), a standard identification route assumes:

- Consistency: if the observed treatment is \(a\), then \(Y=Y^a\), with a sufficiently well-defined treatment.
- Conditional exchangeability: \(Y^a\perp A\mid L\).
- Positivity: \(P(A=a\mid L=l)>0\) in covariate strata used by the target population.

These assumptions give

\[
E[Y^a]=\sum_l E[Y\mid A=a,L=l]P(L=l).
\]

Consistency links observed and counterfactual outcomes, exchangeability permits comparison within strata, and positivity ensures those conditional comparisons are supported by observations.

## Worked example: reverse the crude association

Suppose half the population has a high baseline risk \(L=H\), and half a low risk \(L=L\). Treatment probabilities are \(0.8\) in the high-risk group and \(0.2\) in the low-risk group. The observed risks are:

| Stratum | Treated risk | Untreated risk |
| --- | ---: | ---: |
| High risk | 0.40 | 0.60 |
| Low risk | 0.10 | 0.20 |

Among treated people, the high-risk fraction is \(0.8\), so crude treated risk is \(0.8(0.40)+0.2(0.10)=0.34\). Among untreated people it is \(0.2\), so crude untreated risk is \(0.2(0.60)+0.8(0.20)=0.28\). The unadjusted association suggests a risk increase of \(0.06\).

Standardizing both interventions to the same population gives
\(E[Y^1]=0.5(0.40)+0.5(0.10)=0.25\) and
\(E[Y^0]=0.5(0.60)+0.5(0.20)=0.40\).
Under the assumptions above, the causal risk difference is \(-0.15\). Different treatment group composition caused the crude comparison to reverse.

The table does not establish exchangeability. An unmeasured common cause could invalidate the causal interpretation while preserving every displayed association.

## Lineage and AI relevance

[graphical models](2008%20--%20Wainwright%20and%20Jordan%20--%20Graphical%20Models%20and%20Variational%20Inference%20--%20Study%20Note.md) describes factorization and conditional dependence. Causal models add intervention or counterfactual meaning. More flexible regression improves an estimation step only after the target and identifying assumptions are defensible.

This matters for recommendation policies, treatment assignment, observational product analyses and evaluation of interventions made by automated systems. A predictor can use variables that would be inappropriate to adjust for when estimating a causal effect. Adjusting for every measured feature is not a universal remedy; post-treatment variables and selection mechanisms require reasoning about the data-generating process.

Small estimated standard errors quantify one source of sampling variability. They do not certify identification. The disciplined reading order is estimand → assumptions → identifying formula → estimator → diagnostics and uncertainty, with sensitivity analysis when key assumptions remain uncertain.
