---
title: Bayes' theorem
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Let $H$ be a hypothesis and $E$ the evidence. Bayes' rule is

$$P(H\mid E)=\frac{P(E\mid H)\,P(H)}{P(E)},\qquad P(E)=\sum_h P(E\mid h)\,P(h).$$

For a discrete hypothesis set with $h$ values, four objects carry the rule:

| Term | Symbol | Meaning | Shape |
|---|---|---|---|
| Prior | $\pi_i=P(H=i)$ | belief before the data | $(h)$, sums to 1 |
| Likelihood | $L_{ij}=P(e=j\mid H=i)$ | how well each hypothesis predicts each observation | $(h,o)$, rows sum to 1 over **observations** |
| Evidence | $P(E)$ | total probability of the data, the normaliser | scalar |
| Posterior | $P(H=i\mid E)$ | belief after the data | $(h)$, sums to 1 |

The posterior is proportional to prior times likelihood. The evidence only rescales it to sum to 1.
A likelihood row is a distribution over data for one hypothesis. Read down a column instead, and the
values need not sum to 1: here $P(+\mid D)+P(+\mid\neg D)=0.99+0.05=1.04$. The likelihood is a
function of the hypothesis, not a probability distribution over it.

**Sequential updating.** If observations $e_1,\dots,e_T$ are conditionally independent given $H$,
then $P(H\mid e_{1:T})\propto P(H)\prod_t P(e_t\mid H)$. Updating once per observation, with each
posterior as the next prior, gives the same answer as one batch update.

**Log space.** Take logs of the product and sum them: $\log w_i=\log\pi_i+\sum_t\log L_{i,e_t}$.
Subtract the maximum $m=\max_i\log w_i$ before exponentiating: $P(H=i\mid E)=\exp(\log w_i-m)/\sum_k\exp(\log w_k-m)$.
This is the log-sum-exp trick. The shift also blocks overflow: every exponent is now at most 0.
Without it, $e^{710}$ is already infinite in float64, because $\ln(\text{largest float64})\approx709.78$.

## Worked example

A disease has prevalence $P(D)=0.01$. A test has sensitivity $P(+\mid D)=0.99$ and a false-positive
rate $P(+\mid\neg D)=0.05$. A patient tests positive. What is $P(D\mid+)$?

1. Prior: $P(D)=0.01$.
2. Evidence: $P(+)=0.99\cdot0.01+0.05\cdot0.99=0.0099+0.0495=0.0594$.
3. Posterior: $P(D\mid+)=0.0099/0.0594=1/6\approx16.7\%$.

Natural frequencies make it concrete. In 10,000 people, 100 are sick and 99 of them test positive.
Of the 9,900 healthy people, 495 test positive. That is 594 positives, and 99 of them are sick:
$99/594=1/6$. Most positives are false, since the healthy group is 99 times larger.

**Two tests.** Assume the tests are conditionally independent given disease status. Each positive
multiplies the odds by the likelihood ratio $0.99/0.05=19.8$. Prior odds are $1/99$. After one
positive the odds are $19.8/99=1/5$, a probability of $1/6$. After two positives they are
$19.8^2/99=3.96$, a probability of $99/124\approx79.8\%$.

**Log space.** Each factor of 0.5 halves a float64 value. The smallest positive float64 is the subnormal
$2^{-1074}\approx4.9\times10^{-324}$, so $0.5^{1074}$ survives, while $0.5^{1075}=2^{-1075}$ rounds to 0.
A posterior built from a product that long becomes $0/0$.

## Why it works

Bayes' rule is the definition of conditional probability written twice: $P(H\cap E)=P(H\mid E)P(E)=P(E\mid H)P(H)$.
Solving for $P(H\mid E)$ and expanding $P(E)$ by total probability gives the formula (Blitzstein and
Hwang, *Introduction to Probability*, the chapter on conditional probability). For sequential data,
conditional independence makes the joint likelihood a product, so each step multiplies the previous
posterior by one likelihood term.

For values $z_h$, $\log\sum_h e^{z_h}=m+\log\sum_h e^{z_h-m}$ with $m=\max_h z_h$. The shifted sum
contains a term equal to 1, so it cannot underflow to zero. The same trick stabilises softmax and
cross-entropy (Goodfellow, Bengio and Courville, *Deep Learning*, the chapter on numerical
computation). The log-odds form makes the update additive: posterior log-odds equal prior log-odds
plus the log likelihood ratio. A related point (per the Library note on MacKay, 2003): the evidence
$p(D)=\int p(D\mid\theta)\,p(\theta)\,d\theta$ averages fit over the prior on parameters, so the best
single fit generally cannot replace it
([source: library-refresh/01-Mathematics/07-Information-Theory/2003 -- MacKay -- Coding, Bayesian Inference and Learning as Compression -- Study Note.md]).

## Signal vs. noise

- **Signal, the base-rate fallacy:** a test with 99% sensitivity still gives mostly false positives when
  the condition is rare. The prior sets the floor, and the prior is the first number to ask for.
- **Signal:** the posterior is prior times likelihood, normalised. Drop the prior and you get a
  likelihood ranking, usually the wrong answer.
- **Signal:** the posterior is a distribution. Its mode (MAP) and its mean answer different questions.
- **Signal:** a long product of probabilities belongs in log space. Use sums of logs and subtract the
  maximum.
- **Noise for now:** continuous priors and conjugacy (the beta-binomial topic covers that), and
  sampling methods for posteriors with no closed form.

## Where it breaks

- **Impossible evidence.** If $P(E)=0$, the posterior is undefined. Raise an error. Returning NaN or a
  uniform vector hides a broken model.
- **A zero prior stays zero.** No amount of data revives a hypothesis with prior 0. A probability floor
  that adds a small mass to it changes the answer silently.
- **Conditional independence is an assumption.** Two positive tests from the same faulty batch are not
  independent given disease status. Sequential updating then counts the same evidence twice, and the
  posterior becomes too confident.
- **Garbage likelihoods give confident garbage.** The posterior is only as good as $P(E\mid H)$.

## Interview angle

"A disease affects 1% of people, a test has 99% sensitivity and 95% specificity, and a patient tests
positive. How worried should they be?" Answer shape: first the base rate, then the calculation, which
gives about 17% rather than 99%. Then what would change the answer: a confirmatory test, and whether
the tests are independent. Follow-ups: two positives (about 80% if independent), and why a long stream
of observations is updated in log space. Full answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/probability-basics` (total probability supplies the denominator).
- **Unlocks:** `cml/naive-bayes` (Bayes' rule with an independence assumption over features),
  `cml/beta-binomial-conjugate-updating`, `cml/sigmoid-and-log-odds` (updating becomes addition in
  log-odds), `cml/softmax` (the same log-sum-exp trick).
- **Lineage:** Bayes' essay appeared posthumously in 1763. Laplace derived the general rule
  independently in 1774.
