# Interview notes: Bayes' theorem

## Q1. A disease affects 1% of people. A test has 99% sensitivity and 95% specificity. A patient tests positive. How worried should they be?

**60-second answer.** About one in six, roughly 17%, not 99%. Bayes' rule gives
$P(D\mid+)=\dfrac{0.99\cdot0.01}{0.99\cdot0.01+0.05\cdot0.99}=\dfrac{0.0099}{0.0594}\approx0.167$.
The prior sets the floor. The condition is rare, so the 99% healthy group produces about 495
false positives per 10,000 people, against 99 true positives. The next steps are a confirmatory
test and a check that the prevalence fits this patient's risk. When prevalence is low, reducing
false positives matters more than raising sensitivity: going from 95% to 99% specificity lifts the
positive predictive value from 17% to 50%, while sensitivity moving from 99% to 100% adds under a
point.

**Follow-ups.**
- *What if two independent tests are both positive?* Each positive multiplies the odds by the
  likelihood ratio $0.99/0.05 = 19.8$. The odds go from $1/99$ to $19.8^2/99 = 3.96$, a probability of
  about 80%. This holds only if the tests are conditionally independent given disease status.
- *Where does the 17% sit in code?* It is the normalised product of prior and likelihood over the two
  hypotheses. The evidence is their sum, $P(+)=0.0594$.

## Q2. How do you update a belief over a long stream of observations without underflow?

**60-second answer.** Accumulate log-likelihoods: $\log\pi_i + \sum_t \log P(e_t\mid h_i)$ for each
hypothesis. Subtract the maximum before exponentiating, then normalise. Each new observation adds one
term. Multiplying 1,075 factors of 0.5 gives $2^{-1075}$, which rounds to 0 in float64, so the naive normaliser becomes 0/0. The log-sum-exp step keeps the largest term at 1, so the
normaliser is zero only when the evidence is truly impossible.

**Follow-ups.**
- *Is sequential updating the same as batch?* Yes, when observations are conditionally independent
  given the hypothesis. Each posterior becomes the next prior.
- *What if the evidence is impossible?* Then $P(E)=0$ and the posterior is undefined. Raise an error.
  Returning uniform weights hides a broken model.

## Q3. What happens when a hypothesis gets prior zero?

**60-second answer.** It stays at zero for every later update, because Bayes' rule multiplies by the
prior. That is correct when the hypothesis is truly impossible. It is a bug when the zero came from
rounding or from an over-strict prior. The fix is to choose the prior with a small positive mass on
every hypothesis you may need to recover, not to clip after the fact.

**Follow-ups.**
- *MAP or posterior mean?* MAP picks the mode of the posterior. The posterior mean minimises expected
  squared error. They coincide for symmetric unimodal posteriors such as a Gaussian, and generally differ otherwise.

## Common mistakes

- Reporting $P(E\mid H)$ as if it were $P(H\mid E)$. The two conditionals differ by the prior.
- Forgetting the prior, so the answer is a likelihood ranking rather than a posterior.
- Multiplying raw probabilities for long sequences. The normaliser underflows to 0/0.
- Treating repeated tests as independent when they share a cause, which over-counts the evidence.
- Returning NaN or uniform weights when the evidence has probability zero.
