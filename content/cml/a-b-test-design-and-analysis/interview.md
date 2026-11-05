# Interview notes: A/B test design and analysis

## Q1. Design an experiment for a checkout change. What is the metric, how many users, how long?

**60-second answer.** I start with the decision: one primary metric the team will act on, such as checkout
conversion, and a few guardrails, such as refund rate and page latency. I randomise by user, so repeat visits
stay in one arm. I agree the minimum detectable effect with the product owner, for example a two-point lift on
a ten percent baseline. Then I compute the per-arm sample size for two proportions at 5% two-sided and 80%
power, which gives about 3,841 users per arm. I convert that to a duration in whole weeks at the traffic I have,
and I write down the analysis before launch.

**Follow-ups.**
- *Why not use the smallest effect we could ever care about?* Sample size grows with one over the MDE
  squared. Halving the MDE quadruples the users. I pick the smallest effect that changes the decision.
- *What about several metrics?* Designate one primary metric. Treat the others as secondary, and correct for
  multiple comparisons if any of them drives a decision.
- *Could we skip the test and use logs where users chose the feature?* Only under identification assumptions:
  after adjusting for measured confounders, assignment must be independent of the potential outcomes, and
  every group must have a positive chance of being in each arm (overlap). A small standard error does not
  establish those assumptions.

## Q2. The result crossed the significance threshold on day three. Ship it?

**60-second answer.** Not on that basis. Each time the result is checked before the planned horizon, the chance
of a false positive grows. In a simulation with no true effect, five looks at a nominal 5% test rejected about
14% of the time. I would wait for the planned sample, or use a sequential method built for repeated looks, and
then report the interval for the effect against the minimum effect worth shipping.

**Follow-ups.**
- *Can we stop early if the effect is huge?* Only with a pre-specified sequential boundary. Otherwise an
  extreme early result is also the most likely to be a fluke or a novelty effect.
- *What would you check first?* The sample ratio, then the guardrails, then the effect over time.

## Q3. The assignment split is 50.6 / 49.4 on 100,000 users. What do you do?

**60-second answer.** I treat the experiment as broken until I know why. With a planned 50/50 split, that
imbalance has a chi-square statistic of 14.4 and a p-value of about 0.00015. That is sample ratio mismatch.
Common causes are bots filtered differently by arm, redirects that drop users, or logging that fails for one
variant. I would find the cause and rerun, because the outcome comparison is not valid while the assignment is
imbalanced.

**Follow-ups.**
- *Is a small imbalance acceptable?* At large sample sizes even small deviations are evidence of a problem.
  The test answers whether the imbalance is more than chance, not whether it is large.
- *Can you just adjust for the imbalance in the analysis?* Not reliably. The cause may also change who is in
  each arm, which biases the comparison.

## Common mistakes

- Choosing the MDE after seeing the data, so the test is designed around the result.
- Stopping an experiment at the first significant p-value.
- Randomising by session when the decision is about users, so the same user counts in both arms.
- Skipping the sample ratio check and reading the metric directly.
- Reading a statistically significant but tiny lift as a business win without comparing it to the MDE.
