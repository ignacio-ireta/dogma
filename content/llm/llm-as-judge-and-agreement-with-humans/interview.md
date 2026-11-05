# Interview notes — LLM-as-judge and agreement with humans

## Q1. How would you check whether an LLM judge agrees with human labels?

**60-second answer.** I collect human labels on a sample and run the judge on the same items. Raw agreement is the first number, but it hides chance: if most items share one label, two raters agree often without judging well. So I report Cohen's kappa over the label set, with the label distribution beside it. I also compare human-to-human agreement as the ceiling, so I know how much room the judge has.

**Follow-ups.** When is kappa undefined? When both raters use one label for every item, so $p_e = 1$. What is a good kappa? That depends on the task and the human ceiling, so I compare against the humans rather than a fixed threshold.

## Q2. How do you stop a pairwise judge from picking the first answer?

**60-second answer.** I judge every pair twice, once in each order. If the judge picks the same answer in both orders, I count that as a preference. If the two orders disagree, or one of them is a tie, I count the pair as a tie. A judge that always picks slot A scores exactly one half, which is the right answer for pure position bias.

**Follow-ups.** Why score inconsistent pairs as ties rather than dropping them? Dropping them hides how often the judge is unreliable. Scoring them as half a win keeps that rate in the result.

## Q3. A model's win rate rose after a prompt change. What do you check before celebrating?

**60-second answer.** I check whether the judge still agrees with humans after the change, using kappa on a held-out set. I check that the gain is not verbosity, by comparing win rates within length bands. And I check the order-swapped judgments, since a change can move the position bias rather than the quality. Only after those do I trust the new win rate.

**Follow-ups.** What if the judge and the candidate share a model family? Self-preference is possible, so I rerun with a judge from another family.

## Common mistakes

- Reporting raw agreement on skewed labels as if it were judge quality.
- Judging each pair in one order only, then reading the win rate as a preference.
- Breaking ties silently instead of stating the rule.
- Treating a judge's stated confidence as calibrated without a reliability check.
