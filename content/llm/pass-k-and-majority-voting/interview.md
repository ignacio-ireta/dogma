# Interview notes — pass@k and majority voting

## Q1. How would you estimate pass@k for a code model from several samples per problem?

**60-second answer.** I generate $n$ samples per problem, count the $c$ that pass the tests, and compute the unbiased estimator $1 - \binom{n-c}{k}/\binom{n}{k}$. It answers what a user would see with $k$ attempts, drawn without replacement from the $n$ I already have. I compute it as a product of $k$ factors, so large $n$ does not overflow. I report $n$ with every number, because a small $n$ makes the estimate noisy.

**Follow-ups.** Why not $1-(1-c/n)^k$? It treats the samples as draws with replacement, and it plugs a noisy estimate into a power, so it is biased. Is pass@1 the same as $c/n$? Yes, and that is a useful check.

## Q2. When would you use majority voting instead of a verifier?

**60-second answer.** When there is no test suite, but the answer can be reduced to one normalised string, such as a number or a short label. I sample several reasoning paths, normalise each final answer, and return the most common one. Ties need a rule fixed in advance, so the result is reproducible. The cost is $n$ full generations per question, and it helps only when the answers are correct more often than they agree on a wrong one.

**Follow-ups.** What if answers differ only in format? Normalise before counting, and keep punctuation unless it changes meaning. What if the vote is split? Report the tie and the margin, not a confident answer.

## Q3. A teammate reports pass@10 of 0.9 from only 10 samples. What do you ask?

**60-second answer.** With $n = k = 10$, the estimator reduces to whether any sample passed, so 0.9 means nine of ten problems had a passing sample. Before trusting the number, I ask whether the samples were independent. Repeated samples from low temperature can look like many attempts while carrying one idea. I also ask for more samples, so the unbiased estimator has room to work.

**Follow-ups.** Would more samples change the headline? It would tighten the interval. Would a different temperature change it? Yes, because it changes the per-sample pass probability.

## Common mistakes

- Using $1-(1-c/n)^k$ and calling it unbiased.
- Reporting pass@k without the value of $n$.
- Counting near-duplicate samples as independent evidence.
- Breaking vote ties by insertion order or alphabetical order without saying so.
