# Interview notes: Probability basics

## Q1. Are mutually exclusive events independent?

**60-second answer.** No, and the confusion runs both ways. Mutually exclusive means
$P(A\cap B)=0$: the events cannot both happen. Independent means $P(A\cap B)=P(A)P(B)$: knowing one
says nothing about the other. If both events have positive probability, then $P(A)P(B)>0=P(A\cap B)$,
so they are dependent. Exclusivity is a strong form of dependence: learning that $A$ happened tells
you that $B$ did not.

**Follow-ups.**
- *When are they both?* Only when one has probability zero, because then both sides are $0$.
- *How would you test independence from a table?* Compare the joint table with the outer product of
  its marginals. For counts, use a chi-squared test of independence at a stated significance level.
- *Does conditional independence imply independence?* No. Two events can be independent given a third
  and dependent overall, as the shared-bias coin shows.

## Q2. How do you get $P(Y \mid X)$ from a joint table, and what can go wrong?

**60-second answer.** Divide each row by its row sum: $P(Y=j\mid X=i)=P(X=i,Y=j)/P(X=i)$. Every
output row sums to one, which is a free check. Two things go wrong. A row with $P(X=i)=0$ has no
conditional, so return NaN rather than a made-up distribution. And dividing by the column sum gives
the transposed conditional $P(X\mid Y)$, which equals $P(Y\mid X)$ only where $P(X=i)=P(Y=j)$ and the cell is nonzero.

**Follow-ups.**
- *Why not just multiply the marginals?* That assumes independence. The table tells you whether the
  assumption holds, so check before you use it.
- *How does this lead to Bayes' rule?* Swap the roles: $P(X\mid Y)=P(Y\mid X)P(X)/P(Y)$, with the
  denominator $P(Y)$ from total probability.

## Q3. Why does the law of total probability matter in ML?

**60-second answer.** It computes a probability by splitting on a partition you can reason about. If
$A_1,\dots,A_n$ partition the outcomes, then $P(B)=\sum_i P(B\mid A_i)P(A_i)$. It is the denominator of
Bayes' rule, and it is the marginal density of a mixture model: $p(x)=\sum_k \pi_k\,p(x\mid k)$, where the
partition is the set of component labels.

**Follow-ups.**
- *What if the events overlap?* The sum then counts the overlap twice, and you need inclusion-exclusion.
- *What is the partition in a classifier?* The class labels, when the classes are exhaustive and
  mutually exclusive.

## Common mistakes

- Treating mutually exclusive events as independent and multiplying their probabilities.
- Reading $P(X\mid Y)$ off a table when $P(Y\mid X)$ was asked for. The two divide by different sums.
- Dividing by a probability that can be zero, and getting NaN or infinity without noticing.
- Checking that a table sums to 1 with `==`, which fails on floating-point rounding.
