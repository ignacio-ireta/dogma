---
title: Probability basics
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A **sample space** $\Omega$ lists the outcomes of an experiment. An **event** is a set of outcomes,
and $P(A)$ is the sum of the probabilities of its outcomes. Two discrete variables, $X$ with $m$
values and $Y$ with $k$ values, are described by a **joint table** $P\in\mathbb{R}^{m\times k}$ with
$P_{ij}=P(X=i,Y=j)\ge 0$ and $\sum_{ij}P_{ij}=1$. Four operations cover most exercises:

| Object | Formula | Shape | Meaning |
|---|---|---|---|
| Marginal of $X$ | $P(X=i)=\sum_j P_{ij}$ | $(m)$ | Sum each row |
| Marginal of $Y$ | $P(Y=j)=\sum_i P_{ij}$ | $(k)$ | Sum each column |
| Conditional | $P(Y=j\mid X=i)=P_{ij}/P(X=i)$ | $(m,k)$, rows sum to 1 | Keep row $i$, renormalise |
| Product rule | $P(A\cap B)=P(A\mid B)\,P(B)$ | scalar | First $B$, then $A$ given $B$ |

**Conditioning on an event or a variable.** $P(A\mid B)=P(A\cap B)/P(B)$ is defined only when
$P(B)>0$. Conditioning on a random variable $X$ gives a family $P(Y\mid X=i)$, one distribution for
each value $i$ with $P(X=i)>0$. Those are the rows of the joint table.

**Independence** means conditioning changes nothing: $P(A\cap B)=P(A)P(B)$, equivalently
$P(B\mid A)=P(B)$ when $P(A)>0$. In the joint table, independence means every cell factorises,
$P_{ij}=P(X=i)P(Y=j)$, so the table is the outer product of its marginals.

**Law of total probability.** If $A_1,\dots,A_n$ partition $\Omega$ (disjoint and covering), then
$P(B)=\sum_i P(B\mid A_i)\,P(A_i)$. Overlapping pieces double-count, and a piece with $P(A_i)=0$
contributes nothing, so its conditional need not exist. This is how Bayes' rule gets its denominator.

**Decision rule.** Before you multiply two probabilities, check independence. Before you divide by a
probability, check that it is positive.

## Worked example

Let $X$ be the weather (sunny, rain) and $Y$ the traffic (low, medium, high):

$$P=\begin{pmatrix}0.20 & 0.20 & 0.10\\ 0.05 & 0.15 & 0.30\end{pmatrix}.$$

1. The entries sum to 1.
2. Row sums give $P(\text{sunny})=0.5$ and $P(\text{rain})=0.5$.
3. Column sums give $P(\text{low})=0.25$, $P(\text{medium})=0.35$, $P(\text{high})=0.40$.
4. Divide each row by its own row sum. $P(Y\mid\text{sunny})=(0.4,\,0.4,\,0.2)$ and
   $P(Y\mid\text{rain})=(0.1,\,0.3,\,0.6)$. Each row sums to 1.
5. Total probability checks out: $P(\text{high}) = 0.2\cdot0.5 + 0.6\cdot0.5 = 0.4$.
6. Not independent: $P(\text{low}\mid\text{sunny})=0.4\neq0.25=P(\text{low})$. Rain makes heavy
   traffic more likely.

In NumPy: `joint.sum(axis=1)` gives $P(X)$, `joint.sum(axis=0)` gives $P(Y)$, and
`joint / joint.sum(axis=1, keepdims=True)` gives the conditional.

## Why it works

A conditional is the definition of probability applied to a smaller universe. Once you know $X=i$,
only the cells in row $i$ remain, and their probabilities must be rescaled to sum to 1. The division
by $P(X=i)$ is that rescaling. The product rule is the same definition read in the other direction.
Total probability follows because the events $A_i$ split $B$ into disjoint pieces, and the piece
inside $A_i$ has probability $P(B\mid A_i)P(A_i)$. These rules carry most of discrete probability
(Blitzstein and Hwang, *Introduction to Probability*, the chapter on conditional probability).

## Signal vs. noise

- **Signal:** a conditional is a row divided by its row sum. Every defined row sums to 1, which is a
  free check on your code.
- **Signal:** independence means the joint is an outer product of its marginals. Test the table,
  not the story.
- **Signal, the classic confusion:** *mutually exclusive* ($P(A\cap B)=0$) and *independent* are
  different. If $P(A)>0$ and $P(B)>0$, exclusive events are never independent, because
  $0\neq P(A)P(B)$. Knowing $A$ happened tells you $B$ did not.
- **Noise for now:** measure theory, sigma-algebras, continuous densities.

## Where it breaks

- **Zero-probability conditioning.** $P(Y\mid X=i)$ is undefined when $P(X=i)=0$. Return NaN for
  that row. A row of zeros looks like a valid answer and is not one.
- **Transposed conditionals.** $P(A\mid B)\neq P(B\mid A)$ in general. Dividing by the column sum
  instead of the row sum silently computes $P(X\mid Y)$.
- **Rounding.** Floating-point tables often miss 1 by a few units of $10^{-16}$. Validate with a
  tolerance (here $10^{-9}$), never with `==`.
- **Marginals do not fix the joint.** Two tables with the same row and column sums can have very
  different conditionals. Check independence against the joint, not against its marginals.
- **Conditional independence is not independence.** Let a fair coin $Z$ set the bias of two flips $X$
  and $Y$: $P(\text{heads}\mid Z=1)=0.9$ and $P(\text{heads}\mid Z=0)=0.1$. Given $Z$, the flips are
  independent. Yet $P(X=1,Y=1)=0.5(0.81)+0.5(0.01)=0.41$, while $P(X=1)P(Y=1)=0.25$. Check
  independence in the joint, and conditional independence within each value of the conditioning variable.

## Interview angle

"Are mutually exclusive events independent?" Answer shape: no. Exclusive means $P(A\cap B)=0$;
independent means $P(A\cap B)=P(A)P(B)$. Exclusive events with positive probability fail the
independence test by construction. Follow-ups: how to test independence from a table (compare it
with the outer product of its marginals), and where total probability appears (the denominator of
Bayes' rule). Full answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/descriptive-statistics` (a frequency table is an empirical probability table).
- **Unlocks:** `cml/bayes-theorem` (total probability supplies its denominator),
  `cml/bernoulli-binomial-and-poisson`, `cml/expectation-and-variance`,
  `cml/covariance-and-correlation`, `cml/naive-bayes`, `cml/entropy-and-mutual-information`.
- **Lineage:** Pascal and Fermat's 1654 correspondence on the problem of points started the calculus
  of chances. Kolmogorov's *Grundbegriffe* (1933) gave the rules their axiomatic form. His general
  definition conditions on a $\sigma$-algebra rather than dividing by $P(B)$, which handles continuous
  observations. This lesson stays with the finite case (the Library note's locator is Chapter V §§1–4, not checked against the book;
  [source: library-refresh/01-Mathematics/06-Probability-and-Statistics/1933 -- Kolmogorov -- Probability Spaces, Conditioning and Limit Laws -- Study Note.md]).
