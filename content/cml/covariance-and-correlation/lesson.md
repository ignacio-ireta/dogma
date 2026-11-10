---
title: Covariance and correlation
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A data matrix $X \in \mathbb{R}^{n\times d}$ has one **observation per row** and one **variable per
column**. Centre each column by its mean, $X_c = X - \mathbf{1}\bar{x}^\top$, where $\bar{x}\in\mathbb{R}^d$
holds the column means. Every quantity below is built from $X_c$.

| Quantity | Formula | Shape | Meaning |
|---|---|---|---|
| Sample covariance | $C = \dfrac{X_c^\top X_c}{n-1}$ | $(d,d)$ | average co-movement of two variables |
| Diagonal | $C_{jj} = \operatorname{Var}(x_j)$ | scalar per variable | spread of one variable |
| Correlation | $R_{jk} = \dfrac{C_{jk}}{\sqrt{C_{jj}C_{kk}}}$ | $(d,d)$ | covariance with scale removed, in $[-1,1]$ |

The divisor $n-1$ gives the unbiased variance estimate. Dividing by $n$ instead (`ddof=0`) changes $C$
but cancels out of $R$. Two facts follow from the definitions:

- **Positive semi-definite.** For any vector $\mathbf{v}$, $\mathbf{v}^\top C\mathbf{v} = \operatorname{Var}(X\mathbf{v})
  \ge 0$. So $C$ has no negative eigenvalues. It is singular when one column is an exact linear combination of
  the others.
- **Unit diagonal.** $R_{jj} = 1$ for every variable that is not constant.

## Worked example

Take $X = \begin{pmatrix}1&2\\3&4\\5&9\end{pmatrix}$. Column means are $(3, 5)$, so
$X_c$ has rows $(-2,-3)$, $(0,-1)$, $(2,4)$.

1. $C_{11} = \dfrac{(-2)^2+0^2+2^2}{2} = \dfrac{8}{2} = 4$.
2. $C_{22} = \dfrac{9+1+16}{2} = 13$.
3. $C_{12} = \dfrac{(-2)(-3)+0(-1)+2(4)}{2} = \dfrac{14}{2} = 7$.
4. $R_{12} = \dfrac{7}{\sqrt{4\cdot 13}} \approx 0.971$.

Now scale column 1 by 1000. $C_{12}$ becomes 7000, but $R_{12}$ stays $0.971$. Covariance carries units;
correlation does not.

Make column 2 equal to $2\times$ column 1: $C = \begin{pmatrix}4&8\\8&16\end{pmatrix}$, with eigenvalues
$0$ and $20$. The matrix is positive semi-definite and singular, and $R_{12} = 1$ exactly.

## Why it works

Expanding $\operatorname{Var}(X+Y)$ in `cml/expectation-and-variance` produced the cross term
$\operatorname{Cov}(X,Y) = \mathbb{E}[(X-\mathbb{E}X)(Y-\mathbb{E}Y)]$. The sample version averages the
products of centred values, which is exactly $X_c^\top X_c/(n-1)$.

Correlation is a cosine. Treat each centred column as a vector in $\mathbb{R}^n$. Then
$R_{jk} = \cos\theta_{jk}$ between them, because $\cos\theta = \dfrac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}$
and the $(n-1)$ factors cancel. In the example, the centred columns $(-2,0,2)$ and $(-3,-1,4)$ have
$\cos\theta = 14/\sqrt{8\cdot 26} \approx 0.971$, matching $R_{12}$. That is why correlation is
"covariance with the lengths removed" (`cml/cosine-similarity`).

## Signal vs. noise

- **Signal:** $R = D^{-1} C D^{-1}$ with $D = \operatorname{diag}(\sqrt{C_{jj}})$. Correlation is covariance in
  units of each variable's spread.
- **Signal:** correlation measures *linear* association only.
- **Noise for now:** the choice of divisor. It cancels in $R$. Choose $n-1$ and move on.

## Where it breaks

- **Covariance keeps units on wine data.** On the unscaled Wine set, the Proline variance is about
  $99{,}166.72$ and the Alcohol variance is about $0.659$. Their covariance is $164.57$, while their
  correlation is $0.644$. Reading covariance as a unitless association can produce values outside
  $[-1,1]$; divide by both standard deviations for correlation.
- **Anscombe's quartet.** Four small datasets share the same means ($\bar x = 9$, $\bar y = 7.50$), the same
  variance of $y$ (4.12 to 4.13), and the same correlation to two decimals ($r \approx 0.82$). Dataset I is a noisy
  line. In dataset IV, ten points sit at $x = 8$ and one high-leverage point sits at $x = 19$, and that one
  point alone creates the correlation. Always plot before trusting $R$.
- **Outliers.** On $(1,2),(2,1),(3,4),(4,3),(5,5)$ the Pearson correlation is $0.80$. Adding one point at
  $(20,20)$ raises it to $0.99$. Spearman, the Pearson correlation of the ranks, moves from $0.80$ to $0.89$.
- **Monotone but curved.** For $y = x^3$ on $x = 1,\dots,5$, Pearson gives $0.943$ and Spearman gives $1$.
- **Symmetric dependence.** For $x = (-2,-1,0,1,2)$ and $y = x^2$, the covariance and correlation are exactly
  $0$, although $y$ is a function of $x$.
- **Constant columns.** Correlation divides by a standard deviation that is zero, so the answer is undefined.
  p02 returns `NaN` by convention.
- **Causation.** A large correlation between two columns is a fact about the data, not a mechanism. A
  confounder can produce it.

## Interview angle

"What is the difference between covariance and correlation, and when do you normalise?" Answer shape:
covariance is the average product of centred values, with units equal to the product of the two units.
Correlation divides by both standard deviations, so it is unitless and lies in $[-1,1]$. Use it to compare
pairs of variables, and use covariance when the scale matters, as in a portfolio's variance. Follow-ups:
"Is a covariance matrix always positive semi-definite?" (Yes, because $\mathbf{v}^\top C\mathbf{v} = \operatorname{Var}(X\mathbf{v})$,
and it is singular when columns are collinear.) "Pearson or Spearman?" (Pearson for linear association,
Spearman when the relation is monotone but not linear or outliers dominate.) Full answers are in
`interview.md`.

## Connections

- **Prerequisites:** `cml/expectation-and-variance`, `cml/matrix-multiplication`.
- **Unlocks:** `cml/normal-and-multivariate-gaussian` (the covariance is the Gaussian's shape),
  `cml/eigenvalues-and-eigenvectors` (covariance eigenvectors are the principal directions),
  `cml/cosine-similarity` (correlation is a cosine of centred columns).
