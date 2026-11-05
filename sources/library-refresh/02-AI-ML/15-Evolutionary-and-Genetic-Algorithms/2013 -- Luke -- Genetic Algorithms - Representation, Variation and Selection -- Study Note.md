# Genetic Algorithms: Representation, Variation and Selection

**Works:** Sean Luke, *Essentials of Metaheuristics*, second edition (recommended citation 2013), online version 2.4, January 2026; Darrell Whitley's *A Genetic Algorithm Tutorial*, author manuscript associated with the 1994 article. These explain the field; neither copy used is Holland's 1975 book.

A genetic algorithm searches through a population of encoded candidates. An encoding maps a genotype to a candidate decision, and evaluation assigns objective values. Feasibility must be enforced or checked separately. Selection chooses parents; crossover combines their representations; mutation perturbs them; replacement constructs the next population. An evolution strategy is a related family, commonly using real vectors and adaptive mutation distributions. The terms are not interchangeable.

For a maximization example, let a four-bit string encode four independent yes/no choices and set $q(s)=1+\sum_j s_j$. The population $1010,1111,0010,1100$ has scores $3,5,2,3$. Fitness-proportionate parent selection gives probabilities $3/13,5/13,2/13,3/13$: over four draws, the expected number of selections of $1111$ is $20/13$, not a guarantee that it appears. This rule requires nonnegative scores with positive total; arbitrary minimization costs need a justified transformation or a different selector.

Tournament selection instead samples competitors and picks the best, avoiding proportional scaling. Increasing tournament size increases selection pressure but can discard useful diversity. Elitism retains some best-so-far candidates. In a deterministic objective this can preserve the best recorded value while most of the population deteriorates or becomes identical.

Here is an illustrative variation step. Cutting parents $1111$ and $0010$ after the second position produces $1110$ and $0011$. Flipping the fourth bit of the first child restores $1111$. This toy objective makes improvements easy to recognize; it gives no evidence about deceptive landscapes. On a routing problem, ordinary bitwise crossover could produce an invalid tour: the representation and repair policy become part of the optimizer.

For bit strings of length $\ell>1$, a schema $H$ fixes some positions and leaves others free. Its order $o(H)$ counts fixed positions; its defining length $\delta(H)$ spans their first and last positions. Under proportional selection, one-point crossover probability $p_c$, and independent bit mutation probability $p_m$, the familiar survival bound is
$$
\mathbb E[m(H,t+1)\mid P_t]\ge
m(H,t)\frac{\bar q_H}{\bar q}
\left(1-p_c\frac{\delta(H)}{\ell-1}\right)(1-p_m)^{o(H)}.
$$
Here $m$ is the schema count, and the bars are current mean scores. For $H=11**$, $\ell=4$, $p_c=0.6$, $p_m=0.01$, the variation-survival factor is at least $0.8(0.99)^2=0.78408$. Being above average is insufficient for an expected increase unless the fitness ratio offsets this loss.

The bound ignores newly reconstructed instances, conditions on the current population, and does not establish that useful schemata remain useful as their backgrounds change. It supplies neither a runtime bound nor a universal global convergence theorem. See Whitley's explicit criticism and exact-model discussion.

Use evaluations, feasibility checks, seed variation, and held-out problem instances to assess results. A good discovered candidate is different from an [optimality certificate](../../01-Mathematics/08-Optimization/2004%20--%20Boyd%20and%20Vandenberghe%20--%20Convexity%2C%20Duality%20and%20KKT%20Certificates%20--%20Study%20Note.md); even a known LP's best observed feasible value lacks a proof of optimality without a matching bound. Continue with [CMA-ES covariance adaptation](2016%20--%20Hansen%20--%20Evolution%20Strategies%20and%20CMA-ES%20Covariance%20Adaptation%20--%20Study%20Note.md) for adapting continuous search geometry.

## Sources and locators

- Sean Luke — Essentials of Metaheuristics: chapter 3, especially §§3.2–3.3, for GA operators, selection and elitism. [Official author book page](https://people.cs.gmu.edu/~sean/book/metaheuristics/).
- Darrell Whitley — A Genetic Algorithm Tutorial: canonical GA, hyperplane/schema sampling, schema theorem and its limitations. [Author publication record](https://www.cs.colostate.edu/~whitley/research/publications.php). The manuscript copy used has different pagination from the journal article.
