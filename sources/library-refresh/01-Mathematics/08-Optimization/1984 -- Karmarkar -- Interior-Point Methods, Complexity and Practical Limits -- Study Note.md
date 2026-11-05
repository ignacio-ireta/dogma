# Interior-Point Methods, Complexity and Practical Limits

**Historical anchor:** Narendra Karmarkar, *A New Polynomial-Time Algorithm for Linear Programming*, Combinatorica 4(4), 373–395 (1984). The complete journal article substantially revises the earlier STOC paper. Its projective method is an original research contribution. The logarithmic-barrier and primal–dual explanation below uses later textbook machinery; it is not a transcription of Karmarkar's exact algorithm.

[Simplex](1951%20--%20Dantzig%20--%20Linear%20Programming%20-%20Models%2C%20Polyhedra%20and%20Simplex%20--%20Study%20Note.md) moves among bases associated with boundary points. Interior-point methods approach optimality through the interior. Karmarkar's polynomial-time result followed the earlier ellipsoid result; it should not be described as the first proof that rational LP is polynomial-time solvable. This historical sequence does not imply that one algorithm dominates every practical instance.

For the standard-form problem
$$
\min c^\top x\quad\text{subject to }Ax=b,\quad x\ge0,
$$
assume $A$ has full row rank and both primal and dual have strictly feasible points. These assumptions support the clean central-path account; they are not prerequisites for every LP to have an optimum. Replace the nonnegativity boundary by a logarithmic barrier:
$$
\min_{Ax=b,\ x>0}\ c^\top x-\mu\sum_{i=1}^n\log x_i,
\qquad\mu>0.
$$
At its minimizer, stationarity gives $A^\top y+s=c$ with $s_i=\mu/x_i>0$. Thus
$$
Ax=b,\quad A^\top y+s=c,\quad x_is_i=\mu.
$$
The first two equations are feasibility; the third relaxes [complementary slackness](1994%20--%20Goemans%20--%20Linear%20Programming%20-%20Duality%20and%20Alternative%20Certificates%20--%20Study%20Note.md). Their solutions trace the central path as $\mu$ decreases. The duality gap is exactly
$$
c^\top x-b^\top y=x^\top s=n\mu.
$$
For an explicit check, minimize $x_1+2x_2$ subject to $x_1+x_2=1$. With $\mu=3/8$, take $x=(3/4,1/4)$, $y=1/2$, and $s=(1/2,3/2)$. Both feasibility equations hold; both products $x_is_i$ equal $3/8$. The objective is $5/4$, the dual bound is $1/2$, and their gap $3/4$ equals $2\mu$. As $\mu$ tends to zero, this path approaches the optimum $(1,0)$, where strict positivity is lost.

Computing successive path points uses Newton linear algebra. Linearizing the perturbed KKT equations gives systems containing $A\Delta x$, $A^\top\Delta y+\Delta s$, and $S\Delta x+X\Delta s$, where $X=\operatorname{diag}(x)$ and $S=\operatorname{diag}(s)$. Step lengths preserve positivity. Matrix factorization, sparsity, and conditioning therefore matter alongside the count of iterations. This connects the existing [KKT note](2004%20--%20Boyd%20and%20Vandenberghe%20--%20Convexity%2C%20Duality%20and%20KKT%20Certificates%20--%20Study%20Note.md) to a concrete solver mechanism.

Complexity statements need a computational model. For rational input, polynomial time refers to the encoded input length, including coefficient bit lengths, with controlled arithmetic precision. An iteration bound alone is incomplete unless the work and precision per iteration are bounded. Likewise, an approximate gap tolerance is different from an exact rational optimality certificate. Karmarkar's original analysis and the MIT notes address these distinctions; their historical bounds are not claims about the fastest current implementation.

Practical limits follow from the equations. Near a boundary, the barrier Hessian $\mu\operatorname{diag}(x_i^{-2})$ can be ill-conditioned. Redundant equalities require rank handling. A starting strictly feasible point may require a phase-I problem; assuming one exists is not an initialization algorithm. Implementations monitor feasibility residuals as well as the gap, and sparse factorizations can dominate memory and runtime.

Finally, interior-point LP solves the continuous relaxation of an integer model. General integer variants remain NP-hard; fractional solutions require additional valid reasoning or search. The useful comparison is problem structure, reliable bounds, and computational cost, rather than a universal ranking of simplex, interior-point, and heuristic methods.

## Sources and locators

- Karmarkar, original Combinatorica article, §1 and the formal algorithm/analysis; [university-hosted full journal PDF](https://www.stat.uchicago.edu/~lekheng/courses/302/classics/karmarkar.pdf).
- Goemans, Linear Programming, printed LP-21–LP-35 (PDF pp.22–36): encoding, certificates, interior points and bit complexity. The exposition uses Ye's method, not Karmarkar's projective algorithm.
- Boyd and Vandenberghe, Convex Optimization, §§11.2–11.5, printed pp.562–595; §§11.7–11.8, pp.609–620: central paths, initialization, complexity, primal–dual systems and implementation.
