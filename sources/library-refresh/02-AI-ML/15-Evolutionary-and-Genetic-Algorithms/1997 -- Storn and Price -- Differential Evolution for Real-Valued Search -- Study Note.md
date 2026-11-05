# Differential Evolution for Real-Valued Search

**Work:** Rainer Storn and Kenneth Price, *Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces*, *Journal of Global Optimization* 11, 341–359 (1997). The copy used is the full published article, rather than the earlier 1995 report. Luke supplies a broader modern comparison of population methods.

Differential evolution (DE) searches directly over a population of real vectors. Its proposal step uses differences between existing candidates, adapting the scale of perturbations to population spread without requiring derivatives. Despite the word “differential,” it does not evaluate objective gradients or Hessians.

For minimization, the standard DE/rand/1/bin variant uses a generation $x_1,\ldots,x_N\in\mathbb R^d$. For each target $x_i$, draw mutually distinct indices $r_1,r_2,r_3$, all different from $i$, requiring $N\ge4$. With differential weight $F>0$, form a donor
$$
v_i=x_{r_1}+F(x_{r_2}-x_{r_3}).
$$
For independent uniform draws $U_j$, crossover rate $CR\in[0,1]$, and a uniformly chosen forced coordinate $j_*$, form a trial
$$
u_{ij}=\begin{cases}
v_{ij},&U_j\le CR\ \text{or}\ j=j_*,\\
x_{ij},&\text{otherwise}.
\end{cases}
$$
Replace the target if the trial is no worse: $x_i'=u_i$ when $f(u_i)\le f(x_i)$, otherwise $x_i'=x_i$. Some implementations reject ties; that convention should be recorded. Donor sampling from a frozen generation differs from immediately reusing accepted replacements.

Here is a complete illustrative proposal. Minimize $f(a,b)=a^2+b^2$ on $[0,5]^2$, and choose target $(4,2)$ with three other distinct population members $(1,1),(3,0),(1,2)$. For $F=0.5$, the donor is
$$
(1,1)+0.5[(3,0)-(1,2)]=(2,0).
$$
Set $CR=0.5$, force coordinate two, and draw $U_1=0.9$. The trial is $(4,0)$, with value $16$, versus the target's $20$, so it survives. Had coordinate one also crossed, $(2,0)$ would have value $4$. The better donor is not automatically the evaluated trial: crossover changes the proposal.

The forced coordinate guarantees a donor coordinate is used, but not that the numerical vector changes; donor and target coordinates can coincide. If the population collapses to one vector, all differences vanish. Increasing the number of generations then need not restore exploration. This explains why population diversity, restarts and an explicit evaluation budget belong in the experimental description.

A donor can violate bounds even when every population member is feasible. Clipping, reflection, resampling, or constraint-aware selection produce different algorithms and different sampling biases. Ordinary DE also does not preserve integer, permutation or coupled equality constraints automatically. For example, clipping $(6,-1)$ to $[0,5]^2$ yields $(5,0)$, creating boundary mass that an unconstrained donor distribution would not have.

The donor equation respects consistent affine transformations of vectors, but coordinatewise binomial crossover generally breaks invariance to rotations. Thus a rotated benchmark can probe a real structural weakness rather than merely change notation. On noisy objectives, accepting a trial from one lucky observation can reverse the true ordering; repeated evaluation consumes additional budget.

For a deterministic unconstrained run, initialization costs $N$ evaluations and each complete generation ordinarily adds $N$. Compare algorithms at the same evaluation budget, including repairs that invoke the objective. An observed minimum supplies an incumbent, while [optimality certificates](../../01-Mathematics/08-Optimization/2004%20--%20Boyd%20and%20Vandenberghe%20--%20Convexity%2C%20Duality%20and%20KKT%20Certificates%20--%20Study%20Note.md) require additional structure and bounds. Compare [CMA-ES covariance adaptation](2016%20--%20Hansen%20--%20Evolution%20Strategies%20and%20CMA-ES%20Covariance%20Adaptation%20--%20Study%20Note.md) to see a different way to learn continuous search geometry.

## Sources and locators

- Rainer Storn and Kenneth Price — Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces: §2, printed pp.343–346, equations (1)–(4), and the DE/rand/1/bin nomenclature; §3 supplies the original finite benchmark comparison. [Publisher identifier](https://doi.org/10.1023/A:1008202821328).
- Sean Luke — Essentials of Metaheuristics: chapter 3, differential evolution discussion, and experimental-methodology chapter. [Official author book page](https://people.cs.gmu.edu/~sean/book/metaheuristics/).
