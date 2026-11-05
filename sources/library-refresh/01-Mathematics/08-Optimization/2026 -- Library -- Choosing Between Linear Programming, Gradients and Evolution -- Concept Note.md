# Choosing Between Linear Programming, Gradients and Evolution

An optimization method is a way to exploit a mathematical model. Start by identifying the variables, objective, constraints, available information, and evaluation cost. An algorithm's familiar name is not evidence that it fits those structures.

## What structure is actually available?

For a continuous linear objective with affine constraints, [linear programming](1951%20--%20Dantzig%20--%20Linear%20Programming%20-%20Models%2C%20Polyhedra%20and%20Simplex%20--%20Study%20Note.md) exposes a feasible polyhedron and supplies dual bounds. Derivatives are not the missing ingredient: the coefficients already describe the entire model. Simplex and interior-point methods exploit that description. Requiring integer decisions changes the problem class; an LP relaxation can still give useful bounds without solving the integer problem.

For a differentiable objective, gradients provide local directional information. Convexity may turn suitable stationarity conditions into global certificates; a nonconvex loss generally does not. Gradient computation and convergence to a useful solution are separate issues. A numerical derivative, an automatic derivative, and an unbiased stochastic gradient estimator also have different assumptions and costs.

When a simulator only returns a score, a black-box method can compare sampled candidates without differentiating the simulator. [Genetic search](../../02-AI-ML/15-Evolutionary-and-Genetic-Algorithms/2013%20--%20Luke%20--%20Genetic%20Algorithms%20-%20Representation%2C%20Variation%20and%20Selection%20--%20Study%20Note.md) varies encoded decisions; evolution strategies adapt real-valued perturbations; differential evolution uses population differences. Their evaluation budgets can be large. Discreteness alone does not imply that a GA is the best choice: integer programming, dynamic programming, or problem-specific algorithms may exploit structure that a generic encoding discards.

## Put a candidate and a certificate on the same scale

Consider

\[
\max_{x,y\ge0}3x+2y,\qquad x+y\le4,\quad x\le2.
\]

Suppose a population search returns the feasible candidate (1.8,2.2), with value 9.8. Its search history does not prove that no better point exists. But the linear inequalities give

\[
3x+2y=2(x+y)+x\le2\cdot4+2=10.
\]

The candidate is within 0.2 of a valid upper bound. The feasible point (2,2) attains 10 and closes the gap. The bound is supplied by the model's dual structure, regardless of how either candidate was found. A hybrid search can combine a heuristic incumbent with a certified relaxation bound; they answer different parts of the problem.

If the simulator violated those linear assumptions, this bound would no longer apply. If x and y had to be integers, (1.8,2.2) would not be a valid incumbent. Rounding is another proposed decision and must be checked against every constraint.

## Feasibility, uncertainty, and tradeoffs

A finite penalty does not generally turn constraint violations into impossibilities. Keep the original feasibility test visible, and state repair, rejection, or constraint-ranking rules. On noisy evaluations, one lucky score is weak evidence; repeated evaluations and held-out seeds consume part of the budget.

With conflicting objectives, [Pareto optimization](../../02-AI-ML/15-Evolutionary-and-Genetic-Algorithms/2002%20--%20Deb%20et%20al%20--%20Pareto%20Search%2C%20NSGA-II%20and%20Honest%20Benchmarking%20--%20Study%20Note.md) describes dominance and diversity. A nondominated population need not equal the true Pareto set. A weighted sum adds preferences and can miss unsupported efficient points; changing objective units changes a fixed weighting's meaning.

Record the final feasible value, any valid bound and gap, objective evaluations, compute time, failures, seeds, tuning cost, and the problem distribution. The relevant standard is whether the mathematical claim and the empirical evidence answer the decision being made.

## Sources

- Boyd and Vandenberghe, Convex Optimization: §4.3 and chapter 5, linear models and dual certificates.
- Goemans, Linear Programming: simplex, alternatives, encoding length, and interior-point methods.
- Luke, Essentials of Metaheuristics: search operators, constraint handling, and experimental methodology. This is a synthesis note and a worked example, not a benchmark claiming one method universally wins.
