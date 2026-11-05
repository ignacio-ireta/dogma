# Linear Programming: Duality and Alternative Certificates

**Work studied:** Michel X. Goemans, *Linear Programming*, MIT Advanced Algorithms lecture notes, October 1994. The copy studied includes a later OCW 2008 cover. This is a modern explanation of established duality, not an attribution of Farkas' lemma or strong duality to Goemans.

A numerical objective value does not explain why improvement is impossible. Linear programming offers compact witnesses for three different outcomes: an optimum, infeasibility, or an improving feasible ray. Keeping their assumptions separate prevents a failed search from being mistaken for a mathematical conclusion.

Use the primal–dual pair
$$
(P)\quad\min\{c^\top x:Ax=b,\ x\ge0\},
\qquad
(D)\quad\max\{b^\top y:A^\top y\le c\}.
$$
Here $y$ is unrestricted in sign because the primal constraints are equalities. Set $s=c-A^\top y\ge0$. For any feasible pair,
$$
c^\top x-b^\top y=x^\top s\ge0.
$$
This proves weak duality directly: a dual feasible point supplies a lower bound. If the gap is zero, both points are globally optimal. No search history or trust in a solver's pivot choices is needed.

For a small example, minimize $x_1+2x_2$ subject to $x_1+x_2=1$ and $x\ge0$. The candidate $x=(1,0)$ costs $1$. The dual has $y\le1$ and $y\le2$; choose $y=1$, yielding $s=(0,1)$. The bounds meet at $1$, certifying optimality. Moreover, $x_1s_1=x_2s_2=0$: each positive primal variable has zero corresponding dual slack.

Strong duality adds existence: a feasible LP with finite optimal value has attained primal and dual optima with equal values. Strict positivity of every slack is unnecessary. This is a stronger structural fact than simply applying the strict version of Slater's condition. The existing [convex duality note](2004%20--%20Boyd%20and%20Vandenberghe%20--%20Convexity%2C%20Duality%20and%20KKT%20Certificates%20--%20Study%20Note.md) supplies the general framework. Here the KKT equations reduce to feasibility, $c-A^\top y-s=0$, and complementary slackness $x_is_i=0$; affine geometry makes these global certificates.

**Infeasibility uses a different witness.** Farkas' alternative says exactly one of the following holds:
$$
\exists x\ge0:Ax=b;
\qquad
\exists w:A^\top w\ge0,\ b^\top w<0.
$$
If both held, $b^\top w=x^\top A^\top w\ge0$ would contradict the strict inequality. For $x_1+x_2=-1$, $x\ge0$, choose $w=1$: $A^\top w=(1,1)$ and $b^\top w=-1$. This proves impossibility, rather than reporting that iterations found no solution.

**Unboundedness requires feasibility plus a direction.** Find $x_0\ge0$ with $Ax_0=b$ and $d\ge0$ with $Ad=0$, $c^\top d<0$. Then $x_0+td$ is feasible for every $t\ge0$ and its objective tends to $-\infty$. For $x_1-x_2=0$ and objective $-x_1$, take $x_0=(0,0)$, $d=(1,1)$. A direction alone does not establish that the feasible set is nonempty. Dual infeasibility alone also does not establish primal unboundedness: both problems can be infeasible.

In floating point, verify primal residuals, dual residuals, signs, and the gap with tolerances scaled to the data. Tiny negative entries or a small residual are approximate evidence, not exact Farkas contradictions. Rational reconstruction or exact checking may be appropriate when a formal certificate matters.

These certificates apply to continuous LP. A dual bound for an integer minimization model certifies its relaxation's lower bound; it proves integer optimality only when an integer feasible solution attains that bound, or further valid bounds close the gap. Heuristic search supplies candidates; certificates establish what their objective values mean.

## Sources and locators

- Goemans, Linear Programming, printed LP-11–LP-18 (PDF pp.12–19): Farkas' lemma, duality, complementary slackness. [Author's teaching provenance](https://math.mit.edu/~goemans/teaching.html).
- Ferguson, A Concise Introduction, chapter 2: duality and equilibrium theorem. [Author-hosted complete text](https://www.math.ucla.edu/~tom/).
- Boyd and Vandenberghe, Convex Optimization, §5.2.3, pp.226–227, refined Slater and LP; §5.5.3, pp.243–246, KKT.

Continue to [interior-point methods](1984%20--%20Karmarkar%20--%20Interior-Point%20Methods%2C%20Complexity%20and%20Practical%20Limits%20--%20Study%20Note.md), where the same gap becomes a stopping criterion.
