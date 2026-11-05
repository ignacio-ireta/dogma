# Linear Programming: Models, Polyhedra and Simplex

**Historical anchor:** George B. Dantzig's chapter XXI in *Activity Analysis of Production and Allocation* (1951). The source is the complete proceedings edited by Tjalling C. Koopmans. This note reconstructs the mathematics with later teaching sources; the chapter's account of work in 1947 does not make the publication a 1947 document.

Linear programming chooses continuous decision variables using a linear objective and finitely many affine constraints. Consider divisible production quantities:
$$
\max_{x,y\ge0}3x+2y
\quad\text{subject to }x+y\le4,\quad x\le2.
$$
The coefficients represent fixed returns and resource consumption. Additivity, constant marginal coefficients, and divisibility are modeling assumptions. Setup costs, products of decisions, and indivisible items require another formulation. Uncertain coefficients also require an explicit uncertainty model; solving an LP exactly does not validate its inputs.

The feasible region is a **polyhedron**, an intersection of finitely many halfspaces and hyperplanes. If bounded, it is a polytope. A linear objective has parallel level sets, so an optimum can lie on an entire face. Existence of one optimal vertex does not imply uniqueness or that every optimizer is a vertex. General polyhedra can contain lines and have no vertices; vertex claims need the appropriate standard-form setting.

Introduce slack variables:
$$
x+y+s_1=4,\qquad x+s_2=2,
\qquad x,y,s_1,s_2\ge0.
$$
This is $Az=b$, $z\ge0$, with two independent equations. A **basis** selects two independent columns of $A$. Setting nonbasic variables to zero and solving the remaining square system gives a basic solution; nonnegativity makes it basic feasible. This reuses linear independence and solving matrix equations. In standard form, after redundant equations are removed, a nonempty feasible LP with finite optimum has an optimal basic feasible solution.

Starting with the slack basis gives $(x,y)=(0,0)$. Increase $x$: the constraints allow $x\le4$ and $x\le2$, so $s_2$ reaches zero first. Pivoting gives $(2,0)$ and value $6$. Increasing $y$ next exhausts the remaining shared resource at $(2,2)$, giving value $10$. A basis exchange updates which constraints determine the current point; reduced costs determine whether a nonbasic variable can improve the objective. The ratio test limits the step so feasibility is preserved.

The arithmetic also supplies an independent check:
$$
3x+2y=2(x+y)+x\le2\cdot4+2=10.
$$
The feasible point $(2,2)$ attains that bound. The coefficients $2$ and $1$ are resource prices: they are a [dual certificate](1994%20--%20Goemans%20--%20Linear%20Programming%20-%20Duality%20and%20Alternative%20Certificates%20--%20Study%20Note.md), linking simplex to the existing [KKT treatment](2004%20--%20Boyd%20and%20Vandenberghe%20--%20Convexity%2C%20Duality%20and%20KKT%20Certificates%20--%20Study%20Note.md).

Degeneracy occurs when a basic variable is zero. A pivot can then change the basis without moving the point; careless pivot rules may cycle. Anti-cycling rules address termination, while common simplex variants still have exponential worst-case examples. This is compatible with useful practical performance and with polynomial-time [interior-point methods](1984%20--%20Karmarkar%20--%20Interior-Point%20Methods%2C%20Complexity%20and%20Practical%20Limits%20--%20Study%20Note.md).

Finally, continuous and integer models differ. Replacing $x\le2$ by $x\le5/2$ yields continuous optimum $(5/2,3/2)$ with value $21/2$. Requiring integer production gives optimum $(2,2)$ with value $10$. General integer linear programming is NP-hard; polynomial-time continuous LP provides a relaxation bound, not an automatic integer solver. Special integral polyhedra can close this gap, but integrality must be proved.

## Sources and locators

- Koopmans, ed., Activity Analysis, Dantzig, chapter XXI, printed pp.339–347. [Official Cowles record](https://cowles.yale.edu/research/cfm-13-activity-analysis-production-and-allocation); original historical exposition.
- Ferguson, Linear Programming: A Concise Introduction, chapters 1, 3–4: modeling, pivots, simplex. [Author-hosted text](https://www.math.ucla.edu/~tom/); undated teaching exposition.
- Goemans, Linear Programming, printed LP-1–LP-10: polyhedra and simplex. October 1994 notes, reused with an OCW 2008 cover; not original discovery papers.
- Boyd and Vandenberghe, Convex Optimization, §4.3; Introduction to Applied Linear Algebra, chapters 5 and 11: independence and linear equations.
