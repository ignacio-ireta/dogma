# Pareto Search, NSGA-II and Honest Benchmarking

**Works:** Deb, Pratap, Agarwal and Meyarivan, *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II* (2002); Wolpert and Macready, *No Free Lunch Theorems for Optimization* (1997). Luke supplies a broader methodological reference. The NSGA-II copy used is the complete, undated KanGAL Report 200001. The 2002 journal publication is a distinct manifestation; this file is not labeled as its final typesetting.

For objectives $f(x)=(f_1(x),\ldots,f_M(x))$ all minimized, $x$ dominates $y$ if $f_j(x)\le f_j(y)$ for every $j$, with at least one strict inequality. A feasible point is Pareto optimal when no feasible point dominates it. A population's nondominated subset approximates a tradeoff set; it need not equal the true Pareto set, and it does not choose the user's priorities.

Consider the illustrative objective vectors $A=(1,5)$, $B=(3,4)$, $C=(5,1)$ and $D=(4,5)$. $B$ dominates $D$; $A,B,C$ are mutually nondominated. Ranking only the first coordinate would discard $C$, although it is best on the second. A scalar objective introduces a preference that dominance alone leaves unspecified.

Even varying weights can miss tradeoffs. Minimize $wf_1+(1-w)f_2$ with $0\le w\le1$. The three values are $5-4w$, $4-w$ and $1+4w$. For $B$ to beat both endpoints requires $w\le1/3$ and $w\ge3/5$, which is impossible. Thus this nondominated point is unsupported by any such weighted sum. General recovery claims need assumptions on the feasible objective image; positive rescaling also changes the meaning of fixed weights.

NSGA-II combines parents and offspring, sorts by nondomination into successive fronts, and fills the next population by increasing front rank. When a front will not fit, it favors larger crowding distance. For a nonboundary point $i$, the normalized contribution from objective $j$ is
$$
\frac{f_j(i+1)-f_j(i-1)}{f_j^{\max}-f_j^{\min}},
$$
using neighbors in that objective's ordering and summing across objectives. Boundary points receive infinite distance; a zero objective range needs explicit handling. This is a diversity preference within a front, not an estimate of distance to the true optimum.

For the three-point front $A,B,C$, each objective range is four, and $B$ receives $4/4+4/4=2$; $A,C$ are boundary points. If only two places remain, this crowding rule keeps the endpoints and loses $B$. A finite archive can therefore omit a valid compromise without any error in its dominance calculation.

The original sorting procedure costs $O(MN^2)$ for population size $N$, apart from objective evaluation. Many objectives can make most candidates nondominated, weakening this ranking signal. Constraints require a declared rule; the paper also proposes constrained domination.

“No free lunch” needs its assumptions. For a finite search space $X$, finite value set $Y$, uniform averaging over all functions $f:X\to Y$, and nonrevisiting search, the distribution of observed value sequences after a fixed number of queries is algorithm-independent. This does not say that optimizers tie on engineering problems, smooth functions, or a chosen benchmark distribution. With $X=\{a,b\}$ and $Y=\{0,1\}$, the four equally likely functions give any first query probability $1/2$ of finding zero; visiting both raises the probability to $3/4$.

Practical comparisons should declare problem classes, seeds, budgets, constraint handling, stopping rules and tuning cost. For Pareto approximations, report coverage and a metric such as hypervolume with its reference point and scaling. Retain failures and distributions, not just the best seed. [Optimality certificates](../../01-Mathematics/08-Optimization/2004%20--%20Boyd%20and%20Vandenberghe%20--%20Convexity%2C%20Duality%20and%20KKT%20Certificates%20--%20Study%20Note.md) explain a different standard of evidence: a matching bound can certify optimality, whereas a favorable heuristic benchmark supports a scoped empirical claim.

## Sources and locators

- Kalyanmoy Deb — A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II: §III, nondominated sorting, crowding distance and the main loop; constrained domination discussion. [Coauthor publication record](https://homes.cs.washington.edu/~sagarwal/papers.html).
- David H. Wolpert and William G. Macready — No Free Lunch Theorems for Optimization: §§II–III, especially Theorem 1; §V and Appendix E discuss benchmarks. [Publisher identifier](https://doi.org/10.1109/4235.585893).
- Sean Luke — Essentials of Metaheuristics: multiobjective optimization and experimental-methodology chapters. [Official author book page](https://people.cs.gmu.edu/~sean/book/metaheuristics/).
