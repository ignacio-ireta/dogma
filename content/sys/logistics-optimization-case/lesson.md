---
title: Logistics optimization system
minutes: 12
min_viable: "Read The system and Where it breaks, then the brief and the twist."
---

## The system

Daily dispatch planning is an allocation problem: how many pallets each depot sends to each zone, and how many
trucks carry them. Write it in four parts.

1. **Variables.** $x_{dz} \ge 0$ is the pallets sent from depot $d$ to zone $z$. $n_{dz}$ is the number of whole
   trucks on that lane.
2. **Constraints.** Each zone's demand is met, $\sum_d x_{dz} = D_z$. Each depot's trucks fit its slots. Each
   lane's pallets fit its trucks: $x_{dz} \le 40\, n_{dz}$.
3. **Objective.** Pallet cost plus a fixed charge per truck: $\sum_{d,z} c_{dz} x_{dz} + 220 \sum_{d,z} n_{dz}$.
4. **Type.** With $n$ continuous this is a linear program (LP). With $n$ integer it is a mixed-integer program
   (MILP). Stop order inside a zone is a routing problem and is out of scope.

## Worked example

Depots X and Y have capacity 100 and 80 pallets. Zone 1 needs 90 and Zone 2 needs 60. Costs per pallet:
X to Zone 1 is 10, X to Zone 2 is 14, Y to Zone 1 is 12, Y to Zone 2 is 11.

Feasible? Demand is 150 and capacity is 180, so yes. The cheapest plan sends 90 from X to Zone 1 and 60 from Y
to Zone 2: cost $900 + $660 = $1{,}560$.

The certificate: zone duals $v_1 = 10$ and $v_2 = 11$, and depot duals $u_X = u_Y = 0$, since neither depot is
full. Each lane's reduced cost is $c_{dz} - v_z + u_d$. Used lanes have zero reduced cost. Unused lanes do not:
X to Zone 2 gives $14 - 11 = 3$ and Y to Zone 1 gives $12 - 10 = 2$. The dual objective is
$90 \times 10 + 60 \times 11 = 1{,}560$, equal to the primal. That equality is the proof.

## Why it works

An LP is the right first model: its answer comes with a proof. Three properties decide how to read it.

- **Feasibility.** Summing all zone demands and comparing them with all depot capacities is a certificate of
  infeasibility. If demand exceeds capacity, no plan exists. Add a shortfall variable with a penalty so the
  model returns a plan and states the miss.
- **Boundedness.** Costs and flows are nonnegative, so the transport LP cannot be unbounded. If a solver reports
  unbounded, the model has a bug, such as a negative-cost cycle or a missing constraint.
- **Duals and degeneracy.** A dual price is the change in cost per extra pallet of demand or capacity, valid only
  within a range. At a degenerate point the dual itself is not unique: the solver returns one of several valid
  values. In the case instance, Zone 4's demand equals Depot D's capacity (240 pallets). One more pallet costs
  15 dollars and one fewer saves 14, so every value from 14 to 15 is a valid dual, and the solver returned 14.

**Choosing a method.** Use the LP when the model is linear and you need duals. Use the MILP when whole trucks
change the answer. Gradient methods need a smooth objective; with constraints they need penalties or projection,
and they give no optimality certificate. Evolutionary search suits a black-box objective, such as a simulator. It
has no certificate, and its results vary by seed. Compare methods on equal evaluation budgets over many seeds.
No method wins everywhere. In the case pilot, a random-start evolutionary search had a median gap of 5.4 percent,
and the same search started from the dispatcher plan had 0.05 percent.

## Signal vs. noise

**Signal:** total demand against total capacity, the lane costs, the fixed truck charge, the shortfall penalty,
and the duals, which show which depots are scarce.

**Noise:** solver settings before the model is agreed, and a gradient method applied to a linear model.

## Where it breaks

- **Linear approximations.** Congestion and driver-hour rules are not linear. A linear stand-in gives a confident,
  wrong plan.
- **Integrality.** The pallet LP understated the case day's cost by about a third (14,300 against 20,930 dollars)
  because it omitted truck charges. Report the model you dispatch from.
- **Equal penalties.** With one penalty for every zone, the solver picks which zones go short by lane cost, not
  by business priority. In the case's twist the pallet LP puts the whole shortfall on Zone 5, while the truck
  MILP, which must fill whole trucks, spreads it across all six zones. Set zone priorities explicitly.
- **Seed luck.** A single evolutionary run is an anecdote. Report medians, worst seeds and the budget.

## Interview angle

Sixty-second shape: "It is an allocation of pallets and trucks per lane, under depot capacity and zone demand. I
model it as an LP first, because its duals show which depots are scarce, then as a MILP for whole trucks, with a
shortfall variable for feasibility. For a depot outage I re-solve, read the shadow prices, and cap the change to
loaded trucks. I use evolutionary search only where the objective is a black box, and I compare it on equal
budgets across many seeds." Expect the follow-ups "why not gradient descent?" and "what does a dual price mean for
the business?"

## Connections

Builds on linear programming and duality. A Lagrange multiplier is a shadow price in another setting. The
predictive-maintenance case also ranks under a budget, but there the question is the model's value, not an exact
optimum.
