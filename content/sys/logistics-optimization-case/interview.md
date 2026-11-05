# Interview notes — Logistics optimization system

## Q1. Why would you formulate dispatch as a linear program rather than a heuristic?
**60-second answer.** Dispatch is an allocation of pallets and trucks per lane, under depot capacity and zone
demand. A linear program gives an exact answer and duals: the duals say which depots are scarce, and they certify
optimality, since a plan and its duals with equal objectives prove the plan is best. Trucks come in whole loads, so the
dispatched plan is a mixed-integer program, and the LP becomes the bound and the sensitivity tool. I would start with the
LP to check feasibility and get the duals, then solve the MILP for the plan.
**Follow-ups.** When is a heuristic enough? When the MILP is too slow for the timetable and the gap to the bound is small
and known. What does a dual price tell the business? The value of one more pallet of a constraint, valid within a range.

## Q2. A depot closes after the plan is set. What do you do in the next fifteen minutes?
**60-second answer.** I first test feasibility: compare total demand with the remaining capacity. If demand is larger, I
add a shortfall variable with a penalty so the model still returns a plan. I re-solve the truck model with the new capacity,
read the shadow prices to see which depots are now scarce, keep the loads already built where they fit, and cap the number
of changed trucks against the dispatchers' tolerance.
**Follow-ups.** Why not just reassign the closed depot's pallets to the nearest depot? That ignores capacity and truck charges
and can create new shortfalls elsewhere. Where does the shortfall fall? With one penalty, the solver chooses the cheapest pattern of shortfall by lane cost. The
pallet LP puts all of it on Zone 5, the zone most expensive to serve without the closed depot; the whole-truck plan
spreads it across zones. Zone priorities are a business decision and belong in the penalties.

## Q3. When would you use evolutionary search instead of an exact solver, and how would you compare them?
**60-second answer.** Evolutionary search fits a black-box objective, such as a simulator with congestion, or constraints that
are awkward to write linearly. It gives no optimality certificate and its results vary by seed. I would compare it against the
exact solver or a bound on the same instances, with many seeds, equal evaluation budgets, and the starting point reported,
because a warm start can dominate the method.
**Follow-ups.** Why not gradient descent? The LP and MILP are not differentiable in the way gradient methods need, and with
constraints they need penalties or projection. What would make you trust the result? Replicated seeds, a stated budget, and a
comparison with a bound.

## Common mistakes
- Reporting the pallet LP as the plan when trucks are whole, which understates cost by about a third here.
- Quoting one dual price for a degenerate constraint, when a range of values is valid.
- Comparing an evolutionary run with an exact solver on different evaluation budgets, or on one seed.
- Re-solving from scratch after a closure without checking how many loaded trucks change.
