# Logistics optimization system, debrief

## Reframe

Which depot should serve each zone, and how many trucks go on each lane, so that the next day's demand is met at
the lowest total cost, with any shortfall placed where the business chooses? Success is cost per day against the
best full-truck plan (20,930 dollars on the base day), the service level (the share of demand shipped on the
dispatch day), and the share of loaded trucks a re-plan changes. The LP is a model. The plan is judged by dollars and service.

## Ask

Ask whether demand and capacity are firm, because outages change capacity and forecasts change demand. Ask whether
trucks are whole and what each costs, since that decides integrality. Ask whether demand can go unmet, and how zones
are ranked when it does, because a single penalty makes the model choose the zone to short. Ask for the timetable and
the stability rule, which set the solver's time limit and the re-plan's objective. Ask whether routing is in scope,
and whether any other method has been tried, on what budget. Each answer changes the model, not only the wording.

## Structure

Model the allocation with pallets per lane and whole trucks per lane. Solve the LP first, for speed and duals, then the
MILP for the plan dispatch uses. On the base day the pallet LP costs 14,300 dollars, its fixed-charge relaxation 20,350,
and the MILP 20,930 with 30 trucks. The pallet LP leaves out truck charges, so it understates the day by about a third.
The relaxation is 2.8 percent below the optimum: that is a measure of model quality, a bound, not a plan.

Feasibility: total demand of 1,100 is below capacity of 1,320, so the plan is feasible. Keep a shortfall variable at 60
dollars a pallet so a later infeasible case still returns a plan. On the base day, depots B and D are full and carry
shadow prices of 1 and 2 dollars a pallet; A and C have spare capacity and zero price. Zone 4 sits on a kink: its demand
equals depot D's capacity, so one more pallet costs 15 dollars and one fewer saves 14, and the solver's dual of 14 is a
valid value inside that range. Report the range, not one number.

Method: the exact MILP solves the base day in under a tenth of a second in our reference run, so the constraint on a
re-plan is data and approval, not solver time. For this instance an evolutionary search with 20,000 evaluations and 30 seeds
had a median gap of 5.4 percent from random starts and 0.05 percent from the dispatcher plan, with 28 of 30 runs within
0.5 percent. The starting point matters more than the method, and no method wins on every instance.
A gradient method is not the first choice here: the objective is linear with integer truck steps, so
gradients do not exist at the steps, and the constraints do the real work. Use the evolutionary search
only where the objective is a black box, and report it on equal budgets.

## Synthesize

Model quality and decision value are different numbers. The relaxation's 2.8 percent gap is model quality. The dispatcher
plan's 5.4 percent excess is decision value: about 1,130 dollars a day on the base day, or about 340,000 dollars a year over
300 operating days, before solver and integration costs.

The twist closes depot B and raises zone 4 to 300 pallets. Demand is 1,160 against 1,000 pallets of capacity, so at least
160 pallets must be short. The pallet LP sends the whole shortfall to Zone 5, the zone most expensive to serve without B,
and costs 22,950 dollars including the penalty. The truck MILP costs 29,020 dollars for the day: 13,920 in pallets, 5,500
in 25 trucks and 9,600 in penalty. Because trucks are whole, it spreads the 160 short pallets across all six zones,
with 50 on Zone 5 and 10 to 30 on each of the others. That is 8,090 dollars more than the base day: the 9,600-dollar penalty, less 1,510 saved on pallets and trucks. Shadow prices
explain the shift: one extra pallet of capacity is now worth 39 dollars at A, 47 at B (reopening it displaces a 60-dollar
shortfall with a 13-dollar lane), 42 at C and 45 at D. On the base day those prices were 0 to 2. Zone marginal values rise
from 11 to 16 dollars to 51 to 60 dollars a pallet.

Recommendation at 06:30: keep the loads already built at A, C and D where they fit, and re-solve the truck model with the new
capacity. Ask the business for zone priorities before the solver chooses the zone that goes short, and encode them as zone
penalties. Add a penalty for changed lanes, and check the change against the 10 percent rule before sending the plan to
dispatchers. Pilot the plan in shadow for four weeks at one depot pair, comparing cost, shortfall and dispatcher acceptance
with the current plans, and switch from the dispatcher plan only after the result holds on real days.
