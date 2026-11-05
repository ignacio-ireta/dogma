---
title: "Learning-rate decay schedules"
minutes: 12
min_viable: "Read The system and the off-by-one paragraph in Where it breaks, then solve p01 using Hint 1 if needed."
---

## The system

A schedule is a function from a counter to a learning rate. Fix three things before you
write any formula. First, the counter $t$: the number of updates already taken, starting at
$t=0$. Second, its unit: optimizer steps (batches) or epochs. Third, the total length $T$, in
the same unit. The base rate $\eta_0>0$ is the rate at $t=0$. The floor $\eta_{\min}$ satisfies
$0\le\eta_{\min}\le\eta_0$. The rate after $t$ updates is $\eta_t$, a scalar.

| Schedule | Closed form, for $0\le t\le T$ | Shape |
|---|---|---|
| Step decay | $\eta_0\,\gamma^{\lfloor t/s\rfloor}$, with block length $s$ and $0<\gamma\le 1$ | staircase |
| Exponential | $\eta_0\,\gamma^{t}$ | geometric curve |
| Polynomial | $(\eta_0-\eta_{\text{end}})\,(1-t/T)^{p}+\eta_{\text{end}}$ | power curve; $p=1$ is linear |
| Cosine annealing | $\eta_{\min}+\tfrac12(\eta_0-\eta_{\min})\bigl(1+\cos(\pi t/T)\bigr)$ | smooth, one half-cosine |

$T$ is the number of updates from the base rate to the floor, so $\eta_T=\eta_{\min}$ for the
cosine. Frameworks call it `T_max`, and the floor is `eta_min`. Past $T$, the formula is not
defined by the schedule itself: `torch.optim.lr_scheduler.CosineAnnealingLR` keeps oscillating,
and the contract in this topic's problem holds the floor instead.

## Worked example

Take $\eta_0=0.1$, $T=4$, $\eta_{\min}=0$. The cosine phase $(1+\cos(\pi t/4))/2$ equals
$1$, $0.854$, $0.5$, $0.146$, $0$ at $t=0,\dots,4$. The rates are $0.1$, $0.0854$, $0.05$,
$0.0146$, $0$. Next, step decay with $s=2$ and $\gamma=0.5$. The rates at $t=0,\dots,4$ are
$0.1$, $0.1$, $0.05$, $0.05$, $0.025$, because $\lfloor t/2\rfloor$ is $0,0,1,1,2$.

## Why it works

The cosine derivative with respect to $t$ is $-\tfrac12(\eta_0-\eta_{\min})(\pi/T)\sin(\pi t/T)$.
It is zero at both ends and largest in the middle, so the rate lingers near its high and low
values and changes fastest mid-run. Loshchilov and Hutter use this form inside each restart
cycle, with $T_i$ as the cycle length [corpus:library-papers/1608.03983.pdf.md p.4]. On a small
ResNet, the AdamW paper found cosine annealing outperformed the step-drop and other schedules it
tried, and adopted it for its later runs [corpus:library-papers/1711.05101.pdf.md p.6]. Treat
that as evidence from one setting, not a law.

## Signal vs. noise

- **Signal:** the unit of $t$, the value of $T$ in that unit, the floor, and the counting
  convention below. These decide where the rate sits at the last update.
- **Signal:** cosine has three values to set, $\eta_0$, $T$ and the floor $\eta_{\min}$. Step decay
  needs drop times too, so it
  has more choices to get wrong.
- **Noise for now:** the exact decay constant of each recipe, and the names some frameworks give
  to exponential and polynomial decay. Learn the closed forms and move on.

## Where it breaks

- **Unit mismatch.** Setting `T_max` to the number of epochs and calling `scheduler.step()` after
  every batch ends the cosine after `T_max` batches, not epochs. Convert first:
  $T_{\text{updates}} = T_{\text{epochs}} \times \text{steps per epoch}$.
- **Off-by-one.** A new scheduler reports $\eta_0$ before any call, at $t=0$. Each
  `scheduler.step()` moves $t$ forward by one. In a loop of `optimizer.step()` then
  `scheduler.step()`, the $k$-th update (counting from 0) uses $\eta_k$. A run of $T$ updates
  therefore uses $\eta_0$ to $\eta_{T-1}$. The floor $\eta_T$ is set after the last update and is
  never used. Writing "step $k$" for "after $k$ calls" is the usual slip.
- **Running past $T$.** Torch's cosine rises again after $T$. A longer run than planned gives a
  rising rate with no error message. Check total updates against $T$.
- **Step decay that never fires.** If $s$ is larger than the run, the rate never drops. Check
  that a drop occurs inside the run you plan.
- **Resuming from a checkpoint.** If the counter is not restored, the schedule restarts at
  $\eta_0$.

## Interview angle

Start with the unit and the total length, then write the closed form. Cosine goes from $\eta_0$
at $t=0$ to $\eta_{\min}$ at $t=T$. For a fixed budget I default to it because it needs no
drop times. Then I check the off-by-one: which update uses which rate. Full answers and follow-ups
are in `interview.md`.

## Connections

This topic builds on gradient descent [cml/gradient-descent]: the step size is now a scalar
function of the update counter. The next topic, learning-rate warmup, puts a linear ramp in front
of this cosine. The AdamW paper multiplies both the step and the decoupled weight-decay term by one
schedule multiplier $\eta_t$, so the schedule in this topic can control weight decay too
[corpus:library-papers/1711.05101.pdf.md p.3].
