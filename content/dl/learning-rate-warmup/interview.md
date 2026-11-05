# Interview notes — Learning-rate warmup

## Q1. Why warm up the learning rate in a large model trained with Adam?
**60-second answer.** Adam divides the first moment by the square root of the second, and both are
bias-corrected averages. At the first update the normalized step is about the rate itself in each
coordinate, whatever the gradient size. Early second-moment estimates rest on very few gradients,
so the steps can be large and noisy. Warmup keeps the rate small until those estimates settle, which
is the common explanation rather than a proven result. Then the schedule decays to a floor, usually a
cosine. I check the value at the warmup boundary and at the end of the run.
**Follow-ups.** Why not just use a smaller rate throughout? That slows the whole run. Does warmup
always help? No. It is a practical guard, and I would check the first steps on the actual curve
before I tune the peak.

## Q2. What is the inverse-square-root schedule, and how does it differ from cosine with warmup?
**60-second answer.** It is the Vaswani et al. schedule: a linear ramp to a peak at the warmup length,
then a decay proportional to the inverse square root of the update number, scaled by the model width.
It has no floor and no fixed end. Cosine with warmup has a floor and a known total length, so I
can plan a budget with it.
**Follow-ups.** Why is the update number from 1? Because the decay term is undefined at 0. How do
you choose the peak? For the Vaswani schedule the width and the warmup length set it, so you do not
tune a separate base rate.

## Common mistakes
- Starting a cosine at the wrong step, so the peak is not exactly at the warmup boundary.
- Using a scheduler counter that starts at 0 where the paper numbers updates from 1.
- Forgetting that the total length must exceed the warmup length.
- Treating warmup as a fix for bad data rather than a guard for the optimizer's early state.
