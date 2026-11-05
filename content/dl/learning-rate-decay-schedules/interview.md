# Interview notes — Learning-rate decay schedules

## Q1. Which learning-rate schedule would you use for a fixed training budget?
**60-second answer.** I first fix the counter: updates or epochs, and the total length in that
unit. For a fixed budget I default to cosine annealing from the base rate to a small floor. It
has two knobs, the base rate and the period, and it spends time near both extremes. Step decay
needs drop times as well. Before I trust the run, I check where the rate sits at the last update
and whether the run stays inside the period.
**Follow-ups.** What happens after T? A torch cosine rises again, so the run must not exceed T.
Why not exponential decay? It never reaches a floor, so the final rate depends on the run length
in a way you have to tune.

## Q2. A scheduler's values look one step off from what you expected. What do you check?
**60-second answer.** I check the counting convention. The scheduler reports the base rate before
any call. Each `scheduler.step()` advances it by one. The k-th optimizer update, counting from
zero, uses the rate for step k. I then check the unit: per-batch stepping needs T in batches,
not epochs.
**Follow-ups.** How do you test this? Replay the scheduler on a dummy parameter and compare
each rate with the closed form at every step. How do you resume? Restore the counter as well as
the optimizer state.

## Common mistakes
- Setting T in epochs while stepping per batch.
- Treating the rate after k calls as the rate for step k - 1, or the reverse.
- Forgetting that the cosine oscillates past T in some frameworks.
- Restarting a schedule from the base rate after loading a checkpoint.
