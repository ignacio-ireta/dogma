# Interview notes: Stochastic and mini-batch gradient descent

## Q1. Why mini-batch gradient descent rather than full-batch?

**60-second answer.** A full-batch gradient costs $O(nd)$ per update. A mini-batch of size $B$
costs $O(Bd)$, so you get about $n/B$ updates for the same compute per epoch. The batch gradient is
an unbiased estimate of the full gradient, with variance that falls like $1/B$, so small batches add
noise while large batches give a steadier direction. The trade-off is therefore updates versus noise,
and the batch size is a tuning knob for hardware throughput as well. Vectorised hardware is
efficient on batches, so $B$ is also limited by memory and latency, not only by statistics.

**Follow-ups.**
- *Does the noise always hurt?* This analysis shows what noise costs, not a general benefit. For large
  datasets, cheap early steps often matter more than final precision (per the Library note, §4.4).
- *Cost of an epoch?* About $O(nd)$ whatever $B$ is. The batch size changes how many updates that
  work is split into.

## Q2. How do you choose the batch size and the learning rate together?

**60-second answer.** Pick the batch size from hardware and noise needs, then tune the learning
rate. The stability bound must hold for the sharpest batch, not only for the full data, because a
small batch can have a larger local curvature. When $B$ grows and the noise drops, a larger
learning rate often works, which is the linear scaling heuristic from large-batch training. I would
check the loss curve for divergence first, then use a schedule or early stopping on validation loss.

**Follow-ups.**
- *What is the linear scaling rule?* Scale the learning rate with the batch size, after a warm-up,
  as in the large-batch ImageNet work. It is a heuristic, and it can fail at very large batches.
- *How would you detect that the learning rate is too large?* Non-finite or oscillating loss, or a
  weight norm that keeps growing.
- *Why does a constant step stop improving?* The noise term of the bound, $L\alpha\sigma^2/b$, does not shrink
  with the number of steps, so it stays while the step is constant. Decay the step, as in the Robbins–Monro conditions.

## Q3. Why shuffle the data, and what goes wrong without it?

**60-second answer.** Batches are only a fair sample of the data if the rows are mixed. Sorted or
grouped data makes each batch come from one group, so the iterate drifts toward the last group
visited and the end-of-epoch weights depend on the order. Reshuffling each epoch removes that
bias, and it gives each epoch a different set of batches. The shuffle must use a seeded generator,
so the run can be reproduced.

**Follow-ups.**
- *Shuffle once or every epoch?* Every epoch. A single shuffle repeats the same batch sequence each
  epoch, which keeps any bias from the order.
- *Does shuffling matter for full-batch GD?* No. The gradient does not depend on the row order.

## Common mistakes

- Dividing the batch sum by `batch_size` rather than the batch's own length, which is wrong for the last slice.
- Dropping the last smaller batch, so some rows are never used.
- Rebuilding the random generator every epoch, which repeats the same shuffle.
- Shuffling the data in place, which changes the caller's arrays.
- Choosing the learning rate from the full-data curvature alone, and missing divergence on a sharp batch.
