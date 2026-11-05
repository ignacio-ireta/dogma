## Hint 1 — concept

Early stopping is a patience rule on the validation curve. Keep the best loss seen so far and a counter of
epochs since it last improved. The training run is judged by its best checkpoint, not by its final epoch.

## Hint 2 — formula / approach

Start with `best = +inf` and `wait = 0`. An epoch improves when `loss < best - min_delta`. On improvement,
set `best`, `best_epoch` and `wait = 0`. Otherwise add 1 to `wait`, and stop at the first epoch where
`wait == patience`. If the loop ends without stopping, the stop is the last epoch.

## Hint 3 — code shape

Validate the inputs first. Then loop over `enumerate(losses)`. Use an `if loss < best - min_delta` branch that
updates the three variables, and an `else` branch that increments `wait` and returns
`(best_epoch, epoch)` once `wait >= patience`. After the loop, return `(best_epoch, len(losses) - 1)`.
