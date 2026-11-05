# Q-learning: one update and a gridworld run

Implement two functions. `q_learning_update` applies one tabular Q-learning update and returns a new
table. `train_q_learning` runs epsilon-greedy Q-learning on a fixed deterministic gridworld with a
fixed seed and returns the learned table.

### The gridworld

- States are numbered `0 .. 8` in row-major order on a 3 x 3 grid: `s = 3 * row + col`.
- Start every episode in state `0` (top-left). State `8` (bottom-right) is the goal and is terminal.
- Four actions: `0` up, `1` right, `2` down, `3` left. There are no interior walls. A move that
  would leave the grid leaves the agent in place. Moves are deterministic.
- Reward `r = 1.0` when the next state is the goal, otherwise `0.0`. Entering the goal ends the
  episode. An episode also ends after `max_steps` steps without reaching the goal.

## Contract

`q_learning_update(Q, s, a, r, s_next, done, alpha, gamma)`

- `Q`: 2-D finite real array of shape `(S, A)`. Not modified.
- `s`, `a`, `s_next`: integer indices in range for `Q`.
- `r`: finite real number. `done`: a `bool`, `True` when `s_next` is terminal. The function uses
  `done` as given and does not check it against `s_next`.
- `alpha` in `(0, 1]`, `gamma` in `[0, 1]`. `bool` and NaN are invalid for both.
- Target: `r` when `done`, otherwise `r + gamma * max over a' of Q[s_next, a']`. No bootstrap when
  `done`.
- Update: `Q[s, a] <- Q[s, a] + alpha * (target - Q[s, a])` on a copy. Only that entry changes.
- Returns a float64 table of the same shape. Raises `ValueError` on invalid arguments.

`train_q_learning(n_episodes, seed, alpha=0.5, gamma=0.9, epsilon=0.3, max_steps=100)`

- `n_episodes`: integer `>= 0`. `seed`: integer `>= 0`. `max_steps`: integer `>= 1`.
- `alpha`, `gamma`, `epsilon`: as above, with `epsilon` in `[0, 1]`.
- Use `rng = numpy.random.default_rng(seed)`, and table `Q = zeros((9, 4))` as float64.
- For each episode, set `s = 0`. For up to `max_steps` steps, choose an action and then:
  1. Draw `u = rng.random()` once. This happens on every choice.
  2. If `u < epsilon`, draw `a = rng.integers(4)` once and use it.
  3. Otherwise `a` is the lowest index among the largest `Q[s]`. Draw nothing further.
  Then move with the gridworld rule, call `q_learning_update` with `done` set when the goal is
  entered, set `s` to the next state, and stop the episode if it is done.
- Returns the final table, shape `(9, 4)`, float64.

## Examples

```python
Q = np.zeros((9, 4))
q_learning_update(Q, 0, 1, 0.0, 1, False, 0.5, 0.9)
# -> the table with Q[0, 1] = 0.0 (the next state's values are all zero)
```

```python
Q = np.zeros((9, 4))
Q[8, :] = 1e6  # the goal row has huge values; a terminal update ignores them
q_learning_update(Q, 7, 1, 1.0, 8, True, 0.5, 0.9)
# -> Q[7, 1] = 0.5 exactly, since the target is r alone
```

```python
train_q_learning(0, seed=0)
# -> zeros of shape (9, 4): no episodes, no updates
```

## Constraints

Allowed: Python standard library and NumPy. The draw order above is part of the contract, so do not
draw from any other source, and do not use the global `numpy.random` state. Not allowed: `torch`,
`scipy.stats` and `random`, which would break reproducibility.

## Done looks like

The examples pass. Hidden tests check one update against the formula on random values, that a
terminal update ignores the next state's values, that the input table is not modified, and that
only the updated entry changes. They compare the full run with an independent loop that uses the
same draw order, check the analytic optimum for this deterministic gridworld after a long run, check
that a one-step cap only touches the start state, check reproducibility from a fixed seed, and cover
invalid arguments.
