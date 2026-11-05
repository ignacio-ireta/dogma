# Sarsa and expected Sarsa updates

Implement two one-step tabular TD control updates. `sarsa_update` bootstraps from the action the
agent takes next. `expected_sarsa_update` bootstraps from the average of the next state's action
values under an epsilon-greedy policy.

Both take the table `Q`, the transition, the step size `alpha`, and the discount `gamma`. Each
returns a new table. The input table is left unchanged.

## Contract

- `Q`: 2-D real array-like of shape `(S, A)` with `S >= 1` and `A >= 1`. Entries are finite.
  Integers are allowed.
- `s`, `a`: integer indices with `0 <= s < S` and `0 <= a < A`.
- `r`: finite real reward.
- `s_next`: integer state in `[0, S)`, or `None` when the next state is terminal.
- `a_next` (`sarsa_update` only): integer action in `[0, A)`. It is ignored when `s_next is None`,
  so it may be `None` then.
- `alpha`, `gamma`, `epsilon`: real numbers in `[0, 1]`.
- Target: `y = r` when `s_next is None`. Otherwise `y = r + gamma * Q[s_next, a_next]` for Sarsa,
  and `y = r + gamma * sum_b pi(b) * Q[s_next, b]` for expected Sarsa.
- Policy `pi` for expected Sarsa, with `k` the number of actions tied for the largest value of
  `Q[s_next, :]`: `pi(b) = epsilon / A + (1 - epsilon) / k` for each tied action, and
  `pi(b) = epsilon / A` for every other action. Ties are exact equality, and the exploit share
  `1 - epsilon` is split evenly among the tied actions.
- Update: `Q_new[s, a] = Q[s, a] + alpha * (y - Q[s, a])`. Every other entry is unchanged.
- Returns: new `float64` array of shape `(S, A)`.
- Raises `ValueError` when `Q` is not a non-empty 2-D real array, when an index is out of range or
  not an integer, when a reward or value is not finite, or when `alpha`, `gamma` or `epsilon` lies
  outside `[0, 1]`.
- Must not mutate inputs.

## Examples

```python
>>> Q = [[0.0, 0.0], [1.0, 3.0]]
>>> sarsa_update(Q, s=0, a=1, r=1.0, s_next=1, a_next=0, alpha=0.5, gamma=0.5)
array([[0.  , 0.75],
       [1.  , 3.  ]])
```

```python
>>> expected_sarsa_update(Q, s=0, a=1, r=1.0, s_next=1, alpha=0.5, gamma=0.5, epsilon=0.5)
array([[0.   , 1.125],
       [1.   , 3.   ]])
```

```python
>>> sarsa_update(Q, s=0, a=1, r=1.0, s_next=None, a_next=None, alpha=0.5, gamma=0.5)
array([[0. , 0.5],
       [1. , 3. ]])
```

## Constraints

- Allowed: NumPy array operations (`np.asarray`, `np.max`, comparisons, `np.full`). Compute the
  policy weights yourself.
- Not allowed: `random` and `scipy.stats`. These updates are deterministic, so no sampling is
  involved.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check hand-computed targets
on a two-state, three-action table, including a tie among the greedy actions. They confirm that
expected Sarsa reduces to the Q-learning max target at `epsilon = 0` and to the uniform mean at
`epsilon = 1`. They compare against an explicit enumeration of the policy on random tables with
ties, and check terminal next states, boundary values of `alpha` and `epsilon`, the convex
combination property of the update, the rejection of invalid indices and parameters, and inputs
that are left unchanged.
