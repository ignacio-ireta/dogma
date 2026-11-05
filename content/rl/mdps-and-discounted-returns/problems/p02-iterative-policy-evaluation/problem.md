# Iterative policy evaluation for a tabular MDP

Implement `iterative_policy_evaluation(P, R, pi, gamma, tol=1e-10)`. Return the state values of
the fixed policy `pi` in a small finite MDP by repeated sweeps. Start from zero values and apply
the Bellman expectation update until the largest change in any state value is below `tol`.

Let `S` be the number of states and `A` the number of actions. The update is

`V_new[s] = sum over a of pi[s, a] * (R[s, a] + gamma * sum over s' of P[s, a, s'] * V[s'])`

The reference updates synchronously (each sweep uses the values from the previous sweep); an
in-place sweep converges to the same values and is also accepted. The stopping rule is exact: after each sweep,
compute `delta`, the largest absolute change `max_s |V_new[s] - V[s]|`, and stop at the first sweep
where `delta < tol` (strictly). Return the values from that sweep. No sweep cap is needed, because
the map contracts by `gamma < 1`. The returned values satisfy the Bellman expectation equation to
within the tolerance.

## Contract

- `P`: real array of shape `(S, A, S)`, non-negative, finite. For every `(s, a)`, the row
  `P[s, a, :]` sums to 1 within `1e-9`. `P[s, a, s']` is the probability of moving to `s'`.
- `R`: real array of shape `(S, A)`, finite. `R[s, a]` is the expected reward for taking `a` in `s`.
- `pi`: real array of shape `(S, A)`, non-negative, finite. For every state `s`, `pi[s, :]` sums
  to 1 within `1e-9`.
- `gamma`: real number with `0 <= gamma < 1`. `bool`, NaN and other values are invalid.
- `tol`: positive finite real number. `bool` is invalid.
- Returns: a new float64 array of shape `(S,)`.
- Raises `ValueError` when any shape, dtype, range, row sum, `gamma` or `tol` rule is broken.
- Must not mutate any input array.

## Examples

```python
P = np.array([[[1.0]]])
R = np.array([[2.0]])
pi = np.array([[1.0]])
iterative_policy_evaluation(P, R, pi, 0.5)
# -> array([4.])
```

```python
# State 0 moves to state 1 with reward 1; state 1 moves back to state 0 with reward 0.
P = np.array([[[0.0, 1.0]], [[1.0, 0.0]]])
R = np.array([[1.0], [0.0]])
pi = np.ones((2, 1))
iterative_policy_evaluation(P, R, pi, 0.5)
# -> array([1.3333333, 0.6666667])
```

## Constraints

Allowed: Python standard library and NumPy, including matrix products and `einsum`. Not allowed:
`numpy.linalg` and `scipy.linalg`. Do not solve the Bellman linear system directly. Iterate
until the stopping rule holds.

## Done looks like

The two examples pass, the zero-discount case returns the policy-averaged reward, and invalid
inputs raise `ValueError`. Hidden
tests compare against an independent linear-system solve on random MDPs with several discounts,
check the zero-discount and absorbing-state cases, a duplicated action with split probability,
large reward magnitudes, invalid probabilities, `gamma = 1`, invalid `tol`, shape mismatch, and
input mutation.
