# Incremental action-value estimates

Implement `action_value_estimates(actions, rewards, n_actions, step_size=None, initial_value=0.0)`.
Process a stream of pulls in order. Each pull names an action and gives its reward. Update only
the estimate of the action that was pulled, and return the final estimate for every action.

With `step_size=None` use the sample average. The update for a pull of action `a` with reward `r`
is `Q[a] <- Q[a] + (1/n)(r - Q[a])`, where `n` is the number of pulls of `a` so far, counting this
one. With a number `step_size = alpha` use the constant step size `Q[a] <- Q[a] + alpha(r - Q[a])`.
Every estimate starts at `initial_value`.

## Contract

- `actions`: one-dimensional integer NumPy array of length `N >= 0`. Entry `i` is an action index in
  `0 .. n_actions - 1`. Float, boolean and object arrays are invalid.
- `rewards`: one-dimensional real NumPy array of length `N`, every entry finite. Complex and boolean
  arrays are invalid. Integer arrays are valid.
- `n_actions`: integer `>= 1`. `bool` is invalid.
- `step_size`: `None`, or a real number `alpha` with `0 < alpha <= 1`. `bool`, NaN and other values
  are invalid.
- `initial_value`: finite real number, default `0.0`.
- Returns: a new float64 array of shape `(n_actions,)`. An action that is never pulled keeps
  `initial_value`.
- Raises `ValueError` when the shapes differ, an action index is out of range, the dtype is invalid,
  a reward is not finite, `n_actions` or `step_size` is invalid, or `initial_value` is not finite.
- Must not mutate the input arrays.

## Examples

```python
action_value_estimates(np.array([0, 1, 0]), np.array([1.0, 4.0, 3.0]), 2)
# -> array([2., 4.])          sample averages: (1 + 3) / 2 and 4 / 1
```

```python
action_value_estimates(np.array([0, 0]), np.array([1.0, 0.0]), 1, step_size=0.5)
# -> array([0.25])            Q: 0 -> 0.5 -> 0.5 + 0.5(0 - 0.5)
```

```python
action_value_estimates(np.array([0]), np.array([2.0]), 3, initial_value=5.0)
# -> array([2., 5., 5.])      action 0 is overwritten by its first reward; the others keep 5
```

## Constraints

Allowed: Python standard library and NumPy. Do not store the rewards of each action to compute a
mean at the end. Update the estimates as the stream is read. Not allowed: `scipy.stats`, which
computes the mean of a sample in one call.

## Done looks like

The examples pass, and empty streams and one-action problems are handled. Hidden tests compare the
sample average with the batch mean on random streams, check the constant-step result against its
closed form with exponential weights, check that the first sample-average step forgets the initial
value, check that an optimistic start decays toward the reward, cover large and small reward
magnitudes, invalid arguments and step sizes, dtype, and input mutation.
