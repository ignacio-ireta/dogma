# Discounted returns for an episode

Implement `discounted_returns(rewards, gamma)`. Given the rewards of one finished episode,
return the discounted return from every step. Use one backward pass, so each return is built
from the one after it.

Let `rewards[t]` be the reward received after the action taken at step `t`, for `t = 0..T-1`.
The return from step `t` is `G_t = sum over k >= 0 of gamma**k * rewards[t + k]`, where the sum
stops at the end of the episode. This is the discounted sum of the rewards from step `t` onward.

## Contract

- `rewards`: one-dimensional NumPy array of length `T >= 0`, integer or floating dtype, every
  entry finite. Complex, boolean and object arrays are invalid.
- `gamma`: a real number in `[0, 1]`. `bool`, NaN and values outside the interval are invalid.
- Returns: a new float64 array of shape `(T,)`. `result[t]` is `G_t`. An empty episode returns
  an empty float64 array.
- Raises `ValueError` when `rewards` is not one-dimensional, has an invalid dtype, contains a
  non-finite entry, or when `gamma` is invalid.
- Must not mutate the input array.

## Examples

```python
discounted_returns(np.array([1.0, 0.0, 2.0]), 0.5)
# -> array([1.5, 1.0, 2.0])
```

```python
discounted_returns(np.array([1.0, 1.0, 1.0]), 1.0)
# -> array([3., 2., 1.])
```

```python
discounted_returns(np.array([]), 0.9)
# -> array([], dtype=float64)
```

## Constraints

Allowed: Python standard library and NumPy. Not allowed: `scipy.signal`, whose discounted
cumulative-sum filter returns the whole answer in one call. Write the backward accumulation
yourself.

## Done looks like

The examples pass, and the empty episode and the `gamma = 0` edge case are handled. Hidden tests
compare against an exact rational computation for a dyadic discount, against a direct forward
sum for random episodes, and check gamma = 1, gamma = 0, empty and tiny or huge rewards, integer
input, invalid rewards and gamma values, and input mutation.
