# Group-normalised advantages with population standard deviation

GRPO samples a group of `G` responses for each prompt and scores them. It does not learn a value
function. Each response's advantage is its reward measured against its own group: subtract the
group mean and divide by the group's spread.

Implement `group_advantages(rewards, *, eps=1e-6)`. It returns, for each group, the normalised
advantage of every response.

For a group of rewards $r_1, \dots, r_G$, use the **population** standard deviation (divide the sum of
squared deviations by $G$, not $G - 1$):

$$\hat A_i = \frac{r_i - \bar r}{\sigma + \epsilon}, \qquad \bar r = \frac{1}{G}\sum_{j} r_j, \qquad
\sigma = \sqrt{\frac{1}{G}\sum_{j} (r_j - \bar r)^2}$$

## Contract

- `rewards`: floating-point tensor, shape `(P, G)`, with `P >= 1` prompts and `G >= 2` responses per
  prompt. Row `p` holds the rewards of the group for prompt `p`.
- `eps`: keyword-only float, finite and `> 0`. It is added to the standard deviation, outside the
  square root.
- Returns a tensor of shape `(P, G)` with the dtype of `rewards`. Entry `[p, i]` is $\hat A_i$ for
  group `p`. Groups are independent: each row depends only on its own rewards.
- An all-equal group has standard deviation zero. Its advantages are all zero, and they are finite
  (not NaN).
- Raises `ValueError` when: `rewards` is not a tensor; it is not 2-D; `P == 0`; `G < 2`; the dtype is
  not floating-point; any value is not finite; `eps` is not a finite positive number.
- Must not mutate the input.

## Examples

Tensors are shown as nested lists for brevity. Pass `torch.tensor(...)` values.

```python
group_advantages([[1.0, 0.0, 0.0, 1.0]])
# mean 0.5, population std 0.5  ->  about [0.999998, -0.999998, -0.999998, 0.999998]
```

```python
group_advantages([[1.0, 2.0, 3.0, 4.0], [0.7, 0.7, 0.7, 0.7]])
# row 0: mean 2.5, population std sqrt(1.25) ~= 1.118034
#        -> about [-1.341641, -0.447214, 0.447214, 1.341641]
# row 1: all equal, std 0 -> about [0.0, 0.0, 0.0, 0.0]; finite, not NaN
```

```python
group_advantages([[0.0, 1.0]])
# population std 0.5 -> about [-0.999998, 0.999998]
# a sample standard deviation would give about [-0.707106, 0.707106]: that is the mistake to avoid
```

## Constraints

- Allowed: Python standard library and PyTorch tensor operations, such as `mean`, `sum`, `sqrt`,
  `clamp`, `where`, `isfinite`.
- Not allowed: `torch.nn` (this covers `torch.nn.functional`), and the library standard-deviation and
  variance routines `torch.std`, `torch.std_mean`, `torch.var` and `torch.var_mean`, which would
  compute the convention for you. Write the mean and the spread from their definitions. Calling the
  `std` or `var` tensor methods is banned too, for the same reason.

## Done looks like

The three examples above pass. The hidden tests compare against a reference computed with a
population standard deviation on seeded random groups, check that the sample standard deviation is
not used, check that all-equal groups return finite zeros for several magnitudes, and that each
group's advantages sum to zero. They also check invariance to adding a constant to a group, groups
being independent, a large common offset, float32 dtype, a float64 finite-difference gradient check,
the invalid-input rules and no in-place mutation.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).
