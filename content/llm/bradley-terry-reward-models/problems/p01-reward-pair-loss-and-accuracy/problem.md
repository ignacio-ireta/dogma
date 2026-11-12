# Bradley-Terry pair loss and pair accuracy

A reward model scores each response with one real number. Training compares a chosen and a rejected
response to the same prompt. For pair $i$ with chosen reward $r_{c,i}$ and rejected reward $r_{r,i}$,
the Bradley-Terry model says the chosen response wins with probability $\sigma(d_i)$, where
$d_i = r_{c,i} - r_{r,i}$ and $\sigma(z) = 1 / (1 + e^{-z})$.

Implement `bradley_terry_loss_and_accuracy(reward_chosen, reward_rejected)`. It returns the mean
negative log-likelihood of the chosen wins, and the fraction of pairs the model ranks correctly.
The loss must stay finite and accurate for very large margins in either direction.

## Contract

- `reward_chosen`: floating-point tensor, shape `(B,)`, `B >= 1`. Element `i` is the reward of the
  chosen response in pair `i`.
- `reward_rejected`: floating-point tensor with the same shape and dtype as `reward_chosen`. Element
  `i` is the reward of the rejected response in pair `i`.
- Returns a tuple `(loss, accuracy)`:
  - `loss`: 0-dimensional tensor with the dtype of the inputs. It is the mean over pairs of
    $-\log \sigma(d_i)$.
  - `accuracy`: 0-dimensional tensor with the dtype of the inputs. It is the fraction of pairs with
    $r_{c,i} > r_{r,i}$ **strictly**. A tie counts as incorrect.
- `loss` must stay connected to autograd: gradients flow into both reward tensors.
- The gradient of `loss` is the derivative of the mean of $-\log \sigma(d_i)$ at every margin, a tie
  included: the chosen reward of pair `i` receives $-(1 - \sigma(d_i)) / B$ and the rejected reward
  the opposite. At a tie ($d_i = 0$) that is $-1/(2B)$, not 0.
- Raises `ValueError` when: an argument is not a `torch.Tensor`; either argument is not 1-D; the
  shapes differ; `B == 0`; the dtype is not floating-point; the two dtypes differ; any value is not
  finite (NaN or inf).
- Must not mutate inputs.

## Examples

Tensors are shown as nested lists for brevity. Pass `torch.tensor(...)` values, for example
`torch.tensor([2.0, 0.0])`.

```python
reward_chosen = [2.0, 0.0]
reward_rejected = [0.0, 1.0]
bradley_terry_loss_and_accuracy(reward_chosen, reward_rejected)
# margins d = [2, -1]
# loss ~= 0.7200948, accuracy = 0.5
```

```python
reward_chosen = [1.0, 1.0]
reward_rejected = [1.0, 1.0]
bradley_terry_loss_and_accuracy(reward_chosen, reward_rejected)
# every margin is 0, so every loss is log 2
# loss ~= 0.6931472, accuracy = 0.0   (ties count as incorrect)
```

```python
reward_chosen = [0.0]
reward_rejected = [1000.0]
bradley_terry_loss_and_accuracy(reward_chosen, reward_rejected)
# d = -1000, so the loss is 1000 exactly in float64 (the naive log of sigmoid returns -inf)
# loss = 1000.0, accuracy = 0.0
```

## Constraints

- Allowed: Python standard library and PyTorch tensor operations, such as `abs`, `clamp`, `exp`,
  `log1p`, `mean`, `isfinite`.
- Not allowed: `torch.nn` (this covers `torch.nn.functional`, so `F.logsigmoid` and `F.softplus` are
  banned too), `torch.binary_cross_entropy_with_logits` (a loss routine that computes this loss as a
  whole), and any other routine that computes the loss as a whole. Write the stable form yourself
  from the definition.
- A stable form has a positive part, $\max(x, 0)$. `clamp(x, min=0)` and `where(x > 0, x, 0)` have a
  kink at 0, and autograd gives a tied pair no gradient through them. `(x + abs(x)) / 2` and
  `maximum(x, zeros_like(x))` have the same values and slope 1/2 at 0, which is the slope the loss
  has there.

## Done looks like

The three examples above pass, including the tie and the large-margin cases. The hidden tests
compare the loss with a PyTorch log-sigmoid reference on seeded random pairs, check that adding one
constant to both rewards changes nothing, check margins of plus and minus 1000 stay finite and exact
in float64, and check that float32 keeps its dtype. They also check the gradient against a float64
finite-difference estimate, that the rejected reward receives the opposite gradient sign, that a
tied pair receives gradient $-1/(2B)$, that ties are counted as wrong, and the shape, empty, NaN and
mutation rules.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).
