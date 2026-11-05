# Clipped GRPO token loss with a KL penalty and response masks

A GRPO step takes sampled responses for one or more prompts and turns them into one scalar loss.
Each response has one advantage, shared by all of its tokens. Each token has three log-probabilities:
under the current policy, under the policy that sampled it, and under the frozen reference.

Implement `grpo_token_loss(logp_new, logp_old, logp_ref, advantages, mask, *, clip_eps=0.2,
kl_coef=0.04)`. It returns the scalar loss to minimise. For token $t$ of response $i$, with
$\rho_{i,t} = \exp(\text{logp\_new} - \text{logp\_old})$ and $A_i$ the response's advantage:

$$s_{i,t} = \min\big(\rho_{i,t} A_i,\ \operatorname{clip}(\rho_{i,t}, 1 - \varepsilon, 1 + \varepsilon)\, A_i\big)$$

$$k_{3,i,t} = e^{\,\text{logp\_ref} - \text{logp\_new}} - (\text{logp\_ref} - \text{logp\_new}) - 1$$

Per-token loss is $-(s_{i,t} - \beta\, k_{3,i,t})$. The loss averages over the response tokens **of each
response first**, then over the $G$ responses:

$$\mathcal{L} = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{n_i} \sum_{t:\, m_{i,t}} -\big(s_{i,t} - \beta\, k_{3,i,t}\big),
\qquad n_i = \sum_t m_{i,t}$$

where $m_{i,t}$ is the mask, $\varepsilon$ is `clip_eps` and $\beta$ is `kl_coef`.

## Contract

- `logp_new`, `logp_old`, `logp_ref`: floating-point tensors, shape `(G, T)`, same dtype, all finite.
  `logp_new` is the current policy's log-probability of each sampled token; `logp_old` is the
  sampling policy's; `logp_ref` is the reference's.
- `advantages`: floating-point tensor, shape `(G,)`, same dtype, finite. Entry `i` applies to every
  token of response `i`.
- `mask`: boolean tensor, shape `(G, T)`. `True` marks a response token. Every row must contain at
  least one `True`. Positions marked `False` do not contribute, whatever their values.
- `clip_eps`: keyword-only float with `0 < clip_eps < 1`.
- `kl_coef`: keyword-only float, finite and `>= 0`.
- Returns a 0-dimensional tensor with the dtype of the inputs. It stays connected to autograd through
  `logp_new`. The function does not detach its arguments; the caller passes `logp_old` and
  `logp_ref` without gradient.
- Raises `ValueError` when: any argument is not a tensor; shapes disagree with `(G, T)` or `(G,)`;
  `G == 0` or `T == 0`; a dtype is not floating-point, or the three log-prob dtypes differ; the mask
  is not boolean; a row of the mask has no `True`; any log-prob or advantage is not finite;
  `clip_eps` is outside `(0, 1)`; `kl_coef` is negative or not finite.
- Must not mutate inputs.

## Examples

Tensors are shown as nested lists for brevity. Pass `torch.tensor(...)` values.

```python
# One response with one token, ratio 1, advantage 1, no KL: the loss is -1.
grpo_token_loss(
    logp_new=[[0.0]],
    logp_old=[[0.0]],
    logp_ref=[[0.0]],
    advantages=[1.0],
    mask=[[True]],
    kl_coef=0.0,
)
# -> -1.0
```

```python
# A positive advantage with ratio 1.5 is clipped at 1.2, so the surrogate is 1.2 and the loss is -1.2.
grpo_token_loss(
    logp_new=[[math.log(1.5)]],
    logp_old=[[0.0]],
    logp_ref=[[math.log(1.5)]],
    advantages=[1.0],
    mask=[[True]],
    clip_eps=0.2,
    kl_coef=0.0,
)
# -> -1.2
```

```python
# Two responses, one token each, advantage +1 and -1, ratios 1, no KL: surrogates are +1 and -1,
# so the loss is the mean of -1 and +1, which is 0.0.
grpo_token_loss(
    logp_new=[[0.0], [0.0]],
    logp_old=[[0.0], [0.0]],
    logp_ref=[[0.0], [0.0]],
    advantages=[1.0, -1.0],
    mask=[[True], [True]],
    kl_coef=0.0,
)
# -> 0.0
```

## Constraints

- Allowed: Python standard library and PyTorch tensor operations, such as `exp`, `clamp`, `minimum`,
  `where`, `sum`, `mean`.
- Not allowed: `torch.nn` (this covers `torch.nn.functional`, so `F.kl_div`, `F.mse_loss` and similar
  are banned too). Write each term from the definitions above.

## Done looks like

The three examples above pass. The hidden tests compare with an independent per-response loop
reference on seeded random ragged batches. They check the clipped branch for positive and negative
advantages, that a clipped token gets no policy gradient, that masked positions do not contribute even
with extreme values, the per-response averaging convention, that the KL term is zero when the policy
equals the reference, a float64 finite-difference gradient check away from clip boundaries, float32
dtype, and the validation and mutation rules.
