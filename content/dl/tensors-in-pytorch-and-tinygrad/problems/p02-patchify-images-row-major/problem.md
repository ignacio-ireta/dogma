# Patchify images in row-major order

Implement `patchify(images, patch)`. Split each image into non-overlapping square patches of side
`patch` and flatten every patch into one feature vector. This is the first step of a vision
transformer: the patch vectors become tokens. Use tensor shape operations (see Constraints) rather than computing values.

## Contract

- `images`: tensor of shape `(B, C, H, W)`, any dtype. `B`, `C`, `H` and `W` are each zero or more.
- `patch`: a positive `int` (not a `bool`). `H` and `W` must both be divisible by `patch`.
- Returns: a tensor of the same dtype as `images`, shape `(B, N, C * patch * patch)`, with
  `N = (H // patch) * (W // patch)`.
  - Patch index `n = gi * (W // patch) + gj`, where `(gi, gj)` is the patch's grid row and column.
    Patches are numbered in row-major order over the grid.
  - Within a patch, features are channel-major, then row, then column:
    `out[b, n, c*p*p + r*p + s] = images[b, c, gi*p + r, gj*p + s]`, with `p = patch`.
- The result owns its memory. Writing into it must not change `images`, even when one patch covers
  the whole image.
- Values are copied exactly. Signed zeros, subnormal numbers and huge values are unchanged.
- Raises: `ValueError` when `images` is not 4-dimensional, when `patch` is not a positive integer,
  or when `H` or `W` is not divisible by `patch`.
- Must not mutate inputs.

## Examples

```python
images = torch.arange(16).reshape(1, 1, 4, 4)  # one channel, values 0..15 row by row
patchify(images, 2)
# -> tensor([[[ 0,  1,  4,  5],
#             [ 2,  3,  6,  7],
#             [ 8,  9, 12, 13],
#             [10, 11, 14, 15]]])
```

```python
# channel 0 is [[1, 2], [3, 4]]; channel 1 is [[5, 6], [7, 8]]; shape (1, 2, 2, 2)
images = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
patchify(images, 1)
# -> tensor([[[1, 5],
#             [2, 6],
#             [3, 7],
#             [4, 8]]])
```

## Constraints

- Allowed: tensor shape operations (`reshape`, `permute`, `view`, `clone`, `stack`, slicing) and
  Python loops over the patch grid.
- Not allowed: `torch.nn` (including `torch.nn.functional.unfold`) and any call named `unfold`
  (including `Tensor.unfold`). Those give the answer directly, and the point is to build it.

## Done looks like

Your function matches both examples, and the hidden tests check the patch order and the feature
order on random images against an independent reference. They also check one patch covering the
whole image, a single-pixel patch, non-square images, empty batches, dtype preservation, exact
values at extreme magnitudes, rejection of invalid arguments, that the result does not share memory
with the input, and that the input is unchanged.

The hidden tests draw their bulk inputs from the real dataset `digits` (see `docs/DATASETS.md`).
