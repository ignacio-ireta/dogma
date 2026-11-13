# Convolution forward pass (conv2d)

Implement `conv2d_forward(x, weight, bias=None, stride=1, padding=0)`, the forward pass of one 2-D
convolutional layer. In deep learning the operation called "convolution" is cross-correlation: the
kernel slides over the input without being flipped. Each output channel has its own filter that
sums over **all** input channels. The output is

$$\mathrm{out}[n,o,i,j]=b_o+\sum_{c,u,v} x_{\mathrm{pad}}[n,c,\,is+u,\,js+v]\;W[o,c,u,v],$$

where $s$ is the stride, $x_{\mathrm{pad}}$ is $x$ with $p$ zeros added on every side of the height and
width, $n$ indexes examples, $c$ input channels, $o$ output channels, and $(u,v)$ kernel offsets.

## Contract

- `x`: floating-point tensor of shape `(N, C_in, H, W)`, every axis at least 1. `N` is the batch.
- `weight`: tensor of shape `(C_out, C_in, kH, kW)` with `C_out, kH, kW >= 1`. Its dtype is `x`'s dtype.
  Its second axis must equal `C_in`.
- `bias`: `None`, or a tensor of shape `(C_out,)` with `x`'s dtype. `None` means no bias.
- `stride`: integer `s >= 1`, the same on both spatial axes. Booleans are invalid.
- `padding`: integer `p >= 0`, the same on both sides of both spatial axes. Booleans are invalid.
- Output size: $H_{\text{out}}=\left\lfloor\frac{H+2p-k_H}{s}\right\rfloor+1$, and the same formula for
  $W_{\text{out}}$ with $k_W$. The floor rounds down.
- Returns: tensor of shape `(N, C_out, H_out, W_out)` with `x`'s dtype.
- Raises `ValueError` when: `x` is not a floating 4-D tensor, or has a zero-size axis; `weight` is not a
  4-D tensor of `x`'s dtype, or its second axis is not `C_in`; `bias` has the wrong shape or dtype; `stride`
  is not an integer `>= 1`; `padding` is not an integer `>= 0`; or the kernel does not fit in the padded
  input, so that $H_{\text{out}}$ or $W_{\text{out}}$ would be below 1.
- Must not mutate `x`, `weight` or `bias`.
- Differentiable: the result stays on the autograd graph. When an input has
  `requires_grad=True`, backpropagating through the result gives the gradient of the formula
  above. Do not detach, and do not leave PyTorch (`.item()`, `.numpy()`, `torch.no_grad()`) on
  the way.

## Examples

```python
>>> x = torch.arange(9.0, dtype=torch.float64).reshape(1, 1, 3, 3)   # rows [0 1 2], [3 4 5], [6 7 8]
>>> conv2d_forward(x, torch.ones(1, 1, 2, 2, dtype=torch.float64))
tensor([[[[ 8., 12.],
          [20., 24.]]]], dtype=torch.float64)   # each entry sums one 2x2 window
```

```python
>>> x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float64)
>>> w = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float64)   # 1 at top-left only
>>> conv2d_forward(x, w)
tensor([[[[1.]]]], dtype=torch.float64)   # cross-correlation: the top-left value, not the flipped 4
```

```python
>>> x = torch.tensor([[[[5.0]]]], dtype=torch.float64)
>>> w = torch.tensor([[[[2.0]]]], dtype=torch.float64)
>>> conv2d_forward(x, w, padding=1)
tensor([[[[ 0.,  0.,  0.],
          [ 0., 10.,  0.],
          [ 0.,  0.,  0.]]]], dtype=torch.float64)   # H_out = floor((1 + 2 - 1) / 1) + 1 = 3
```

```python
# An input of width 6 with a kernel of width 3 and stride 2 has W_out = floor(3 / 2) + 1 = 2 columns,
# not 3. A partial window at the right edge is dropped.
```

## Constraints

- Allowed: PyTorch tensor operations such as `zeros`, slicing, `reshape`, `einsum` and `matmul`,
  plus the standard library (`math`).
- Not allowed: `torch.nn`, including `torch.nn.functional` and `torch.nn.Conv2d`, and `torch.conv2d`.
  They are the test oracles, not the tool. Write the zero padding, the window positions, the
  channel sum and the bias yourself.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They check: agreement with
`torch.nn.functional.conv2d` on seeded random configurations (several strides, paddings, kernel sizes,
rectangular inputs and multiple channels); a kernel that is not symmetric, so a flipped kernel
(true convolution) is detected; the output size where floor and ceil disagree; channel mixing, so
every output channel sums all input channels with its own filter; the bias added once per output
channel; the batch examples computed independently; a float32 dtype check; the input and kernel left
unchanged; and every `ValueError` rule. When an output matches a known wrong variant, the failure names
that mistake.

The hidden tests also backpropagate through the result and compare the gradients with PyTorch's
own.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer` and `digits` (see
`docs/DATASETS.md`).
