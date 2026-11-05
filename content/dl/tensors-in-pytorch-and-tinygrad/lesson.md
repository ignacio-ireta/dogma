---
title: "Tensors in PyTorch and tinygrad"
minutes: 12
min_viable: "Read The system and Worked example, then solve p02 (patchify) with a hint if needed."
---

## The system

A tensor is a view onto a flat block of memory. Five fields describe it:

| Field | Meaning |
|---|---|
| storage | the flat buffer of numbers, shared by every view of it |
| shape $(n_0,\dots,n_{k-1})$ | the size along each axis |
| strides $(s_0,\dots,s_{k-1})$ | the step, in elements, to move one index along each axis |
| dtype | how each element is encoded (`float32`, `float64`, `int64`, ...) |
| device | where the storage lives (`cpu` here; the exercises run on CPU) |

Element $(i_0,\dots,i_{k-1})$ sits at storage position $o+\sum_j i_j s_j$, where $o$ is the view's offset.
Three rules follow.

1. **Shape and stride operations are views.** `view`, `reshape` (when possible), `permute`,
   `transpose`, `t()`, slicing with `:`, `expand` and `unsqueeze` change $(shape, strides, o)$
   and leave the storage alone.
2. **Some operations must copy.** `contiguous()` copies a non-contiguous tensor. `reshape` copies
   when the strides cannot describe the requested shape. Advanced indexing (an index tensor or a
   boolean mask) copies.
3. **Broadcasting stretches with stride 0.** Shapes align from the right. An axis of size 1, or a
   missing leading axis, is repeated by setting its stride to 0. No memory is allocated.

**tinygrad** (concept only; the exercises here use torch) is a lazy framework. It records operations
and computes values only when one is needed, for example at a `.numpy()` call, so it can fuse kernels
and skip intermediates. PyTorch in eager mode runs each operation as soon as it is called.

## Worked example

Let $x$ have shape $(2,3)$ and contents `[[0,1,2],[3,4,5]]`, dtype `float32`.

- Contiguous storage: `[0,1,2,3,4,5]`, shape $(2,3)$, strides $(3,1)$, offset $0$.
- Element $(1,2)$ sits at $0+1\cdot3+2\cdot1=5$, which holds `5`. Correct.
- $y = x^\top$ has shape $(3,2)$ and strides $(1,3)$. It shares storage, so `y.data_ptr()` equals
  `x.data_ptr()`. `y.is_contiguous()` is `False`.
- `y.reshape(-1)` must read storage in $y$'s order: $0,3,1,4,2,5$. The strides cannot describe that
  order, so `reshape` copies. Its data pointer differs from `x`.
- `x[:, ::2]` has shape $(2,2)$ and strides $(3,2)$. It is a view and still shares storage.

Broadcasting: shapes $(2,1,4)$ and $(3,1)$ align as $(2,1,4)$ and $(1,3,1)$, giving $(2,3,4)$.
The size-1 axes get stride 0.

Dtype promotion: `torch.tensor([1, 2]) + 1.5` gives `float32`, because the integer tensor meets a
float scalar and the default float dtype is used. `float32_tensor * 2.0` stays `float32`.
`float32_tensor + float64_tensor` gives `float64`, because both have dimensions.

## Why it works

With storage and strides you can predict aliasing, copies and broadcast cost. A `reshape` is a view
exactly when the new axes can be formed by merging neighbouring old axes whose strides line up. The
patch exercise uses this: a `(B, C, H, W)` image splits into `(B, C, H/p, p, W/p, p)` as a view,
`permute` reorders axes as a view, and the final flatten copies unless there is only one patch.

Promotion is a separate rule. Dtypes have categories, ordered bool < integer < floating < complex. A
tensor with dimensions sets the result dtype within its category. A zero-dimensional tensor or a
Python scalar of the same category does not raise it. That is why `float32 * 2.0` stays `float32`.

NumPy interop uses the same storage: `torch.from_numpy(a)` and `t.numpy()` share memory on CPU when
dtype and layout match. The mathematics is the multi-dimensional array of
[book: Goodfellow, Bengio and Courville, Deep Learning, ch. 2].

## Signal vs. noise

Signal: which operations alias storage, which force a copy, and what shape a broadcast gives.
Read shapes and strides before running code. Noise: tinygrad's API and graph internals, and device
placement details. Learn the lazy model as a concept only.

## Where it breaks

- `reshape(-1, 0)` on a zero-element tensor is ambiguous: the unknown size could be any value. When another
  dimension is zero, give the sizes explicitly.
- A view that looks like a copy can alias. In p02, when one patch covers the whole image, a plain
  `reshape` returns a view of the input. Writing into the result then changes the image.
- Equal sizes hide meaning. Two axes of length $B = N$ add without error even when one is the
  batch and the other is the point count. Name the axes and check shapes at the function boundary.
- Mixed dtypes promote silently. `float32` with `float64` returns `float64` and doubles the memory.
  In p01 the output dtype is part of the contract.
- The expansion $\|x\|^2-2x\cdot y+\|y\|^2$ cancels when points share a large offset. Use direct
  differences, or centre the data first, and clamp at zero.

## Interview angle

Start from storage: "a tensor is a shape, strides and an offset into a shared buffer." Then say what
each operation does to those three, and name the call that forces a copy (`contiguous`, or a `reshape`
that needs one). Follow-ups: "how do you check whether two tensors share memory?" (`untyped_storage`
and `data_ptr`), and "why is a broadcast cheap?" (stride 0 means no allocation).

## Connections

Requires `dl/building-tensors-by-broadcasting`, which covered broadcasting on NumPy arrays. This topic
moves those rules into torch and adds strides, views and dtype promotion. Next comes
`dl/gradients-with-framework-autograd`, which records operations on these tensors. Patchify is the input
step of a vision transformer, and `dl/image-patch-embedding` builds on it.
