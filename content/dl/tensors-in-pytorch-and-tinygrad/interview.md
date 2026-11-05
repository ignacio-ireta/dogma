# Interview notes: Tensors in PyTorch and tinygrad

## Q1. Is this tensor operation a view or a copy, and how do you know?

**60-second answer.** Every tensor is a shape, strides and an offset into a shared storage buffer. Shape and stride operations (`view`, `permute`, `t()`, basic slicing, `expand`) return views that alias the original. Operations that need a different flat order or an index tensor (`contiguous`, a `reshape` on a non-contiguous tensor, advanced indexing) copy. I check with `data_ptr()` and the strides before I trust a write-through assumption, and I state the copy cost when it matters.

**Follow-ups.** How do you check that two tensors share memory? Compare `untyped_storage().data_ptr()`, or check that writes to one show in the other. Why is an expanded tensor free? Expansion sets a stride to 0, so one element is read for every position along that axis.

## Q2. What is the dtype of `a + b` when `a` is `float32` and `b` is `float64`, and why does it matter?

**60-second answer.** The result is `float64`. Both tensors have dimensions, so the promotion takes the higher dtype within the floating category. The cost is double the memory and bandwidth, and a precision assumption that may not hold downstream. A Python scalar or a zero-dimensional tensor of the same category does not promote, so `float32 * 2.0` stays `float32`.

**Follow-ups.** What does an integer tensor plus a Python float give? A floating tensor in the default dtype, usually `float32`. How would you keep a distance computation in `float32`? Cast both inputs to the same dtype first, and state that the output dtype is part of the contract.

## Common mistakes

- Assuming `reshape` never copies. It copies whenever the strides cannot describe the new shape.
- Writing into a result that aliases the input, for example a single-tile reshape of the whole image.
- Computing $\|x\|^2 - 2x\cdot y + \|y\|^2$ in float32 for points with a large common offset, then trusting tiny distances.
- Writing `reshape(-1, 0)` on an empty tensor and expecting a size. The unknown size is ambiguous, so give it explicitly.
