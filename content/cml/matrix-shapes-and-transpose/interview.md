# Interview notes — Matrix shapes and transpose

## Q1. Walk me through the shapes in a linear layer, Y = XW + b.

**60-second answer.** $X$ is (batch, in), $W$ is (in, out), $b$ is (out,) and broadcasts across
the batch, and $Y$ is (batch, out). I write each shape down, then check that the inner dimension
matches (in = in) and that the output takes its shape from the outer dimensions. If the weights
are stored as (out, in), as PyTorch's `nn.Linear` stores them, the forward pass is
$XW^\top + b$. The shape bookkeeping is the whole job; the arithmetic is the easy part.

**Follow-ups.**
- *Is transposing expensive?* In NumPy, `.T` is an O(1) view. The cost is in memory access, not in
  arithmetic, so a transposed operand can be slower in a kernel that expects contiguous rows.
- *Why does a bias of shape (out,) work with a (batch, out) output?* Broadcasting aligns trailing
  dimensions and stretches the size-1 axis, so the bias is added to every row.

## Q2. Why is (AB)ᵀ equal to BᵀAᵀ, and not AᵀBᵀ?

**60-second answer.** Expanding one entry shows it. The summed index sits between the factors,
and transposing swaps the output indices but not the summed one, so the factor order must reverse
to keep that index inside. In practice AᵀBᵀ is often not even defined: with $A$ of shape (2, 3)
and $B$ of shape (3, 4), $A^\top$ is (3, 2) and $B^\top$ is (4, 3), so the inner dimensions are 2
and 4.

**Follow-ups.**
- *When does it matter in code?* Gradients of a linear layer. The gradient with respect to $W$ is
  $X^\top \delta$, and getting that transpose wrong gives a shape error or, worse, a silently
  wrong update when the shapes happen to match.
- *Does order of the transposes change the cost?* No. The cost depends on the product, not on
  the transposes, which are views.

## Q3. How do you debug a shape error in a model you did not write?

**60-second answer.** I reproduce it with the smallest input that runs: batch size 1 or 2, a
single example. Then I print the shape at each module boundary, and compare it with the shape I
expect from the layer definitions. The usual causes are a missing transpose, a flattened axis
that was not reshaped back, or a batch-first versus batch-last mix-up. A shape that broadcasts
without error is the dangerous case, so I also check the output shape of every elementwise
operation.

**Follow-ups.**
- *What if both shapes are square?* Then a wrong order can still run without error and give a
  wrong answer. Check the values on a tiny input with known answers.
- *Why not just let the framework check?* Frameworks check that a product is defined. They do
  not check that you meant the product you wrote.

## Common mistakes

- Transposing the wrong factor, or forgetting the transpose when the weights are stored as (out, in).
- Applying `.T` to a 1-D array and expecting a column.
- Using `reshape` where a transpose was meant. The shape matches; the data order does not.
- Concluding a shape is correct because nothing raised, when broadcasting silently stretched an axis.
