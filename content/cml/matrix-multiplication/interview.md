# Interview notes — Matrix multiplication

## Q1. What does an m × n times n × p product cost, and why does order matter in a chain?

**60-second answer.** It costs $mnp$ multiply-adds, because each of the $mp$ output entries is a
dot product of length $n$. In a chain such as $ABC$, the product is the same whichever way you
bracket it, but the work is not. The bracketing sets the sizes of the intermediate results, so I
count the operations for each option and pick the cheapest. For shapes $(10, 100)$, $(100, 5)$
and $(5, 50)$, the best bracketing does 7,500 operations and the worst does 75,000.

**Follow-ups.**
- *Is there a faster algorithm?* Strassen's method is sub-cubic, but it is less numerically stable
  and pays off only at large sizes. Libraries use blocked cubic kernels, and the cubic count is
  the model I reason with.
- *Does the bracketing change the answer?* Not in exact arithmetic. In floating point, it changes
  the last few bits, so I compare with a tolerance.

## Q2. Walk through the shapes in scaled dot-product attention.

**60-second answer.** With queries $Q$, keys $K$ and values $V$, each of shape (batch, heads,
$T$, $d$), the scores are $QK^\top$, which is (batch, heads, $T$, $T$). A row-wise softmax keeps
that shape, and multiplying by $V$ gives (batch, heads, $T$, $d$) again. The batch and head axes
are kept, and each matrix product runs over the last two axes. The cost is $O(T^2 d)$, which is why
long sequences are expensive.

**Follow-ups.**
- *Why divide the scores by $\sqrt{d}$?* Dot products grow with $d$, which pushes the softmax into a
  region where its gradients are nearly zero.
- *Can you reorder the product?* Only without the softmax between $QK^\top$ and $V$. With it,
  the order is fixed.

## Q3. How would you check that a hand-written matrix product is correct?

**60-second answer.** Compare it with a trusted implementation on random inputs, using integers
where possible so the comparison is exact, and use a tolerance for floats. Then check the cases a
loop gets wrong: a non-square shape, a $1 \times 1$ case, and inputs where $AB \neq BA$, which
catches swapped operands. Add shape checks that raise on a mismatch, and make the error message
name both shapes.

**Follow-ups.**
- *Why not just test one example?* A single example can be symmetric enough to hide a swapped
  index. Random shapes expose it.
- *What does a property test add?* Identity on both sides, and $(AB)^\top = B^\top A^\top$, catch
  mistakes that a single fixed output would not.

## Common mistakes

- Using `*` where `@` was meant. The shapes broadcast, so nothing raises, and the answer is wrong.
- Swapping the operands. $BA$ may be defined when $AB$ is not, and it has a different shape.
- Counting the cost of a chain from its length, not from its shapes.
- Comparing float results with `==`.
