# Interview notes — Scaled dot-product attention

## Q1. Write scaled dot-product attention and state every shape.

**60-second answer.** Q is L by d_k, K is S by d_k and V is S by d_v. Scores are QK transposed, divided by the square root of d_k, giving an L by S matrix. A row-wise softmax over the keys gives the weights, each row summing to 1, and the output is the weights times V, which is L by d_v. A mask removes disallowed pairs before the softmax. I would state the policy for fully masked rows before writing code.

**Follow-ups.** Why are the scores divided by the square root of d_k? For independent unit-variance components, the dot product has variance d_k, so the standard deviation grows like the square root of d_k; dividing restores unit variance. What is the cost? The score matrix is L by S, so memory grows quadratically with sequence length.

## Q2. How do you make the softmax stable?

**60-second answer.** Subtract the row maximum before exponentiating. The softmax is invariant to a per-row shift, so the result is exact, and every exponent is at most zero. Masked entries are set to minus infinity. The one case where this fails is a row whose entries are all minus infinity: the maximum is minus infinity and the subtraction gives NaN. I would return zeros for that row and state it.

**Follow-ups.** Why not use a large negative constant such as minus one billion? When scores are large, a finite constant can be exceeded, so a masked key leaks. Infinity expresses exclusion exactly. Does the output change? For rows with at least one allowed key, no.

## Common mistakes

- Normalising over the query axis instead of the key axis, so columns sum to 1 instead of rows.
- Dividing by d_k instead of its square root, which under-scales the scores and flattens the softmax.
- Using a mask where True blocks attention, the inverse of the usual convention.
