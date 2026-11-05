# Interview notes: Chain rule

## Q1. How does the chain rule give the gradient of a sigmoid unit with squared error?

**60-second answer.** Write the loss as a path: $\mathbf{w}\to\mathbf{z}=X\mathbf{w}\to\mathbf{s}=\sigma(\mathbf{z})\to(\mathbf{s}-\mathbf{y})^2$. Multiply the local derivatives along the path and sum over samples. The gradient is $\frac{2}{n}X^\top[(\mathbf{s}-\mathbf{y})\odot\mathbf{s}\odot(1-\mathbf{s})]$. The three factors are the outer square, the sigmoid's slope $s(1-s)$, and the linear map, whose adjoint $X^\top$ carries the result back onto the weights.

**Follow-ups.**
- *Why does $X^\top$ appear?* It is the adjoint of $X$, not the derivative. The derivative of $\mathbf{z}=X\mathbf{w}$ with respect to $\mathbf{w}$ is $X$. Reverse mode applies $X^\top$ to the upstream gradient, which returns each sample's gradient to the weights that produced its score.
- *What is the shape of each factor?* $(n,)$ for the residual, the slope and the product, then $(d,n)$ times $(n,)$ gives $(d,)$. A $(n,1)$ target broadcasts to $(n,n)$, which is the usual shape bug.
- *Why run the chain backwards for a scalar loss?* One backward pass gives the derivative with respect to every weight, at a small constant multiple of the forward cost. Forward mode computes $J\dot{\mathbf{x}}$, one directional derivative per pass, so it wins only when there are few input directions.

## Q2. Why does training stall with a sigmoid output and squared error, and what would you use instead?

**60-second answer.** The sigmoid slope $s(1-s)$ multiplies the gradient, and it goes to zero when the unit saturates. A confidently wrong unit therefore gets almost no gradient, and the loss looks flat. Cross-entropy cancels the slope: its gradient with respect to the logit is $\sigma(z)-y$. For classification, use cross-entropy with a sigmoid or softmax output. Keep squared error for regression, where the output is not squashed.

**Follow-ups.**
- *How do you compute the sigmoid safely?* From $e^{-|z|}$, so no exponent is large and positive. The derivative should be $e/(1+e)^2$, because $s(1-s)$ loses digits when $s$ rounds to 1.
- *Why not check the saturated gradients with finite differences?* At $z=30$ the true gradient is $1.9\times10^{-13}$, and central differences return exactly $0$. The two losses differ below float64 resolution, so compare against an analytic reference.

## Common mistakes

- Dropping the sigmoid derivative from the chain, so the gradient is only $\frac{2}{n}X^\top(\mathbf{s}-\mathbf{y})$. Each term is too large by a factor $1/(s(1-s))$, which is at least four because $s(1-s)\le 0.25$, and by much more when the unit saturates.
- Computing $e^{z}/(1+e^{z})$ for all $z$, which gives NaN at $z=710$.
- Using $s(1-s)$ at large $|z|$ and reporting a gradient that is wrong in relative terms.
- Passing a column vector as the target and getting an $(n,n)$ residual with no error.
