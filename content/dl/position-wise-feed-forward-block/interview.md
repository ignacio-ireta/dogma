# Interview notes — Position-wise feed-forward block

## Q1. What does the feed-forward block in a Transformer do, and where do its parameters go?

**60-second answer.** Attention mixes information across positions. The feed-forward block then processes each position on its own: expand the features from $d$ to $d_{ff}$, apply a nonlinearity, and project back to $d$. The block is the same for every position, so it adds no cross-position interaction. Its parameters are $2\,d\,d_{ff}$ plus biases. With the common $d_{ff}=4d$, that is $8d^2$, about two thirds of a standard block's weights. Its cost scales with the number of tokens times $d\,d_{ff}$.

**Follow-ups.**

- Why $4\times$? It is a convention from Vaswani et al. that balances capacity and cost. The ratio is a hyperparameter, not a law.
- GELU or ReLU? Modern models use a smooth activation. The exact GELU is $z\,\Phi(z)$, where $\Phi$ is the standard normal CDF, written with $\mathrm{erf}$.

## Q2. What is SwiGLU, and why does the hidden width shrink to two thirds?

**60-second answer.** SwiGLU replaces the single expansion by two, $W x$ and $V x$. The gate path $W x$ goes through SiLU, and the product $\mathrm{SiLU}(Wx)\odot Vx$ goes through a third matrix $W_2$. A gated block with three matrices has $3\,d\,d_{ff}$ weights. To keep the total equal to the plain block's $8d^2$, set $d_{ff}=\tfrac{8}{3}d$, which is about $\tfrac{2}{3}$ of $4d$.

**Follow-ups.**

- Does the gate help? Shazeer (2020) compares gated variants with the plain feed-forward block in Transformers, and the LLaMA report adopts SwiGLU, saying it improves performance. I have not checked a per-parameter comparison, so do not quote one.
- Where is the bias? The gated block has none in this topic's contract.

## Common mistakes

- Applying the activation to both paths in SwiGLU, or to the up path only.
- Using the tanh approximation of GELU when the reference uses the exact erf form.
- Reshaping the input so positions are mixed, for example flattening the sequence before the linear layers.
- Forgetting the 2/3 width change when switching from a plain to a gated block, which increases the parameter count by half.
