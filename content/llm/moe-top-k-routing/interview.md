# Interview notes — MoE top-k routing

## Q1. How does a top-k mixture-of-experts layer route tokens?

**60-second answer.** The router is a linear map from the token's features to one logit per expert. A softmax over the experts gives probabilities. Each token keeps its $k$ highest-logit experts, with ties going to the lower index. The gates of the kept experts are renormalised so they sum to 1. The layer output is the gate-weighted sum of the chosen experts' outputs. Compute grows with $k$, while the parameter count grows with the number of experts $E$. That is the trade the design makes.

**Follow-ups.**

- What does $k$ change? Per-token FLOPs and activation memory scale with $k$. The number of stored expert weights scales with $E$ and does not depend on $k$.
- Why rank by logit and not by probability? Probabilities can underflow to equal zeros, and then the order is lost. The logits keep it exact.

## Q2. What is the load-balancing loss, and what goes wrong without it?

**60-second answer.** Without a balancing term, nothing pushes the router toward even dispatch, so a few experts can take most tokens and the rest sit under-used. (That mechanism is our reasoning; the corpus states the loss's purpose.) The standard form, as ST-MoE writes it, is $\alpha E\sum_i f_i P_i$. Here $f_i$ is the fraction of tokens whose top-1 expert is $i$ and $P_i$ is the mean router probability for $i$. The loss equals $\alpha$ when both vectors are uniform. $f$ is a count, so the gradient flows through $P$. The coefficient $\alpha$ is a trade-off. ST-MoE uses $10^{-2}$, large enough to balance load and small enough not to overwhelm the cross-entropy objective.

**Follow-ups.**

- Why is $f$ not differentiated? It is a count of discrete choices, so it has no gradient. The loss is still a useful signal, because the gradient through $P$ pushes probability away from overloaded experts.
- What is capacity for? It bounds the work per expert so that every device has a fixed load. Tokens over the limit are dropped for that expert.

## Common mistakes

- Renormalising over all experts instead of over the chosen ones.
- Counting only the top-1 slot in the expert load when $k>1$.
- Ranking by probability, so that underflowed values tie.
- Describing total parameters as if only the active ones were stored.
