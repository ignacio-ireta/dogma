# Interview notes — Speculative decoding speedup math

## Q1. Derive the expected tokens per round for speculative decoding.

**60-second answer.** A round drafts $\gamma$ tokens and the target verifies them in one pass. If each draft token is accepted independently with probability $\alpha$, the accepted prefix length $A$ satisfies $P(A\ge k)=\alpha^k$. The round emits $N=A+1$ tokens, because the target always supplies one more token, either the correction or the bonus. So $\mathbb E[N]=1+\sum_{k=1}^{\gamma}\alpha^k=(1-\alpha^{\gamma+1})/(1-\alpha)$. The round costs $\gamma c+1$, so the speedup is $\mathbb E[N]/(\gamma c+1)$.

**Follow-ups.** Why is the gain bounded? As $\gamma$ grows, $\mathbb E[N]$ approaches $1/(1-\alpha)$, while the cost keeps rising linearly. What if $\alpha$ depends on position? Replace the sum with a product of per-depth rates.

## Q2. When does speculative decoding make things slower?

**60-second answer.** When $\alpha$ is low and the draft is not cheap. At $\alpha=0$ the round emits one token and still pays $\gamma c+1$, so the speedup is $1/(\gamma c+1)$. A draft at 0.3 of the target's cost with acceptance 0.5 and $\gamma=8$ gives a speedup near 0.6; at 0.5 of the cost it drops to about 0.4. Verification can also cost more than one decode step when the batch is large, which pushes the same arithmetic against you.

**Follow-ups.** What measurement do you need first? The per-position acceptance rate on your own traffic, and the draft-to-target cost ratio at your batch size.

## Q3. Why does the output distribution not change?

**60-second answer.** Each drafted token is accepted with probability $\min(1,p/q)$. A rejected position is resampled from the residual $\operatorname{norm}(\max(0,p-q))$. Adding the two contributions per token gives $\min(p,q)+(p-q)^+=p$, so every emitted token has the target's marginal distribution. Speculation changes the cost of sampling, not the sample law.

**Follow-ups.** What if you skip the residual and just resample from $p$? Then tokens the draft already proposed get over-weighted and the output drifts from $p$. What about greedy decoding? It is a special case where the accepted token is the argmax, and the output matches plain greedy decoding.

## Common mistakes

- Counting only the accepted draft tokens and forgetting the target's own token. That undercounts $\mathbb E[N]$ by exactly 1.
- Writing the round cost as $\gamma c$ and dropping the verification pass.
- Dividing by $1-\alpha$ at $\alpha=1$ instead of using the $\gamma+1$ limit.
- Resampling from $p$ on rejection, which breaks exactness.
