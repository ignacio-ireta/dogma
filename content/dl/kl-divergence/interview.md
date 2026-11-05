# Interview notes — KL divergence

## Q1. What is KL divergence, and why is it not a distance?

**60-second answer.** KL divergence is the expected log-likelihood ratio: $\mathrm{KL}(P\|Q)=\sum_j p_j\log(p_j/q_j)$. It is the average extra code length, in nats, when data from $P$ are encoded with a code built for $Q$. It is non-negative and zero only when $P=Q$. It is not a distance, because it is asymmetric, and it is infinite when $Q$ assigns zero probability to something $P$ can produce. I state both zero rules explicitly, since the infinite case is where most bugs surface.

**Follow-ups.** What is the relation to cross-entropy? $\mathrm{KL}(P\|Q)=H(P,Q)-H(P)$, so minimising cross-entropy against one-hot labels is minimising the divergence to the data distribution. Is the Jensen-Shannon divergence better? It is symmetric and bounded, which suits GAN analysis, but it is not the regulariser in a VAE.

## Q2. Forward versus reverse KL. Which one does a VAE use, and what does each direction do?

**60-second answer.** Forward $\mathrm{KL}(P\|Q)$ weights errors by $p$, so $Q$ must cover every region where $P$ has mass. It is mean-seeking, and it spreads $Q$ across modes. Reverse $\mathrm{KL}(Q\|P)$ weights errors by $q$, so $Q$ can sit on one mode. It is mode-seeking, and it can underestimate the spread. A VAE's regulariser is $\mathrm{KL}(q(z\mid x)\|p(z))$, with the approximate posterior first. That is the direction variational inference minimises. For diagonal Gaussians it has a closed form.

**Follow-ups.** Why does maximum likelihood use forward KL? It minimises $\mathrm{KL}(\text{data}\|\text{model})$. Does the closed form need diagonal covariance? No. The full-Gaussian formula also exists, with a log-determinant and a matrix inverse. The diagonal case is what p02 implements, because each dimension is independent and the cost is linear.

## Q3. How would you compute a batch KL from logits without overflow?

**60-second answer.** I take log-probabilities by log-sum-exp, $\log p_j = z_j-\mathrm{LSE}(z)$. I never form probabilities and then take logs, because a probability can underflow to exactly zero while its logit is finite. Masked classes have logit $-\infty$, and I drop their terms explicitly rather than computing $0\cdot(-\infty)$. A positive probability in P where Q is zero gives $+\infty$, which I return as is.

**Follow-ups.** What about `torch.nn.functional.kl_div`? Its input is $\log Q$ and its default target is $P$, as probabilities. Its default reduction divides by the batch size and the class count, so it is too small by a factor of the class count. For a per-row KL averaged over the batch, use `reduction="batchmean"`. How do you test it? Compare against `scipy.special.rel_entr` on random batches, and check the zero rules separately.

## Common mistakes

- Passing probabilities where `kl_div` expects log-probabilities, or the reverse.
- Reporting bits when the calculation used natural logs, or the other way round.
- Returning cross-entropy and calling it KL, which is off by $H(P)$.
- Averaging over classes instead of summing them.
