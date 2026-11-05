---
title: "MoE top-k routing"
minutes: 12
min_viable: "Read The system and the Worked example, then do p01 with the worked numbers beside you."
---

## The system

A mixture-of-experts (MoE) layer replaces one feed-forward block with $E$ expert blocks and a
router. Let $x\in\mathbb{R}^{T\times d}$ hold $T$ tokens of width $d$. The router is a linear map
$W_r\in\mathbb{R}^{d\times E}$, giving logits $z=xW_r\in\mathbb{R}^{T\times E}$. For each token $t$:

1. Softmax over experts: $p_{t,i}=\mathrm{softmax}(z_t)_i$.
2. Select the top $k$ experts by logit. Ties go to the lower index. Call the chosen set $S_t$.
3. Renormalise the gates over the chosen set: $g_{t,i}=p_{t,i}/\sum_{j\in S_t}p_{t,j}$ for $i\in S_t$. The gates sum to 1, and this equals a softmax over the $k$ chosen logits. ST-MoE's top-2 algorithm normalises its two gates by their sum in the same way [corpus:library-papers/2202.08906.pdf.md p.36]. This is the convention for p01.
4. The output is $y_t=\sum_{i\in S_t} g_{t,i}\,\mathrm{FFN}_i(x_t)$.

**Gate with $k=1$.** ST-MoE's general formula uses the raw router probability as the gate, with no renormalisation [corpus:library-papers/2202.08906.pdf.md p.4]. Switch Transformer uses $k=1$ and keeps the same raw gate, which it says is what makes the router differentiable [corpus:library-refresh/02-AI-ML/09-Mixture-of-Experts/2021 -- Fedus, Zoph and Shazeer -- Switch Transformers - Scaling to Trillion Parameter Models with Simple and Efficient Sparsity -- arXiv.pdf.md p.6]. With one chosen expert, renormalising would set every gate to 1, so the gate would carry no router gradient.

**Capacity.** ST-MoE defines the expert capacity as $\mathrm{CF}\cdot\text{tokens}/\text{experts}$, where tokens is the group size [corpus:library-papers/2202.08906.pdf.md p.4]. Each token takes $k$ slots, so this lesson's convention is $C=\lceil \mathrm{CF}\cdot kT/E\rceil$. A slot that arrives after an expert is full is **dropped**. Capacity bounds the work per expert, so every device has a fixed load. ST-MoE says the overflow passes to the next layer through a residual connection [corpus:library-papers/2202.08906.pdf.md p.4].

**Load-balancing loss.** Let $f_i$ be the fraction of tokens whose top-1 (argmax) expert is $i$, and $P_i$ the mean router probability for $i$ over all $E$ experts. The loss is $\alpha\,E\sum_i f_i P_i$ [corpus:library-papers/2202.08906.pdf.md p.31]. This form is Eq. 4 of Fedus et al. (2021), hence "Switch-style" [corpus:library-refresh/02-AI-ML/09-Mixture-of-Experts/2021 -- Fedus, Zoph and Shazeer -- Switch Transformers - Scaling to Trillion Parameter Models with Simple and Efficient Sparsity -- arXiv.pdf.md p.7]. That paper says it simplifies the separate load-balancing and importance losses of Shazeer et al. (2017), which penalise a squared coefficient of variation [corpus:library-papers/Shazeer et al. - Sparsely-Gated Mixture-of-Experts 2017.pdf.md p.5]. ST-MoE repeats the Switch form but credits Shazeer et al., so do not quote that attribution. $f$ is a count with no gradient, so the gradient reaches the router through $P$. ST-MoE notes that the standard top-$n$ formulation balances only over the top expert [corpus:library-papers/2202.08906.pdf.md p.38], so $f$ uses top-1 counts even when $k>1$.

**Parameters.** With $N_s$ shared parameters and $N_e$ per expert, the **total** count is $N_s+E\,N_e$ and the **active** count per token is $N_s+k\,N_e$. Total parameters grow with $E$. Per-token compute grows with $k$, not with $E$.

## Worked example

Take $T=2$, $E=3$, $k=2$, with router logits $z_0=[2,1,0]$ and $z_1=[0,0,3]$.

- Token 0. $p_0=[0.665,\ 0.245,\ 0.090]$. Top-2 are experts 0 and 1. Gates $=[0.665,0.245]/0.910=[0.731,\ 0.269]$, the same as $\mathrm{softmax}([2,1])$.
- Token 1. $p_1=[0.045,\ 0.045,\ 0.909]$. Top-1 is expert 2. The tie between experts 0 and 1 goes to 0, so $S_1=\{2,0\}$. Gates $=\mathrm{softmax}([3,0])=[0.953,\ 0.047]$ for experts 2 and 0.
- Counts. Expert 0 gets 2 tokens, expert 1 gets 1, expert 2 gets 1. The total is $T\cdot k=4$.
- Loss with $\alpha=1$. Top-1 choices are experts 0 and 2, so $f=[0.5,\ 0,\ 0.5]$. The mean probabilities are $P=[0.355,\ 0.145,\ 0.500]$. The loss is $3(0.5\cdot0.355+0+0.5\cdot0.500)\approx1.282$ (the unrounded value is 1.28249). A perfectly balanced router gives $3\sum_i (1/3)(1/3)=1$.

## Why it works

Sparse routing separates capacity from compute. Each token touches $k$ experts, and the model holds $E$ times the expert parameters. Shazeer et al. showed that sparse expert layers can add capacity without a matching rise in per-token cost [library:AI-ML/12-Frontiers/Shazeer Sparsely-Gated Mixture-of-Experts 2017.md]. ST-MoE says the auxiliary loss is minimised under a uniform distribution, so its value is $\alpha$ when $f$ and $P$ are both uniform [corpus:library-papers/2202.08906.pdf.md p.31]. It also describes the loss as encouraging even spread across experts, which helps hardware efficiency [corpus:library-papers/2202.08906.pdf.md p.5]. That imbalance leaves capacity unused is our reasoning, not a corpus claim.

## Signal vs. noise

Signal: the routing steps, the gate renormalisation, the capacity rule with its dropped tokens, the two vectors $f$ and $P$, and the total-versus-active parameter count. Noise: the exact values of $\alpha$ and CF, which vary between papers, and router variants such as noisy gating or z-losses.

## Where it breaks

- **Ties.** Equal logits make routing depend on an arbitrary order. Name the tie rule and test it. In p02 a tie moves the loss from about 1.0 to about 1.5.
- **Probability ranking.** Ranking by probability fails where probabilities underflow to zero. Rank by logit.
- **Collapse.** Without a balancing term, nothing pushes dispatch toward even, so a few experts can take most tokens. This is our reading of the mechanism, not a measured result. Balance is a trade-off too. ST-MoE uses $\alpha=10^{-2}$, large enough to balance load and small enough not to overwhelm the cross-entropy objective [corpus:library-papers/2202.08906.pdf.md p.31].
- **Low capacity.** A small CF drops tokens under load imbalance. Dropped tokens look like silent loss of data. Check the fraction of dropped slots.
- **Counting the wrong thing.** Counting only the top-1 slot understates the load of a top-2 layer. The counts must sum to $T\cdot k$.

## Interview angle

Start with the rule: softmax, top-$k$, renormalised gates, and the weighted sum. Then capacity and dropped tokens, then the auxiliary loss and why $f$ carries no gradient. Full versions are in `interview.md`.

## Connections

Prerequisites: [softmax](../../cml/softmax/) for the router, and the [position-wise feed-forward block](../../dl/position-wise-feed-forward-block/) for the experts. The lineage runs from Shazeer et al. (2017) through Switch Transformer (2021) to ST-MoE (2022), which studies stability and transfer.
