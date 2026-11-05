---
title: "KL divergence"
minutes: 12
min_viable: "Read The system and Worked example, then do p01 using Hint 1 if needed."
---

## The system

For two discrete distributions $P$ and $Q$ over the same $k$ classes, with probabilities $p_j$ and $q_j$, the Kullback-Leibler divergence is

$$\mathrm{KL}(P\,\|\,Q)=\sum_{j=1}^{k} p_j\,\log\frac{p_j}{q_j}.$$

It is measured in nats with the natural log, or in bits with $\log_2$. Four rules govern every computation:

| Rule | Statement |
|---|---|
| Zero in $P$ | A term with $p_j = 0$ contributes $0$, so $0\log 0 = 0$, even if $q_j = 0$ |
| Zero in $Q$ | If $q_j = 0$ while $p_j > 0$, the divergence is $+\infty$. Mass in $P$ that $Q$ forbids is unbounded |
| Non-negative | $\mathrm{KL}\ge 0$, with equality exactly when $P = Q$ |
| Asymmetric | $\mathrm{KL}(P\|Q)\ne\mathrm{KL}(Q\|P)$ in general, so it is not a distance |

The identity that connects it to classification: with cross-entropy $H(P,Q) = -\sum_j p_j\log q_j$ and entropy $H(P) = -\sum_j p_j\log p_j$,

$$\mathrm{KL}(P\|Q)=H(P,Q)-H(P).$$

Training a classifier on one-hot targets minimises $H(P,Q)$. Since $H(P)$ is fixed by the data, that is the same as minimising the divergence.

From logits. Let $z\in\mathbb{R}^k$ be logits, and let $\mathrm{LSE}(z)=\log\sum_j e^{z_j}$. Then $\log p_j = z_j-\mathrm{LSE}(z)$. Compute the log-probabilities this way and never as $\log(\text{softmax})$, because a probability can underflow to exactly $0$ while its logit is still finite.

## Worked example

Let $P=(0.5,\,0.5)$ and $Q=(0.25,\,0.75)$. Then

$$\mathrm{KL}(P\|Q)=0.5\log\tfrac{0.5}{0.25}+0.5\log\tfrac{0.5}{0.75}=0.5\cdot0.6931-0.5\cdot0.4055=0.1438\text{ nats}.$$

The reverse is $\mathrm{KL}(Q\|P)=0.25\log\tfrac{0.25}{0.5}+0.75\log\tfrac{0.75}{0.5}=-0.1733+0.3041=0.1308$ nats. The two directions differ, so the order matters.

Now let $Q=(1,\,0)$. Then $\mathrm{KL}(P\|Q)$ contains $0.5\log\tfrac{0.5}{0}$, so it is $+\infty$. But $\mathrm{KL}(Q\|P)=1\cdot\log\tfrac{1}{0.5}+0\cdot(\ldots)=\log 2\approx0.693$, because the zero term in $Q$ contributes nothing.

## Why it works

The divergence is the expected log-likelihood ratio. Draw $x\sim P$ and compare $\log p(x)$ with $\log q(x)$. Its mean is the KL. So it measures the average extra code length, in nats, when data from $P$ are encoded with a code built for $Q$. The forward direction $\mathrm{KL}(P\|Q)$ weights errors by $p_j$, so $Q$ must cover every class where $P$ has mass. It is **mean-seeking**. The reverse direction $\mathrm{KL}(Q\|P)$ weights errors by $q_j$, so $Q$ can concentrate on one mode of $P$ and leave the others empty. It is **mode-seeking**.

Maximum-likelihood training minimises $\mathrm{KL}(\text{data}\|\text{model})$, which is forward. Variational inference and the VAE regulariser $\mathrm{KL}(q(z\mid x)\|p(z))$ use the approximate posterior first, which is reverse with respect to the model. For diagonal Gaussians the KL has a closed form, which p02 implements.

## Signal vs. noise

Keep three things: the direction, the units, and the two zero rules. Those three explain nearly every wrong answer. The exact value of a KL is less important than its sign and the direction of the asymmetry. Choosing a base for interview answers is noise; state "nats" and move on.

## Where it breaks

Five failure modes recur:

1. **Swapped arguments.** `torch.nn.functional.kl_div(input, target)` expects `input` as log-probabilities, $\log Q$, and `target` as probabilities $P$ by default. Passing the model's probabilities as `input`, or logits as `target`, gives a finite and plausible number for the wrong quantity. Its default `reduction="mean"` divides the total by the batch size *and* the number of classes, so the value is too small by a factor of $k$. Use `reduction="batchmean"`, which divides by the batch size only and gives the per-row KL averaged over the batch.
2. **Log-target mismatch.** With `log_target=True`, the target must be log-probabilities too. Mixing the two conventions gives a wrong number, not a divergence.
3. **Masked classes.** A class with logit $-\infty$ has $p=0$. Its term must be dropped, not computed as $0\cdot(-\infty)$.
4. **Underflow.** For logits that differ by more than about 745, $e^{z}$ underflows to exactly zero in float64. A log of the probability returns $-\infty$ and the sum becomes `nan`.
5. **Q with a hole.** A model that puts zero probability where data appears produces an infinite divergence. That is a correct answer, and it should surface as `inf`, not be hidden.

## Interview angle

Give the definition, then say which direction you mean and why. Mention the zero rules, that the value is non-negative and asymmetric, and that KL = cross-entropy minus entropy. For a VAE, say that the regulariser is the KL from the approximate posterior to the prior, and that the closed form exists for diagonal Gaussians. The likely follow-up is "why not use KL as a distance?". The answer is that it is asymmetric and can be infinite.

## Connections

The divergence builds on [entropy and mutual information (`cml/entropy-and-mutual-information`)](../../cml/entropy-and-mutual-information/lesson.md), which supplies $H(P)$, and on [cross-entropy (`cml/cross-entropy`)](../../cml/cross-entropy/lesson.md), which supplies $H(P,Q)$. It needs [softmax (`cml/softmax`)](../../cml/softmax/lesson.md) for the logits. Later topics in this track, the VAE loss and knowledge distillation, reuse it; those topics are not in the content tree yet. Historical source: Kullback and Leibler (1951), who defined the quantity as a measure of information for discrimination between hypotheses.
