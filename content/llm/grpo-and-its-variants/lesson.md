---
title: "GRPO and its variants"
minutes: 12
min_viable: "Read The system and Worked example, then do p01 using Hint 1 if stuck."
---

## The system

For each prompt $x$, sample a **group** of $G$ responses $o_1, \dots, o_G$ from the old policy
$\pi_{\text{old}}$. Score each with a reward $r_i$. The advantage of response $i$ is its reward
measured against its own group:

$$\hat A_i = \frac{r_i - \operatorname{mean}(r_1, \dots, r_G)}{\operatorname{std}(r_1, \dots, r_G) + \epsilon}$$

This lesson uses the **population** standard deviation, which divides by $G$. This is a course
convention, not a claim about the paper. The sample version divides by $G - 1$, so its standard
deviation is larger by $\sqrt{G/(G-1)}$ and every advantage is smaller by $\sqrt{(G-1)/G}$. The sign
and the group-centred meaning are the same either way, but the scale is not. Use $\epsilon$ small, around $10^{-6}$.

Per token, with $\rho_{i,t} = \pi_\theta(o_{i,t} \mid x, o_{i,<t}) / \pi_{\text{old}}(o_{i,t} \mid x, o_{i,<t})$
(the probability ratio of the token under the new and the old policy), the objective is the clipped
surrogate with a KL penalty to the reference:

$$\min\big(\rho_{i,t} \hat A_i,\ \operatorname{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\,\hat A_i\big) - \beta\, k_{3,i,t}$$

$$k_{3,i,t} = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$$

Training maximises this objective. In code the loss is its negative. The token terms are averaged
per response, over its $|o_i|$ tokens, and then averaged over the group. Each response counts once,
whatever its length; a token-level average over the whole batch would weight long responses more.

## Worked example

Group of $G = 4$ with rewards $[1, 0, 0, 1]$. Mean $0.5$, population standard deviation $0.5$, so
$\hat A \approx [1, -1, -1, 1]$ (the $\epsilon$ moves each value in the sixth decimal). The sample
standard deviation is $\sqrt{1/3} \approx 0.577$, which gives $\pm 0.866$. A group of all-correct rewards $[1, 1, 1, 1]$ has
standard deviation $0$, so every advantage is $0$, and that prompt gives no policy gradient.

Clip at $\varepsilon = 0.2$ with $\hat A = +1$. If $\rho = 1.5$, the surrogate is $\min(1.5, 1.2) =
1.2$: the clipped branch wins and the token gets no policy gradient. If $\rho = 0.5$, the surrogate is
$\min(0.5, 0.8) = 0.5$: the unclipped branch wins and the gradient flows. With $\hat A = -1$, $\rho =
0.5$ gives $\min(-0.5, -0.8) = -0.8$, the clipped branch.

## Why it works

GRPO replaces PPO's learned value model with the group mean. PPO estimates advantages with
Generalized Advantage Estimation, which needs a critic $V(s_t)$ of about the policy's size, trained
to predict the final reward from each partial response. DeepSeek-R1 notes that this is hard when only
the final outcome is rewarded and responses are long
[corpus: library-papers/2501.12948.pdf p.15]. With a group, the baseline is the mean of $G$ sampled
rewards for the same prompt, so the critic is unnecessary. The clipped ratio is the PPO mechanism, and
the group-relative advantage is the replacement for the critic.

The KL term also differs from PPO. R1 adds the estimator to the loss, while PPO puts a per-token KL
penalty into the reward [corpus: library-papers/2501.12948.pdf p.15]. The estimator $k_3$ is
non-negative for every sample, and its expectation under $\pi_\theta$ is the KL divergence from the
policy to the reference [Schulman 2020, cited in R1].

## Signal vs. noise

**Signal.** The group-centred advantage; the all-equal group gives no gradient; the clip range
bounds how far one update moves a token; the $k_3$ estimator is non-negative.

**Noise.** The exact $\epsilon$ and the reward scale, provided the advantage is well defined. Those
matter for numerics, not for the method.

## Where it breaks

- **All-equal groups.** When every sample is right or every sample is wrong, the advantages are
  zero and the group teaches nothing. Track the fraction of such groups.
- **Small $G$.** With a few samples the group mean and standard deviation are noisy, so advantages
  are noisy. Dividing by the standard deviation also magnifies groups with small reward spread.
- **Normalisation choices change the weights.** Dividing by the standard deviation, and averaging
  per response instead of per token, change how much each prompt and each long response count. Some
  later GRPO variants change one or both of these choices. I have not checked their exact formulas
  here, so read the variant's paper before comparing runs.
- **Reward hacking** applies as it does to any learned or rule-based reward.

## Interview angle

**60-second answer.** For each prompt I sample $G$ responses from the old policy and score them. The
advantage is the reward minus the group mean, divided by the group standard deviation, so there is no
value model. The loss is PPO's clipped ratio on each token, with a $k_3$ KL penalty to the reference,
averaged per response and over the group. A group with equal rewards gives zero advantage, so it
produces no policy gradient.

**Follow-ups.** See `interview.md`: why no critic is needed, and what the clip does to a token
whose ratio has moved too far.

## Connections

Prerequisites: [DPO](../direct-preference-optimization-dpo/lesson.md) (sequence log-probs, the response
mask, a frozen reference) and [REINFORCE](../../rl/reinforce/lesson.md), the policy-gradient base that
GRPO's clipped objective extends. The KL term plays the anchoring role of DPO's $\beta$ term. GRPO
drops PPO's critic, the PPO for language models topic in the syllabus. The lineage runs from PPO (2017)
to DeepSeekMath's GRPO (2024) to R1's verifiable-reward training (2025).
