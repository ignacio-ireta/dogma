---
title: "REINFORCE"
minutes: 14
min_viable: "Read The system and Worked example, then implement p01 using Hint 1 if needed."
---

## The system

A policy $\pi_\theta(a\mid s)$ is a softmax over logits $z_\theta(s)\in\mathbb{R}^{A}$. The
probability of action $a$ is $\pi_\theta(a\mid s)=\operatorname{softmax}(z_\theta(s))_a$. An episode
of length $T$ gives states $s_0,\dots,s_{T-1}$, actions $a_0,\dots,a_{T-1}$, and rewards
$r_1,\dots,r_T$, where $r_{t+1}$ follows the action at step $t$. In the code, `rewards[b, t]` is
$r_{t+1}$ for episode $b$, so the 0-based array index is one less than the subscript. The reward-to-go is

$$G_t=\sum_{k=0}^{T-1-t}\gamma^k\,r_{t+k+1},$$

with the discount $\gamma\in[0,1]$. The objective is $J(\theta)=\mathbb{E}_{\pi_\theta}[G_0]$.

REINFORCE uses the score-function identity, $\nabla_\theta\mathbb{E}_{x}[f(x)]=\mathbb{E}[f(x)\,\nabla_\theta\log p_\theta(x)]$,
to estimate the gradient from samples:

$$\nabla_\theta J(\theta)\approx\sum_t G_t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t).$$

For a batch of $B$ episodes, the loss to minimise is

$$\mathcal{L}(\theta)=-\frac{1}{B}\sum_{b=1}^{B}\sum_{t=0}^{T-1}\log\pi_\theta(a_{b,t}\mid s_{b,t})\,A_{b,t},$$

where $A_{b,t}=G_{b,t}$ without a baseline. Minimising $\mathcal{L}$ ascends $J$.

## Worked example

One episode, $T=2$, $\gamma=1$, rewards $r_1=0$ and $r_2=1$. The reward-to-go is $G_0=1$ and
$G_1=1$. Let the policy at each step put probability $0.5$ on each of two actions. Suppose step 0
takes action 1, and step 1 takes action 0. Then $\log\pi(1\mid s_0)=\log\pi(0\mid s_1)=\log 0.5$.
The loss is

$$\mathcal{L}=-\big[\log 0.5\cdot G_0+\log 0.5\cdot G_1\big]=-2\log 0.5\approx 1.386.$$

Gradient: the score of a softmax with respect to its logits is $\mathrm{onehot}(a)-\pi$. At step 0 this is
$(-0.5,\,0.5)$, and the loss gradient is $-G_0$ times that: $(0.5,\,-0.5)$. Gradient descent on the
logits therefore raises the logit of action 1, the action that was rewarded.

## Why it works

The score-function identity holds because $\nabla_\theta p=p\,\nabla_\theta\log p$, so the expected
value of a function times the score is the gradient of that expectation [book: Sutton and Barto,
ch. 13]. The policy gradient theorem gives the reward-to-go form: the effect of an action on the
objective depends only on the rewards that follow it. So $G_t$ replaces the whole-episode return,
which would add variance from rewards the action cannot influence.

A baseline $b_t$ that does not depend on $a_t$ leaves the expectation unchanged, because
$\mathbb{E}[\nabla_\theta\log\pi_\theta(a\mid s)]=0$ for a fixed state. Replacing $G_t$ by $G_t-b_t$
is therefore unbiased, and a good $b_t$ lowers the variance. A batch mean of the returns at the
same step is a standard choice. Strictly, it includes the episode itself, so it is not independent of the sample. The
expected gradient shrinks by a factor $(1-1/B)$, a bias of order $1/B$ that vanishes as the batch
grows. A baseline fitted on other data avoids that.

## Signal vs. noise

The signal is the loss form and the reward-to-go. The sign is the most common error, so check it
with the worked example: the rewarded action's logit must go up. The softmax score is standard
algebra and you should be able to write it from memory. The baseline is a variance tool, not a new
objective. Ignore entropy bonuses, clipping and multi-epoch updates for this session. They belong
to the later actor-critic and PPO topics.

## Where it breaks

Taking a log of a softmax in two steps overflows or produces $-\infty$ for large logits. Use a
log-softmax, which subtracts the maximum first. Averaging over $T$ as well as $B$ changes the
effective step size by a factor of $T$. Normalising the returns to unit spread changes the scale of
every update, so the learning rate must be retuned, and it is not an unbiased fix. Finally, a
detached logit gives no gradient at all, and the loss will not move.

## Interview angle

Start with the score-function identity, then the reward-to-go loss, then the sign. Expect two
follow-ups. What does the baseline buy? Lower variance with no change in the expected gradient,
provided it does not depend on the action. Why normalise? To keep the step size stable across
batches, at the cost of changing the weighting. The full answers are in `interview.md`.

## Connections

REINFORCE needs the discounted returns from the first topic, the softmax from the classification
block, and autograd for the gradient. Actor-critic methods replace the Monte Carlo return by a
learned value baseline, and PPO constrains how far each update moves. The vanilla policy-gradient
estimator that PPO builds on, $\hat g=\hat{\mathbb{E}}_t[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat A_t]$, is
written out in [library-papers/1707.06347 sec. 2.1].
