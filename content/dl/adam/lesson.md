---
title: "Adam"
minutes: 12
min_viable: "Read The system and Worked example, then do p01 using Hint 1 if needed."
---

## The system

Adam keeps two exponential moving averages per parameter tensor $\theta$. Let $g_t$ be the gradient at
update $t$, $\beta_1\in[0,1)$ and $\beta_2\in[0,1)$ the decay rates, $\eta$ the learning rate, and
$\epsilon>0$ a small constant. The first moment $m$ averages gradients. The second moment $v$ averages
squared gradients. Both start at zero, and $t$ starts at $0$.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^{2}\\
\hat m_t &= \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}}\\
\theta_t &= \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\end{aligned}
$$

Defaults in the paper are $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$ and $\eta=10^{-3}$
[corpus:library-papers/1412.6980.pdf.md p.3]. The counter $t$ increments before bias correction, so the
first update uses $t=1$.

## Worked example

One scalar, $\theta_0 = 1$, constant gradient $g=2$, $\eta=0.1$. Take $t=1$:
$m_1 = 0.1\cdot 2 = 0.2$ and $v_1 = 0.001\cdot 4 = 0.004$. Bias correction gives $\hat m_1 = 0.2/0.1 = 2$ and
$\hat v_1 = 0.004/0.001 = 4$. The step is $0.1\cdot 2/(2+\epsilon)\approx 0.1$, so $\theta_1 \approx 0.9$.

Second update, $t=2$: $m_2 = 0.38$ and $v_2 = 0.007996$. Then $\hat m_2 = 0.38/0.19 = 2$ and
$\hat v_2 = 0.007996/0.001999 = 4$. Again the step is about $0.1$, so $\theta_2 \approx 0.8$. For a
constant gradient, every step is about $\eta$, whatever the gradient's scale.

## Why it works

**Bias correction.** Starting from zero biases both averages toward zero. At $t=1$, $m_1$ is $10$ times
too small and $v_1$ is $1000$ times too small. Dividing by $1-\beta^t$ undoes the factor. The paper derives
this correction for $v$ and says the derivation for $m$ is analogous
[corpus:library-papers/1412.6980.pdf.md p.3].

The uncorrected update is not tiny, though. Without correction the ratio $m_t/\sqrt{v_t}$ equals
$(1-\beta_1^{\,t})/\sqrt{1-\beta_2^{\,t}}$ for a constant gradient. At $t=1$ that is $0.1/\sqrt{0.001}\approx 3.2$,
so the uncorrected first step is about $3$ times too large. The inflation peaks at about $6.6$ near $t\approx 12$ and then decays toward $1$ as $1-\beta_2^{\,t}\to 1$ (it is about $3.2$ at $t=100$).
Bias correction removes that inflation. The "too small" part is the averages themselves. The paper makes the same point: without correction, "initial steps that are much larger" follow for small $\beta_2$ [corpus:library-papers/1412.6980.pdf.md p.3].

**Step bound.** The ratio $\hat m_t/\sqrt{\hat v_t}$ is a signal-to-noise ratio, and its magnitude is near $1$ for consistent gradients. So each coordinate moves about $\eta$ per update. The paper states that $|\Delta_t|\lessapprox\eta$, and that the boundary case $(1-\beta_1)=\sqrt{1-\beta_2}$ gives $|\Delta_t|<\eta$ [corpus:library-papers/1412.6980.pdf.md p.3]. This makes $\eta$ easy to choose in scale.

**Eps placement.** The update uses $\sqrt{\hat v_t}+\epsilon$, with $\epsilon$ outside the root and after bias
correction. Placing it inside, as $\sqrt{\hat v_t+\epsilon}$, changes the step most when gradients are
small. Choosing $\epsilon$ large, as in the tests, makes the difference visible.

## Signal vs. noise

The signal is the two moments, bias correction, and the counter. The rest is tuning. $\beta_1=0.9$ and
$\beta_2=0.999$ work often enough to be defaults, and $\epsilon$ rarely needs changing. Coordinate-wise
scaling and the sign-like behaviour of the update are the parts to remember.

## Where it breaks

- **Forgotten bias correction.** The first steps are about $3$ times too large at $t=1$, and the inflation reaches about $6.6$ by $t\approx 12$ for the defaults, so training may
  diverge in the first few updates.
- **Counter starting at zero.** With $t=0$ the correction divides by $1-\beta^0=0$, giving NaN or inf.
- **Sparse gradients.** A coordinate that is zero for a long time has a tiny $\hat v$, and its first
  nonzero gradient yields a step near $\eta$ times $(1-\beta_1)/\sqrt{1-\beta_2}$. Watch rare features.
- **Generalisation.** Adam can reach a worse minimum than SGD with momentum on some tasks. Its
  per-coordinate scaling is a different geometry, not a free upgrade.
- **Weight decay.** Adding L2 to the gradient interacts with the second moment; the next topic covers this.

## Interview angle

Give the update in three lines: two moments, bias correction, and a step $\eta\hat m/(\sqrt{\hat v}+\epsilon)$.
Then say the step is about $\eta$ regardless of gradient scale, so $\eta$ carries the meaning. The usual
follow-ups are why bias correction exists and where eps goes. Full versions are in `interview.md`.

## Connections

Adam combines momentum, which is the first moment, with RMSProp's per-coordinate scale, which is the second.
Its prerequisite is the buffer-based momentum step. AMSGrad, in p02, changes one rule. AdamW, the next
topic, separates weight decay from the adaptive step. Lineage: momentum, then RMSProp, then Adam.
