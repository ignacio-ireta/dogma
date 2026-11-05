---
title: "Learning-rate warmup"
minutes: 12
min_viable: "Read The system and the first paragraph of Why it works, then solve p01 using Hint 1 if needed."
---

## The system

Warmup adds a short ramp in front of a decay schedule. Symbols: $t$ is the number of updates
taken, starting at $t=0$ for the first update in this formulation. $W$ is the warmup length, $T$
is the total length, $\eta_0$ is the peak (base) rate and $\eta_{\min}$ is the floor, with
$0\le\eta_{\min}\le\eta_0$. The common large-model schedule has three segments:

| Segment | Rule | Value at the boundary |
|---|---|---|
| Ramp, $t<W$ | $\eta_0\,t/W$ | $0$ at $t=0$ |
| Cosine, $W\le t<T$ | $\eta_{\min}+\tfrac12(\eta_0-\eta_{\min})\bigl(1+\cos(\pi p)\bigr)$, with $p=\frac{t-W}{T-W}$ | $\eta_0$ at $t=W$ |
| Floor, $t\ge T$ | $\eta_{\min}$ | $\eta_{\min}$ at $t=T$ |

The cosine phase in the middle is the same one used in the decay topic. Only its start has
moved to $t=W$. A second family, from Vaswani et al. (2017), uses an inverse square root with no
floor. It is derived in the next section and implemented in p02.

## Worked example

Take $\eta_0=1$, $W=4$, $T=12$, $\eta_{\min}=0.1$. The ramp gives $0$, $0.5$, $1$ at $t=0,2,4$.
At $t=8$ the cosine progress is $p=(8-4)/(12-4)=0.5$, so the phase is $(1+\cos(\pi/2))/2=0.5$ and
the rate is $0.1+0.5\cdot 0.9=0.55$. At $t=12$ the rate is the floor $0.1$. The rate rises for
four updates, falls for eight, and never returns to the peak.

## Why it works

Adam keeps running averages $m_t$ of gradients and $v_t$ of squared gradients, then bias-corrects
them: $\hat v_t=v_t/(1-\beta_2^t)$ [library: AI-ML/03-Optimization/Kingma Ba Adam 2015.md].
At the first update, $\hat m_1=g_1$ and $\hat v_1=g_1^2$, so each coordinate moves by about
$\alpha\,\mathrm{sign}(g_1)$, whatever the gradient's scale. Bias correction fixes the mean of the
early estimate, not its noise: $\hat v_t$ averages only a few squared gradients, so the normalized
step $\hat m_t/\sqrt{\hat v_t}$ can be large and noisy until enough gradients have arrived. A common
explanation for warmup is that a small rate in that stretch limits the damage. This is a heuristic
motivation, not a result the Adam paper proves, so check the first steps on your own curve. In
practice warmup is the usual response. The LLaMA training setup uses 2,000 warmup steps and a cosine
schedule that ends at 10% of the maximal rate [corpus:library-papers/2302.13971.pdf.md p.3].

The Vaswani schedule is the same idea with the shape made explicit. Its formula is
$\eta_t = d_{\mathrm{model}}^{-1/2}\min(t^{-1/2},\, t\,W^{-3/2})$, with the update number $t\ge 1$.
The ramp is linear, the peak sits at $t=W$, and afterwards the rate falls as $t^{-1/2}$
[corpus:library-papers/1706.03762.pdf.md p.7].

## Signal vs. noise

- **Signal:** the ramp starts at zero, the cosine starts at the peak exactly at $t=W$, and the
  floor applies from $t=T$. Most implementation errors are at these three boundaries.
- **Signal:** warmup exists for the optimizer's early state, not for the data. Ask what the first
  steps look like before tuning the peak rate.
- **Noise for now:** the exact warmup length in each published recipe. Treat it as a fraction of
  the run, then check it on your own curve.

## Where it breaks

- **Off-by-one at the peak.** A ramp of $W$ steps that reaches the peak at $t=W-1$ gives the cosine
  a different start from the one you planned. Check the value at $t=W$ by hand.
- **Numbering.** The Vaswani formula is written for update numbers from 1. A scheduler counter that
  starts at 0 must be shifted by one. Step 0 lies outside the formula, because $0^{-1/2}$ is undefined.
- **Total length versus warmup.** $T$ must exceed $W$. If the run is shorter than warmup, the cosine
  never starts.
- **Units.** Warmup and total lengths must use the same unit as the scheduler counter, as in the
  decay topic.

## Interview angle

Describe the three segments, then give the reason: the adaptive optimizer's second-moment estimate
is noisy at the start, and a common explanation is that the first large steps need a small rate. The usual answer ends with
a check: the rate at the warmup boundary, the floor after the run, and the unit of the counter. A
follow-up asks how the Vaswani schedule differs. It has no floor and no fixed end; the peak height
depends on the model width, and the decay is an inverse square root. Full answers are in
`interview.md`.

## Connections

This topic extends the decay schedules: the cosine segment is the one defined there, with its start
moved to $W$. The prerequisite [dl/adam] gives the moment estimates and their bias correction used in
the argument above. AdamW [dl/adamw-decoupled-weight-decay] uses the same schedule multiplier for
its weight-decay term. The historical line runs from the Transformer paper's schedule to the
cosine-with-warmup recipes used in later large-model work.
