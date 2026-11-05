---
title: "Scaling laws and compute budgets"
minutes: 12
min_viable: "Read The system and the Worked example, then do p01 with the two formulas written on paper."
---

## The system

Let $N$ be the number of parameters, $D$ the number of training tokens, and $C$ the total training compute in FLOPs. The working rule is

$$C\approx 6\,N\,D.$$

The factor 6 counts about 2 FLOPs per parameter per token for the forward pass and about 4 for the backward pass. Hoffmann et al. use the same approximation, citing Kaplan et al. [corpus:library-papers/2203.15556.pdf.md p.7].

Loss is modelled as a function of both sizes:

$$L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}.$$

Here $E$ is the irreducible loss of the data, $A/N^\alpha$ is the cost of a model that is too small, and $B/D^\beta$ is the cost of too little data. Hoffmann et al. fit $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$ and $\beta=0.28$ in their Approach 3 parametric fit, Eq. 10 [corpus:library-papers/2203.15556.pdf.md p.25, appendix D.2]. Treat them as one fit, not as universal constants.

**Compute-optimal allocation.** For a fixed $C$, choose $N$ and $D$ to minimise $L$ with $6ND=C$. The result is that $N$ and $D$ grow in roughly equal proportion. Table 2 gives the exponent $a$ of $N_{opt}\propto C^a$ as 0.50, 0.49 and 0.46 across the three approaches, so $C^{1/2}$ is a close approximation, not an exact law [corpus:library-papers/2203.15556.pdf.md p.8]. A useful check is the ratio $r=D/N$. Table 3 gives about 20 tokens per parameter: 400M parameters with 8.0B tokens, 1B with 20.2B, 10B with 205.1B. The larger rows reach about 22 [corpus:library-papers/2203.15556.pdf.md p.8]. Read the ratio as a rule of thumb, roughly 20, not a constant.

## Worked example

Fix $r=D/N=20$. Then $C=6N(20N)=120N^2$, so $N=\sqrt{C/120}$ and $D=20N$.

- Budget $C=1.2\times10^{20}$. $N=\sqrt{10^{18}}=10^9$ and $D=2\times10^{10}$. Check: $6\cdot10^9\cdot2\times10^{10}=1.2\times10^{20}$.
- Budget $C=5.76\times10^{23}$, the Gopher-unit budget in Table 3 [corpus:library-papers/2203.15556.pdf.md p.8]. $N=\sqrt{4.8\times10^{21}}\approx6.9\times10^{10}$, about 69 billion parameters, and $D\approx1.4\times10^{12}$ tokens. Table 3 lists 67B and 1.5T for this budget. Gopher had 280B parameters on 300B tokens [corpus:library-papers/2203.15556.pdf.md p.3, Table 1]. Its compute, $6\cdot280\cdot10^9\cdot300\cdot10^9\approx5.0\times10^{23}$, is close to this budget. So Gopher is a larger, under-trained model for about the same compute.

Integers matter at large budgets. The exact rule is $N=\lfloor\sqrt{\lfloor C/(6r)\rfloor}\rfloor$, which `math.isqrt` computes exactly, so $6ND\le C$ always holds. Worked floor: for $C=1000$ and $r=20$, $C/(6r)=8.33$, its integer floor is 8, and $\mathrm{isqrt}(8)=2$. So $N=2$, $D=40$, and $6\cdot2\cdot40=480\le1000$. The real-valued $N=2.89$ is rounded down, never up.

## Why it works

Once $C$ is fixed, loss falls with $N$ and with $D$ at different rates. Spending the budget on either term alone leaves the other term large. Minimising $L$ under $6ND=C$ balances the two. Kaplan et al. [library:AI-ML/12-Frontiers/Kaplan Scaling Laws 2020.md] found power-law trends and favoured growing the model faster than the data. Hoffmann et al. re-fitted with three methods. Their Approach 1 gives $a=0.50$ and $b=0.50$ [corpus:library-papers/2203.15556.pdf.md section 3.1 and p.8], and their Approach 3 gives $a=0.46$ and $b=0.54$. Kaplan's exponents, as Table 2 reports them, are $N_{opt}\propto C^{0.73}$ and $D_{opt}\propto C^{0.27}$ [corpus:library-papers/2203.15556.pdf.md p.8]. The problem's formula is the equal-proportion case.

**Fitting a power law.** Take logs: $\log y=\log a+b\log x$. Ordinary least squares on $(\log x_i,\log y_i)$ gives $b$ as the slope and $a=e^{c}$ from the intercept $c$. Fitting in log space weights relative errors equally, which suits data that spans decades.

## Signal vs. noise

Signal: $C\approx6ND$; the split $D=rN$; the near-equal exponents; and the power-law fit in log space. Noise: the exact constants $E, A, B, \alpha, \beta$, which depend on the data and architecture; the precise ratio 20, which Table 3 puts between about 20 and 22; and which optimiser or schedule produced the curves.

## Where it breaks

- **Extrapolation.** A line fitted on small runs can be off by a lot at large $C$. The paper notes curvature in the frontier at large budgets [corpus:library-papers/2203.15556.pdf.md appendix E]. Check the range you fit on.
- **Linear-space fits.** A least-squares fit on $y$ directly gives different constants. Fit the logs.
- **Non-positive data.** The logarithm is undefined at zero or below. Reject it before fitting.
- **Cancellation.** The expanded formula $\sum uv - n\bar u\bar v$ loses digits when the log values are close together and large. Centre the data first.
- **Ignoring data limits.** The equal-proportion rule assumes enough distinct text exists for $D=rN$. Past that point, the data is the constraint.

## Interview angle

Open with $C\approx6ND$. Then state the trade: for a fixed budget, a bigger model means fewer tokens, and the loss is minimised when both terms of $L(N,D)$ are balanced. With $D=rN$ and $r\approx20$, both $N$ and $D$ scale as $\sqrt{C}$. The paper's own exponents are only roughly 0.5. Say that the constants come from a fit and carry the range of the fit. Likely follow-ups: "why not just train a bigger model" and "how would you fit this". Full versions are in `interview.md`.

## Connections

Prerequisites: [gradient descent](../../cml/gradient-descent/) for the least-squares idea, and [model parameter and memory footprint](../model-parameter-and-memory-footprint/) for counting $N$. The lineage runs from Kaplan et al. (2020) to Chinchilla (Hoffmann et al., 2022), which revised the allocation.
