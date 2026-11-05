# Interview notes — Adam

## Q1. Write the Adam update and explain what each piece is for.

**60-second answer.** Adam keeps an average of gradients, $m$, and an average of squared gradients, $v$. Both start at zero, so I correct them by $1-\beta^t$ to get $\hat m$ and $\hat v$. The parameter moves by $\eta\hat m/(\sqrt{\hat v}+\epsilon)$. For consistent gradients the ratio is near one, so the step is about $\eta$ whatever the gradient scale. That is why the learning rate is easy to interpret, and why Adam is a per-coordinate method.

**Follow-ups.** Why bias correction? The zero start shrinks both averages, and the uncorrected step is about three times too large at the first update, so correction removes that inflation. Why is eps outside the root? It sets a floor on the denominator only after the scale is estimated; inside the root it would change the scale of small gradients.

## Q2. How would you test an Adam implementation without trusting your own derivation?

**60-second answer.** I compare a trajectory of many steps with a library implementation on seeded random inputs, because a single step can hide errors. I include a case with a large eps, which exposes eps placement, and a check that the first update is about the learning rate in size, which exposes bias correction. I also check the counter: the first update must use $t=1$.

**Follow-ups.** Why not compare only the first step? Bias-correction errors shrink as $t$ grows, so one step may pass by accident. Why many steps? State bugs such as a wrong counter or a stale moment only show after several updates.

## Common mistakes

- Omitting bias correction, or correcting only one moment.
- Putting eps inside the square root.
- Starting the step counter at zero and using it before incrementing.

Source: [corpus:library-papers/1412.6980.pdf.md p.3]. See lesson.md for the derivation and the paper's step bound.
