# Interview notes: Ridge regression and weight decay

## Q1. What does ridge regression do, and how would you choose the penalty?

**60-second answer.** Ridge adds $\alpha\lVert\boldsymbol\beta\rVert^2$ to the least-squares objective and leaves the
intercept unpenalised. With centred data it has a closed form, $\boldsymbol\beta = (\tilde X^\top\tilde X + \alpha I)^{-1}\tilde X^\top\tilde{\mathbf y}$.
The penalty shrinks directions where $X$ has little variance the most, which is how it trades variance for bias. The
penalty only treats features fairly after standardisation, so I scale first. I choose $\alpha$ by K-fold cross-validation
on the training data, with the scaler fitted inside each fold, then refit on all training data.

**Follow-ups.**
- *Why is ridge defined when the design is rank deficient?* Adding $\alpha I$ makes $\tilde X^\top\tilde X$ positive
  definite, so the objective is strongly convex and the minimiser is unique. Convexity alone would not guarantee that.
- *What happens as $\alpha \to \infty$?* The slopes go to zero and the intercept goes to $\bar y$.

## Q2. Is weight decay the same as L2 regularisation?

**60-second answer.** For plain SGD, yes. A step on $L + \tfrac{\lambda}{2}\lVert\mathbf w\rVert^2$ is the same as shrinking the
weights by $(1-\eta\lambda)$ and then taking the gradient step. For adaptive optimisers such as Adam, no. Adam divides each
coordinate's gradient by its running scale, so an L2 term added to the gradient is scaled too, and the effective decay
differs per parameter. AdamW applies the decay directly to the weights, which gives uniform decay. I would use AdamW whenever
the intent is weight decay.

**Follow-ups.**
- *How would you check which one your framework does?* Compare one step with zero data gradient, as in the lesson.
- *Does this matter for a linear model with SGD?* Not for the decay itself, since the two are the same there.

## Q3. A model has many correlated features. What does ridge do to them?

**60-second answer.** Ridge generally retains features and shares the weight among correlated ones on a comparable scale. Their coefficients shrink
toward each other, which stabilises them. Individual coefficients still have wide uncertainty if the correlation is very high,
and ridge does not enforce sparsity, so it is not a feature-selection method. If selection is also needed, I would look at elastic net,
which combines the L2 sharing with an L1 penalty.

**Follow-ups.**
- *Why do correlated coefficients become unstable without a penalty?* Nearly dependent columns make $X^\top X$ nearly singular,
  so small changes in the data move the coefficients a lot.
- *Does ridge fix bias?* No. It introduces bias to reduce variance, and the trade-off depends on $\alpha$.

## Common mistakes

- Penalising the intercept, which makes predictions depend on the target's offset.
- Comparing $\alpha$ values across libraries without checking whether the objective has a $1/n$ or a factor of $\tfrac12$.
- Fitting the scaler on all data before cross-validation, which leaks validation information.
- Calling L2 under Adam "weight decay" without checking the update.
- Choosing $\alpha$ on the training score, which always favours the smallest penalty.
