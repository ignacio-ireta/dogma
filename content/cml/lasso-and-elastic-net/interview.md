# Interview notes: Lasso and elastic net

## Q1. Why does the lasso give exact zeros, and ridge does not?

**60-second answer.** The L1 penalty is not differentiable at zero, and its level sets have corners on the axes. The
optimum often lands on a corner, where some coefficients are exactly zero. The one-coordinate solution makes this
explicit: the soft-threshold sets the coefficient to zero whenever the coordinate's correlation with the residual is no
larger than the penalty. Ridge's L2 penalty is smooth at zero, so the solution shrinks coefficients toward zero without
generally reaching it; exact zero can still occur through symmetry or zero correlation. That makes the lasso a selection tool as well as a shrinkage tool.

**Follow-ups.**
- *What is the smallest $\alpha$ that gives an all-zero model?* $\alpha_\max = \max_j\lvert X_j^\top\mathbf y\rvert/n$, for centred data.
- *Why does the lasso's $\ell_1$ norm decrease as $\alpha$ grows?* Each solution's optimality against the other gives the monotone bound.
- *Why not run gradient descent on the lasso objective?* The $\ell_1$ term has no gradient at zero. Subgradient
  steps keep crossing zero and rarely land on it exactly. A proximal or coordinate step does.

## Q2. What goes wrong with correlated features, and how do you fix it?

**60-second answer.** The lasso often selects one of a set of highly correlated features, and which one survives can be unstable.
A tiny change in the data can swap the winner, and the coefficients are not stable. The elastic net adds an L2 term, which
makes the solution unique and gives correlated features similar weights. If the goal is prediction, ridge also works. If the
goal is a stable selected set, I would group correlated features first, then select within the groups.

**Follow-ups.**
- *Is the arbitrary choice always a problem?* For prediction it often costs little. For interpretation it is misleading.
- *How do you pick the mixing ratio $\rho$?* Cross-validate over a grid, on standardised features.

## Q3. How would you choose $\alpha$ for a lasso model, and what should you check?

**60-second answer.** I standardise the features inside each fold, fit a grid of $\alpha$ values from $\alpha_\max$ downward,
and pick the value with the lowest validation error, often the largest $\alpha$ within one standard error of the minimum.
Then I refit on all training data. I would check that the selected features are stable across folds, because a model
that selects different features each time is not yet trustworthy. I would also check the solver's convergence, since
cyclic coordinate descent is slow on nearly collinear columns.

**Follow-ups.**
- *Can you do inference on the selected coefficients?* Not naively. Selection biases the estimates, so standard errors
  computed after selection are optimistic.
- *Why start the grid at $\alpha_\max$?* It is where the model is empty, so the path runs from no features to many.

## Common mistakes

- Fitting the lasso on unstandardised features, so the penalty depends on units.
- Adding an intercept column to the lasso and penalising it.
- Treating the selected features as causal, or reporting their coefficients as unbiased.
- Using a convergence tolerance that is too loose on correlated columns, then trusting a sparse answer that has not converged.
- Reading a single lasso fit's selected set as stable without checking folds.
