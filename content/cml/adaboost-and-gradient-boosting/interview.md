# Interview notes: AdaBoost and gradient boosting

## Q1. What is gradient boosting, and how does AdaBoost fit into it?

**60-second answer.** Boosting builds an additive model in stages, $F_m=F_{m-1}+\nu h_m$. Each weak learner $h_m$ is fitted to the negative gradient of the loss at the current model. Under squared error that gradient is the residual, so gradient boosting with stumps fits a stump to the residuals and adds a shrunken step. AdaBoost is the same idea with exponential loss $e^{-yF}$ for labels in $\{-1,+1\}$. Its weights are $w_i\propto e^{-y_iF_{m-1}}$. A learner with weighted error $\varepsilon$ gets the vote $\alpha=\tfrac12\ln\frac{1-\varepsilon}{\varepsilon}$, and the next weights shift toward the samples it got wrong.

**Follow-ups.**
- *Why does the learning rate matter?* It limits each stage's step, so more rounds with a smaller rate usually generalise better, at more compute.
- *How do you choose the number of rounds?* Early stopping on held-out loss.

## Q2. Your AdaBoost model does well on training data and poorly on a new batch. What would you check?

**60-second answer.** First, label noise: AdaBoost's exponential loss gives a persistently wrong point growing weight each round, so the model fits noise. I would check the labels of the highest-weight samples and compare against a robust loss. Second, too many rounds: the training error keeps falling while the validation error rises, so I would stop by validation loss and shrink the learning rate. Third, distribution shift: if the new batch differs from training, no number of rounds fixes that. I would look at the features before tuning the model.

**Follow-ups.**
- *What if a weak learner has error 0?* It is perfect, so $\alpha$ is infinite. Stop the loop there.
- *What if the error is above one half?* The learner is worse than chance. Its vote is not useful, so stop.

## Common mistakes

- Using AdaBoost's exponential loss on data with label noise without checking the highest-weight samples.
- Taking the residual of the wrong model, for example one stage behind, which quietly slows or breaks the fit.
- Treating the learning rate and the number of rounds as independent.
- Continuing a loop after an error of one half or zero, instead of stopping.
