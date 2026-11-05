# Interview notes: Gaussian mixtures and EM

## Q1. Walk me through one EM iteration for a Gaussian mixture.

**60-second answer.** The E-step computes, for each sample, the responsibility of each component:
the prior weight times the component density, divided by the sum over components. Compute it in log
space and normalise with log-sum-exp, so that far-apart samples do not underflow. The M-step is
weighted maximum likelihood. Each component gets its effective count $N_j = \sum_i r_{ij}$, its
weight $N_j/n$, a responsibility-weighted mean, and a weighted scatter about the new mean. Each
iteration costs $O(nkd^2)$ for the scatter and log-density terms, plus $O(kd^3)$ for the factorisations.

**Follow-ups.**
- *Why is the covariance computed about the new mean?* That is the maximiser of the weighted
  likelihood. Using the old mean adds a bias term.
- *What if a component gets zero responsibility?* Its $N_j$ is zero, so it cannot be refitted. Give
  it weight zero and keep its parameters, rather than dividing by zero.

## Q2. Why does EM never decrease the likelihood, and does a covariance floor break that?

**60-second answer.** The log-likelihood is at least a bound $F(q,\theta)$ that holds for any
distribution $q$ over labels. The E-step sets $q$ to the responsibilities, which makes the bound
tight at the current parameters. The M-step maximises the bound over the parameters. So the bound,
and the likelihood, can only rise. A floor keeps this true if it is a constrained maximiser: clip
each eigenvalue to at least the floor. Adding $\varepsilon I$ is not the constrained maximiser, and it can lower the
likelihood.

**Follow-ups.**
- *Does the bound hold with the floor?* Yes, for the clipped maximiser, because the previous
  parameters are feasible and the M-step maximises over a set that contains them.
- *What does monotonicity not give you?* A global maximum. EM can stop at a local one.

## Q3. A component keeps shrinking to a tiny covariance. What is happening, and what do you do?

**60-second answer.** Its responsibilities have concentrated on a single point or on duplicates, so
its scatter approaches zero. The density then spikes, and the likelihood grows without bound on
that component. This is a degenerate maximum, not a good fit. Floor the eigenvalues at a small
$\varepsilon$, check for duplicates, and consider dropping the component or using a prior. Also
check that the start is not forcing the collapse.

**Follow-ups.**
- *How do you pick the floor?* Small relative to the data's scale, and tuned with a held-out
  likelihood, not the training one.
- *How do you pick $k$?* Information criteria such as BIC, or held-out likelihood. The likelihood
  alone always increases with $k$.

## Common mistakes

- Computing responsibilities without normalising, so that the rows do not sum to one.
- Using the old mean in the covariance update, or dividing the scatter by $n$ rather than $N_j$.
- Multiplying densities directly instead of log densities, which underflows far from the components.
- Calling the first converged fit the answer, without restarts from different starts.
- Adding a constant to each covariance and calling it a floor. It changes the fitted model and can lower the likelihood.
- Forgetting that the likelihood can be infinite for a collapsed component, so a huge value is not a better fit.
