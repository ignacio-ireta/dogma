# Interview notes — MDPs and discounted returns

## Q1. Define an MDP and the return an agent maximises.

**60-second answer.** An MDP is a state set, an action set, a transition kernel P(s'|s,a), a reward R(s,a), and a discount gamma. The agent picks actions by a policy pi(a|s). The return G_t sums the discounted future rewards from step t. Gamma below one keeps the infinite sum finite and encodes a preference for near-term reward. Episodic tasks end at a terminal state, so gamma can be one there.

**Follow-ups.** What does the discount buy you? Finite returns in continuing tasks and a contraction in the Bellman operator. Why is the reward defined on (s,a) rather than (s,a,s')? It is the expectation over next states; the two forms are interchangeable through P.

## Q2. How do you evaluate a fixed policy in a small MDP?

**60-second answer.** Write the Bellman expectation equation as V = R_pi + gamma P_pi V, where P_pi and R_pi average over the policy. For a small state space, solve the linear system directly. For larger ones, iterate the same map from zero; it contracts by gamma in the max norm, so it converges geometrically.

**Follow-ups.** When would you prefer iteration? When the state space is large enough that a dense solve is too costly. How do you know it has converged? The max-norm change between sweeps is below tolerance, and the error is bounded by gamma over one minus gamma times that change.

## Common mistakes

- Using the immediate reward as the return, which drops the bootstrap term.
- Averaging over actions with the wrong axis, so pi is applied to next states instead of actions.
- Setting gamma to one in a continuing task and expecting a finite value.
