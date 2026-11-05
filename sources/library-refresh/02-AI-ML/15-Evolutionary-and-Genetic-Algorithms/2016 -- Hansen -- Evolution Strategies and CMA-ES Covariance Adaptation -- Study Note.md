# Evolution Strategies and CMA-ES Covariance Adaptation

**Works:** Nikolaus Hansen, *The CMA Evolution Strategy: A Tutorial*, first submitted 2016; arXiv v2, revised 10 March 2023. Tim Salimans and colleagues, *Evolution Strategies as a Scalable Alternative to Reinforcement Learning*, arXiv v2, 7 September 2017.

Evolution strategies optimize through sampled perturbations and selection. In a $(\mu,\lambda)$ strategy, the next parents come from $\lambda$ offspring; a $(\mu+\lambda)$ strategy also allows old parents to survive. CMA-ES learns a Gaussian search distribution for continuous minimization rather than performing classical bit-string crossover.

At generation $t$, sample $z_k\sim\mathcal N(0,I)$ and
$$
x_k=m_t+\sigma_t C_t^{1/2}z_k,\qquad C_t\succ0.
$$
The sampling covariance is $\sigma_t^2C_t$, not $C_t$ alone. Order offspring by objective, and set $y_i=(x_{i:\lambda}-m_t)/\sigma_t$, $y_w=\sum_{i=1}^{\mu}w_i y_i$, where positive weights sum to one. The new mean is $m_{t+1}=m_t+\sigma_t y_w$; $\mu_{\rm eff}=1/\sum_iw_i^2$ quantifies effective selection mass.

To isolate the mechanism, a positive-weight update in the branch where the path gate equals one is
$$
p'_c=(1-c_c)p_c+\sqrt{c_c(2-c_c)\mu_{\rm eff}}\,y_w,
$$
$$
C'=(1-c_1-c_\mu)C+c_1p'_c{p'_c}^{\!\top}
+c_\mu\sum_{i=1}^{\mu}w_i y_i y_i^\top.
$$
Choose nonnegative learning rates with $c_1+c_\mu<1$. This simplified form preserves positive definiteness. The full algorithm also specifies step-size adaptation, the path gate and its correction, and can use carefully controlled negative covariance weights. These equations are an explanation, not complete implementation pseudocode.

A worked two-dimensional example makes the outer products visible. Start with $C=I$, suppress the path term ($c_1=0$), take $c_\mu=0.2$, and suppose two equally weighted selected steps are $(2,0)$ and $(0,1)$. Their second-moment matrix is $\operatorname{diag}(2,0.5)$, so $C'=\operatorname{diag}(1.2,0.9)$. At fixed $\sigma$, sampling now spreads farther along the first axis. Replacing those steps by correlated vectors would create off-diagonal entries. This is a selected-step second moment about the previous mean, not an ordinary centered covariance estimate of the winning points.

The evolution path accumulates directions across generations. Step-size control separately uses a path formed from whitened mean steps, comparing its length with a standard Gaussian reference. Eigenvectors and eigenvalues of $C$ describe the search ellipsoid; [vector spaces and inner products](../../01-Mathematics/05-Linear-Algebra/2024%20--%20Axler%20--%20Vector%20Spaces%2C%20Inner%20Products%20and%20Representation%20Geometry%20--%20Study%20Note.md) supplies the necessary linear algebra. Covariance adaptation may improve conditioning, but noisy ranking, limited evaluations, poor initialization and constraints still matter. A learned $C$ is not an exact inverse-Hessian certificate.

Salimans et al. provide a different modern bridge. For reward $F$, fixed $\sigma>0$ and suitable integrability,
$$
J_\sigma(\theta)=\mathbb E_{\epsilon\sim\mathcal N(0,I)}F(\theta+\sigma\epsilon),\quad
\nabla J_\sigma=\frac1\sigma\mathbb E[F(\theta+\sigma\epsilon)\epsilon].
$$
A sample average estimates a gradient of a smoothed objective without differentiating the simulator. This isotropic estimator does not adapt CMA-ES covariance. For the illustrative scalar reward $F(\theta)=-\theta^2$, $J_\sigma=-\theta^2-\sigma^2$, so its gradient is $-2\theta$: smoothing happens not to shift this example's optimum. That coincidence need not hold for multimodal rewards.

Parallel workers can evaluate perturbations, but wall-clock speed, total environment interactions and communication are distinct resources. Report them separately. Neither paper turns black-box sampling into the [optimality certificate](../../01-Mathematics/08-Optimization/2004%20--%20Boyd%20and%20Vandenberghe%20--%20Convexity%2C%20Duality%20and%20KKT%20Certificates%20--%20Study%20Note.md) available for structured convex problems.

## Sources and locators

- Nikolaus Hansen — The CMA Evolution Strategy: A Tutorial: §§1–4 and Appendix A; §§3.2–3.4 distinguish covariance updates and cumulation. [Version record](https://arxiv.org/abs/1604.00772v2).
- Tim Salimans — Evolution Strategies as a Scalable Alternative to Reinforcement Learning: §2, Algorithm 1, and §2.1; §3 reports task-specific experiments. [Version record](https://arxiv.org/abs/1703.03864v2).
