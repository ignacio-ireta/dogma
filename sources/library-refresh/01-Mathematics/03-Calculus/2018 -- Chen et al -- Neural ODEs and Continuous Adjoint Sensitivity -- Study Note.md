# Neural ODEs and Continuous Adjoint Sensitivity

**Work:** Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud, *Neural Ordinary Differential Equations* (NeurIPS 2018). The paper introduces a continuous-depth neural modeling approach and applies established adjoint sensitivity ideas; it does not originate differential equations or adjoints.

A residual update $z_{k+1}=z_k+h f_\theta(z_k,t_k)$ resembles an Euler discretization. Instead of fixing a stack of such updates, one can define
$$
\dot z(t)=f_\theta(z(t),t),\qquad z(t_0)=z_0,
$$
and let a numerical solver approximate $z(T)$. The learned object is a vector field. The solver determines where and how often to evaluate it, subject to numerical tolerances.

The mathematical model needs a well-defined solution. Continuity in time and local Lipschitz continuity in the state give local existence and uniqueness under standard hypotheses; extension to the requested interval requires that the solution remain in a region where it exists. Smoothness in state and parameters supports the sensitivity equations. Neither a neural parameterization nor a solver call alone supplies these conditions.

Forward sensitivity to a parameter obeys
$$
\dot S=J_zf\,S+\partial_\theta f,\qquad
S(t_0)=\partial_\theta z_0.
$$
When there are many parameters and one scalar terminal loss $L=\Phi(z(T))$, propagating the whole matrix $S$ can be expensive. Define a column adjoint with terminal condition $a(T)=\nabla_z\Phi(z(T))$. Then
$$
\dot a(t)=-J_zf(z(t),t)^\top a(t),
$$
and, for parameter-independent $z_0$ and no direct parameter term in $\Phi$,
$$
\nabla_\theta L=\int_{t_0}^{T}
[\partial_\theta f(z(t),t)]^\top a(t)\,dt.
$$
The adjoint runs backward; the displayed gradient integral uses forward limits. Parameter-dependent initial conditions add $[\partial_\theta z_0]^\top a(t_0)$.

A scalar check makes the signs tangible. For $\dot z=\theta z$, $z(0)=1$, and $L=z(T)$, the exact solution is $e^{\theta T}$. The adjoint is $a(t)=e^{\theta(T-t)}$, so the integrand $a(t)z(t)=e^{\theta T}$ gives $\partial_\theta L=T e^{\theta T}$, matching direct differentiation.

Chen et al. propose reconstructing the trajectory while solving an augmented system backward, avoiding storage of every forward step. Their memory claim concerns this strategy as a function of the number of forward solver steps. It does not mean zero solver memory, fixed evaluation cost, or an automatic guarantee of numerically exact gradients. Reverse reconstruction can amplify error. Also, differentiating a continuous ODE and then discretizing need not produce exactly the derivative of a particular discretized solver at finite tolerance. The scalar test above is a useful check of both state and gradient error.

The paper also connects continuous dynamics to density estimation:
$$
\frac{d}{dt}\log p_t(z(t))
=-\operatorname{tr}J_zf_\theta(z(t),t),
$$
under the smooth flow and density assumptions in its instantaneous change-of-variables theorem. Divergence measures local volume expansion, converting a calculus object into a likelihood contribution.

The conceptual route is existence/uniqueness → vector-field evolution → [reverse sensitivity](2018%20--%20Baydin%20et%20al%20--%20Automatic%20Differentiation%20as%20Executable%20Chain%20Rules%20--%20Study%20Note.md) → trainable continuous-time models. Irregular observation times and learned dynamics motivate the model; numerical conditioning determines whether the computation is dependable. This supplies a foundation for studying later differential-equation models without presenting a 2018 method as a survey of the 2026 frontier.

## Sources and locators

- Chen et al., Neural Ordinary Differential Equations, §1, pp.1–2 (continuous-depth formulation); §2 and Algorithm 1, pp.2–3 (adjoints); §4, pp.4–5 (continuous normalizing flows and change of variables). [Official NeurIPS record](https://proceedings.neurips.cc/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html). The copy used has 13 PDF pages; supplementary appendices mentioned in its text are not asserted to be included.
- Lebl, Basic Analysis I, version 6.3, §6.3, printed pp.246–254, and §7.6, pp.296–300, give Picard and fixed-point foundations. [Author edition](https://www.jirka.org/ra/realanal.pdf).
- Baydin et al., AD survey, §3.2, pp.12–14, for discrete reverse-mode sensitivity. [JMLR record](https://jmlr.org/papers/v18/17-468.html). Numerical qualifications above distinguish the continuous mathematical derivative from its computed approximation.

