# EM and latent variable likelihood

Maximum likelihood becomes difficult when the observed density sums over hidden assignments. A mixture model is a simple example: each observation may come from several components, and its unknown component changes the parameter update. The expectation-maximization algorithm turns this coupled problem into repeated inference and fitting steps.

## Sources and reading locators

Maximum Likelihood from Incomplete Data via the EM Algorithm is the 1977 journal article with discussion, rather than its 1976 presentation record. Read §2 for definitions, §3 and Theorem 1 at printed p. 7 for likelihood monotonicity, and §4.3 for finite mixtures. [Publisher record](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x). For the variational interpretation, see Graphical Models, Exponential Families, and Variational Inference, §5.2.2, and the [author publication list](https://people.eecs.berkeley.edu/~jordan/publications.html).

## Mathematical idea and assumptions

Let \(x\) be observed and \(z\) hidden. For a normalized distribution \(q(z)\),

\[
\mathcal L(q,\theta)=
E_q[\log p_\theta(x,z)]+H(q),
\qquad
\log p_\theta(x)=
\mathcal L(q,\theta)+
D_{\rm KL}(q\Vert p_\theta(z\mid x)).
\]

The expectations and conditional distributions must be well defined, with compatible support. The KL term is nonnegative, so \(\mathcal L\) is a lower bound.

The E step sets \(q_t(z)=p_{\theta_t}(z\mid x)\), making the bound tight at the current parameter. The M step maximizes \(E_{q_t}\log p_\theta(x,z)\) over \(\theta\), with \(q_t\) held fixed. Raising this bound raises observed log likelihood. Generalized EM needs an increase of the bound rather than an exact maximum.

## Worked example: soft counts

Suppose component A produces a binary success with probability \(0.8\) and component B with probability \(0.2\). Their probabilities are known; only the mixing weight \(\pi=P(z=A)\) is unknown. Observe \(x=(1,1,0)\), and start from \(\pi_0=0.5\).

For a success, Bayes's rule gives responsibility

\[
r_1=\frac{0.5(0.8)}{0.5(0.8)+0.5(0.2)}=0.8.
\]

For a failure the responsibility is \(r_0=0.2\). Thus the E step assigns an expected \(1.8\) of the three observations to A, and the M step gives
\(\pi_1=(0.8+0.8+0.2)/3=0.6\).

The model's marginal success probability moves from \(0.5\) to \(0.56\). The observed likelihood moves from \(0.5^3=0.125\) to
\(0.56^2(0.44)=0.137984\), verifying improvement in this iteration.

A hard assignment would allocate both successes to A and the failure to B, producing \(2/3\). EM instead averages latent assignments using the current model. These are different algorithms with different objectives and convergence properties.

## What the guarantee does and does not buy

Likelihood monotonicity does not establish global optimality or rapid convergence. Stationary points, symmetries and boundary behavior matter. In unrestricted Gaussian mixtures, a component can collapse around one data point while its covariance tends to zero, making likelihood unbounded; practical implementations impose additional structure or regularization.

The lineage runs from Fisher on likelihood through missing-data likelihood to a general alternating inference/optimization method. Earlier special cases existed; the 1977 contribution organizes and analyzes a broad family. The variational decomposition connects EM to [graphical models](2008%20--%20Wainwright%20and%20Jordan%20--%20Graphical%20Models%20and%20Variational%20Inference%20--%20Study%20Note.md) and later latent variable models. If the exact posterior is computationally unavailable, restricting \(q\) yields variational inference; the tight-bound step then changes. In AI this distinction explains why an ELBO can improve while the exact marginal likelihood need not improve by the same amount, and why better inference and better parameter fitting are separate tasks.
