# Graphical models and variational inference

A joint distribution over many variables may be easy to write and prohibitively expensive to sum. Graphical models express local structure; inference algorithms exploit that structure or approximate the resulting probability calculations. This distinction separates model specification from the cost of answering a query.

## Sources and reading locators

Graphical Models, Exponential Families, and Variational Inference is a 301-page author manuscript associated with the 2008 monograph. Its cover contains placeholder publication fields; it is not the publisher's 305-page edition. The [author publication list](https://people.eecs.berkeley.edu/~jordan/publications.html) identifies the published counterpart and its 2003 report predecessor. Read §§2.1–2.2 for graphical structure, §2.5.1 for tree messages, §§3.2, 3.5–3.6 for exponential families and duality, and §§5.1–5.2 for mean field. Maximum Likelihood from Incomplete Data via the EM Algorithm, §§2–3, supplies the companion latent variable optimization perspective.

## Exact structure and a small calculation

A directed model factorizes as a product of conditional distributions associated with parent sets. An undirected model uses nonnegative potentials and a normalizer:

\[
p_\theta(x)=\exp\{\langle\theta,\phi(x)\rangle-A(\theta)\}.
\]

For regular finite exponential families,
\(\nabla A(\theta)=E_\theta\phi(X)\) and
\(\nabla^2 A(\theta)=\operatorname{Cov}_\theta(\phi(X))\).
Inference and likelihood gradients therefore share the task of computing expectations.

Consider a three-node binary chain \(X_1\to X_2\to X_3\). Let \(P(X_1=1)=0.5\), and let each transition copy its parent with probability \(0.9\). Then \(P(X_2=1)=0.5\), and

\[
P(X_2=1\mid X_3=1)
=\frac{0.5(0.9)}{0.5(0.9)+0.5(0.1)}
=0.9.
\]

For the same evidence, \(X_1\) and \(X_3\) agree after two transitions with probability \(0.9^2+0.1^2=0.82\), so
\(P(X_1=1\mid X_3=1)=0.82\).
These are normalized sums over hidden states. On a longer chain, messages reuse partial sums instead of enumerating every complete assignment.

## Approximation as optimization

For a candidate distribution \(q\), the finite-state identity

\[
A(\theta)=
E_q\langle\theta,\phi(X)\rangle+H(q)
+D_{\rm KL}(q\Vert p_\theta)
\]

shows how maximizing energy plus entropy recovers the true distribution. Restricting \(q\) to a tractable family yields a lower bound. In naive mean field, \(q(x)=\prod_iq_i(x_i)\); the approximation removes correlations even when the true model contains them.

For the chain above, the uniform product distribution gives
\(q(X_1=1,X_2=1)=0.25\), whereas the true joint probability is \(0.45\). Matching single-variable marginals therefore does not establish an accurate joint approximation. Downstream queries involving coordinated events can be wrong even when individual probabilities look plausible.

## Lineage and AI relevance

[probability spaces and limit laws](1933%20--%20Kolmogorov%20--%20Probability%20Spaces%2C%20Conditioning%20and%20Limit%20Laws%20--%20Study%20Note.md) provides conditioning and expectation. Exponential families connect likelihood to sufficient statistics; entropy and KL supply an optimization view; graph structure determines which computations can be reused. [EM](1977%20--%20Dempster%2C%20Laird%20and%20Rubin%20--%20EM%20and%20Latent%20Variable%20Likelihood%20--%20Study%20Note.md) alternates exact posterior inference and fitting, while variational methods permit a restricted posterior family.

Trees permit exact message passing under the usual finite-state setup. Graphs with cycles can require much larger intermediate objects; loopy message updates do not inherit a universal exactness or convergence guarantee. A sparse graph is therefore not by itself a guarantee of inexpensive inference.

This framework informs hidden-state sequence models, structured prediction, Bayesian latent variable models and amortized variational inference. A graph states probabilistic factorization assumptions; causal meaning requires additional semantics. The next bridge is [causal identification](2020%20--%20Hern%C3%A1n%20and%20Robins%20--%20Identification%20Before%20Estimation%20--%20Study%20Note.md), where observational fit alone does not identify an intervention.
