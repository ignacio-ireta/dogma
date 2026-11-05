# Matrix Differentials for Machine Learning

Library synthesis and study commentary

Matrix calculus is the interface between scalar objectives and linear-algebra computations. Expanding every parameter into indices is possible, but it hides shapes and transposes. The useful step is to differentiate the whole computation while keeping track of the inner product that defines a gradient.

## State the convention first

For a differentiable real scalar function \(f(W)\), with unconstrained \(W\in\mathbb R^{d\times r}\), use the Frobenius convention

\[
df=\langle\nabla_Wf,dW\rangle_F
=\operatorname{tr}((\nabla_Wf)^TdW).
\]

The differential is a linear function of a perturbation; the gradient is its matrix representation under the chosen inner product. Both shapes and the metric matter. Complex parameters require conjugation conventions, and symmetric or orthogonal constraints require permitted tangent directions. A formula copied without these assumptions can be wrong despite looking familiar.

For fixed \(X\in\mathbb R^{n\times d}\), targets \(Y\in\mathbb R^{n\times r}\), and \(E=XW-Y\), consider

\[
f(W)=\tfrac12\|E\|_F^2.
\]

Since \(dE=X\,dW\),

\[
df=\operatorname{tr}(E^TX\,dW),\qquad
\nabla_Wf=X^T(XW-Y).
\]

This derivation gives the correct \(d\times r\) shape immediately. Setting the gradient to zero recovers normal equations, but does not dictate a numerical solver. The [least-squares note](1955-1970%20--%20Penrose%2C%20Golub%20and%20Reinsch%20--%20Least%20Squares%2C%20Pseudoinverses%20and%20Numerical%20Stability%20--%20Concept%20Note.md) explains why a stationary-point equation and a stable implementation are different questions.

## Three identities that repay derivation

For a constant square matrix \(A\),

\[
\nabla_x(x^TAx)=(A+A^T)x.
\]

The shortcut \(2Ax\) needs symmetry. The antisymmetric part contributes zero to the quadratic form, which is an immediate check on the result.

If \(C\) is invertible, differentiating \(CC^{-1}=I\) gives

\[
d(C^{-1})=-C^{-1}(dC)C^{-1}.
\]

For symmetric positive-definite \(C\),

\[
d\log\det C=\operatorname{tr}(C^{-1}dC).
\]

These identities enter Gaussian log likelihoods, covariance estimation and differentiable linear systems. They assume one remains in the domain where the inverse or real logarithm is defined. Singular limits can make the derivative unbounded; automatic differentiation does not erase that conditioning issue.

For a linear layer \(Z=XW\) and an upstream derivative \(G=\nabla_Zf\), the chain rule gives \(\nabla_Wf=X^TG\) and \(\nabla_Xf=GW^T\). Backpropagation here is application of adjoint linear maps. One can obtain these expressions through differentials rather than explicitly constructing a large Jacobian.

## Place in the lineage and AI use

Scalar multivariable calculus supplies differentiation; inner-product duality converts differentials into gradients; adjoints make composition efficient. This is a conceptual prerequisite chain. Petersen and Pedersen's 2012 reference collects established identities rather than originating matrix differentiation or backpropagation.

The distinction becomes practical for [From Latent Semantic Analysis to Embeddings and Low-Rank Adaptation](1990-2021%20--%20Library%20--%20From%20Latent%20Semantic%20Analysis%20to%20Embeddings%20and%20Low-Rank%20Adaptation%20--%20Concept%20Note.md): differentiating a loss through \(BA\) produces gradients for two small factors, yet the optimization problem is still determined by the full model and data. Low parameter count alone does not certify good conditioning or a globally optimal solution.

As a diagnostic exercise, derive the gradient of \(\tfrac12\|XW-Y\|_F^2+\tfrac\lambda2\|W\|_F^2\), check its dimensions, and compare directional derivatives against \(\langle\nabla f,H\rangle_F\). Such a check tests the mathematical convention, not just a memorized pattern.

## Primary reading and locators

- Petersen and Pedersen, The Matrix Cookbook: title page specifies the 15 November 2012 version; §§2.1–2.2 cover determinant and inverse derivatives; §§2.4–2.8 cover scalar forms, traces, norms and structured matrices. [Complete DTU-hosted reference](https://www2.imm.dtu.dk/pubdb/edoc/imm3274.pdf).
- Boyd and Vandenberghe, Introduction to Applied Linear Algebra: Appendix C, especially §C.1, supplies the derivative-to-optimization bridge; §§18.1–18.3 apply it to nonlinear least squares.
