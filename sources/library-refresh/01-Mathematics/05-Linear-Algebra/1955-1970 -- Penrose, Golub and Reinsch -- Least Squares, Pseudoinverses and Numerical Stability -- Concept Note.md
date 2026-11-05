# Least Squares, Pseudoinverses and Numerical Stability

Library synthesis and study commentary

Least squares answers an approximation problem; a pseudoinverse selects a canonical solution; a stable algorithm controls numerical error. These are three separate requirements. The missing link between an inverse formula and a trustworthy regression is understanding which requirement a method actually satisfies.

Begin with [vector spaces and projection geometry](2024%20--%20Axler%20--%20Vector%20Spaces%2C%20Inner%20Products%20and%20Representation%20Geometry%20--%20Study%20Note.md) for the geometric prerequisites.

## Projection, QR and a unique fit

For \(A\in\mathbb R^{m\times n}\), \(m\ge n\), minimize \(\|Ax-b\|_2\). If \(A\) has full column rank, write its reduced QR factorization as \(A=QR\), where \(Q^TQ=I_n\) and \(R\) is nonsingular upper triangular. Orthogonal decomposition gives

\[
\|Ax-b\|_2^2
=\|Rx-Q^Tb\|_2^2+\|(I-QQ^T)b\|_2^2.
\]

Solve \(R\hat x=Q^Tb\) by triangular substitution. The residual is perpendicular to the column space. Although \(\hat x=(A^TA)^{-1}A^Tb\) is an exact formula under the same rank assumption, forming it is a different computational decision.

For the spectral condition number,

\[
\kappa_2(A^TA)=\kappa_2(A)^2.
\]

Normal equations therefore expose rounding errors to a squared condition number. Orthogonal transformations offer a safer route: a Householder reflection \(H=I-2vv^T/(v^Tv)\), for nonzero \(v\), preserves Euclidean norms and can eliminate entries without explicitly forming \(A^TA\). This explains modern QR practice; it does not mean the 1970 paper invented every form of QR.

## Singular and underdetermined systems

Penrose characterizes a unique generalized inverse \(A^+\) by

\[
AA^+A=A,\quad A^+AA^+=A^+,\quad
(AA^+)^*=AA^+,\quad(A^+A)^*=A^+A.
\]

For the compact rank-\(r\) SVD, \(A=U_r\Sigma_rV_r^*\), it is \(A^+=V_r\Sigma_r^{-1}U_r^*\). This works for rectangular and rank-deficient matrices. The vector \(A^+b\) is the minimum-Euclidean-norm member of the least-squares solution set. The products \(AA^+\) and \(A^+A\) are orthogonal projectors onto the column and row spaces, respectively.

The pseudoinverse does **not** automatically suppress noise: a small positive singular value is inverted and can greatly amplify perturbations. Truncated SVD deliberately discards selected directions. Ridge regression, with \(\lambda>0\), instead applies factors \(\sigma_i/(\sigma_i^2+\lambda)\) to \(u_i^*b\). Both change the estimator; a numerical rank threshold must reflect scale and uncertainty rather than treating every nonzero bit as signal.

## What stable computation does and does not fix

Golub and Reinsch reduce a real rectangular matrix to bidiagonal form with Householder transformations and then use a QR iteration on that structured matrix. Their paper explicitly cites Golub and Kahan's 1965 work, making that particular historical connection documented.

Backward stability means a computed answer solves a nearby problem. It cannot make an ill-conditioned problem insensitive to data. Least-squares forward sensitivity also depends on the residual geometry, so replacing normal equations with QR does not remove all large-error scenarios. In AI, correlated features, nearly duplicate embeddings and ill-conditioned local optimization systems make these distinctions operational.

Continue to [Matrix Differentials for Machine Learning](2012%20--%20Petersen%20and%20Pedersen%20--%20Matrix%20Differentials%20for%20Machine%20Learning%20--%20Study%20Note.md) for derivatives of fitted objectives and [Randomized SVD as a Computational Bridge](2011%20--%20Halko%2C%20Martinsson%20and%20Tropp%20--%20Randomized%20SVD%20as%20a%20Computational%20Bridge%20--%20Study%20Note.md) for reducing decomposition cost.

## Primary reading and locators

- Penrose, A Generalized Inverse for Matrices (1955): Theorem 1, pp. 406–407, establishes existence and uniqueness of the four-equation inverse. [Publisher record](https://doi.org/10.1017/S0305004100030401).
- Golub and Reinsch, Singular Value Decomposition and Least Squares Solutions (1970): §1.1, pp. 403–404, illustrates the numerical loss from forming \(A^TA\); §§1.2–1.3 give the two computational stages. [Springer record](https://link.springer.com/article/10.1007/BF02163027).
- Boyd and Vandenberghe, Introduction to Applied Linear Algebra: §§10.4, 11.5, 12.1–12.3 and 15.4. Axler, Linear Algebra Done Right, 4th edition: §6C gives a modern geometric pseudoinverse treatment.
