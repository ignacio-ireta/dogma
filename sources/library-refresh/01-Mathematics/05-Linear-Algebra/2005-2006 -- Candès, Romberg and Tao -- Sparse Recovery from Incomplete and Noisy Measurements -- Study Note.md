# Sparse Recovery from Incomplete and Noisy Measurements

Library synthesis and study commentary

An underdetermined linear system has many solutions. A least-norm pseudoinverse chooses one, but it need not recover the particular sparse signal that generated the measurements. Sparse recovery changes the admissible signal class and the optimization criterion. The decisive question becomes whether the measurements distinguish that class reliably in the presence of noise.

## Structure replaces missing equations

Use \(A\in\mathbb R^{m\times n}\), \(m<n\), and

\[
y=Ax_0+e,\qquad \|e\|_2\le\epsilon.
\]

Assume \(x_0\) has at most \(s\) nonzero coordinates in the stated representation. The decoder is

\[
\hat x\in\operatorname*{arg\,min}_x\|x\|_1
\quad\text{subject to}\quad\|Ax-y\|_2\le\epsilon.
\]

The sparsity assumption alone is insufficient: two sparse vectors can give identical measurements. If a nonzero \(2s\)-sparse vector lies in the kernel, it can be split into a difference of two \(s\)-sparse vectors, making uniform identification impossible. Measurement design must rule out these ambiguities and control near ambiguities.

The restricted isometry constant \(\delta_t\) quantifies this control:

\[
(1-\delta_t)\|z\|_2^2\le\|Az\|_2^2
\le(1+\delta_t)\|z\|_2^2
\]

for every vector \(z\) supported on at most \(t\) coordinates. This is approximate length preservation on a union of coordinate subspaces, rather than on the entire ambient space.

## The paper's stability result

Candès, Romberg and Tao's Theorem 1 assumes

\[
\delta_{3s}+3\delta_{4s}<2.
\]

It concludes that every \(s\)-sparse signal is recovered within \(C\epsilon\) in Euclidean norm. The constant depends on the measurement matrix's restricted isometry bounds, not on the individual signal or noise realization. This is a sufficient condition from this particular paper, not an assertion that its constants are optimal.

Theorem 2 adds the approximation term for a signal that is only approximately sparse:

\[
\|\hat x-x_0\|_2\le C\epsilon+
C' {\|x_0-x_{0,s}\|_1\over\sqrt s},
\]

where \(x_{0,s}\) retains the largest \(s\) coefficients. Noise and model mismatch are therefore separate sources of error. Exact recovery follows in the noiseless, exactly sparse case under the hypotheses; arbitrary dense vectors still cannot be reconstructed from fewer independent measurements than coordinates.

The proof combines feasibility, which confines the measurement error, with the geometry imposed by \(\ell_1\) minimization. Random matrix ensembles can provide suitable measurement operators with high probability, but a specific dataset or learned dictionary does not inherit restricted isometry merely because some entries look random.

## Why this belongs beside numerical linear algebra

The [pseudoinverse route](1955-1970%20--%20Penrose%2C%20Golub%20and%20Reinsch%20--%20Least%20Squares%2C%20Pseudoinverses%20and%20Numerical%20Stability%20--%20Concept%20Note.md) minimizes solution norm, while sparse decoding favors a different structure. Neither criterion is universally correct; it expresses a prior about which solutions matter. Sparse coding, compressed acquisition and inverse problems connect that prior to machine learning. A learned sparse representation must still be justified for the data and paired with an adequate measurement operator.

There is a useful conceptual connection to [Random Projection and the Johnson-Lindenstrauss Guarantee](2003%20--%20Dasgupta%20and%20Gupta%20--%20Random%20Projection%20and%20the%20Johnson-Lindenstrauss%20Guarantee%20--%20Study%20Note.md): both analyze geometry after dimension reduction, but a finite-set distance theorem and uniform recovery over all sparse supports have different quantifiers. This is not a claim that one historical paper alone caused the other. The 2005 preprint itself cites preceding exact-recovery work and positions noise robustness as its additional problem.

## Primary reading and locators

- Candès, Romberg and Tao, Stable Signal Recovery: complete arXiv:math/0503066v2, dated 7 December 2005; §1.1 defines restricted isometry, Theorems 1–2 appear on preprint pp. 3–4, and §§2.1–2.2 give their proofs.
- [arXiv record](https://arxiv.org/abs/math/0503066) verifies version history. The subsequent journal article is *Communications on Pure and Applied Mathematics* 59(8), 1207–1223 (2006), as listed on [Candès's Stanford profile](https://profiles.stanford.edu/emmanuel-candes). The preprint copy used has its own pagination.
