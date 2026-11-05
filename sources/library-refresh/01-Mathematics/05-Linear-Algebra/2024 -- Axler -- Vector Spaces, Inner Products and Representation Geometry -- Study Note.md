# Vector Spaces, Inner Products and Representation Geometry

Library synthesis and study commentary

The missing foundation is the distinction between a vector, its coordinates, and the geometry imposed on its space. An array is a representation; a linear map is an operation on vectors that can be represented by different matrices. This distinction explains why changing an embedding basis may preserve some quantities while changing others.

## From coordinates to geometry

Work in finite-dimensional spaces over \(\mathbb R\) or \(\mathbb C\). A basis gives unique coordinates, and a linear map \(T:V\to W\) satisfies

\[
T(au+bv)=aT(u)+bT(v).
\]

If bases change, the matrix changes although the map does not. Rank measures the dimension of the range, while rank–nullity gives \(\dim V=\dim\ker T+\dim\operatorname{range}T\). An inverse exists only when the map is both injective and surjective. These statements organize rectangular feature transformations before any decomposition is computed.

An inner product adds lengths, angles and orthogonality. Over real coordinates, \(\langle x,y\rangle_G=x^TGy\) is an inner product when \(G\) is symmetric positive definite. The usual dot product chooses \(G=I\); it is a choice of geometry, not an automatic consequence of storing vectors in arrays. Orthogonal transformations preserve this Euclidean geometry. A general invertible transformation preserves linear independence but need not preserve distances or cosine similarities.

For nonzero real vectors,

\[
\cos\theta={x^Ty\over\|x\|_2\|y\|_2}.
\]

Thus normalization changes which information a retrieval score uses: cosine removes magnitude, while an unnormalized dot product retains it. The zero vector requires a separate policy because its cosine is undefined. No theorem about vector spaces by itself establishes that either score captures semantic relevance.

## Projection supplies the bridge

For an orthonormal-column matrix \(Q\), \(P=QQ^*\) projects onto its column space. Every vector decomposes into \(Px+(I-P)x\), and the two terms are orthogonal. Consequently \(Px\) uniquely minimizes distance to that subspace. This is the geometric reason least squares works; the normal equations are its coordinate expression.

The SVD extends the same viewpoint to a map between different spaces: input directions, singular strengths and output directions. Centering a data matrix before PCA changes the question from distance to the origin to variation around the sample mean. A useful exercise is to construct two vectors whose nearest-neighbor ordering changes after stretching one coordinate; this separates basis invariance from metric invariance concretely.

## Place in the lineage

Grassmann's vector-space history is outside this import; treat this modern treatment as a prerequisite chain. Axler's 2024 exposition is not a historical cause of earlier numerical algorithms. Its value here is a clean route from spaces and maps to projections, pseudoinverses and SVD. Boyd and Vandenberghe supply the complementary computational and data-modeling route.

Continue to [Least Squares, Pseudoinverses and Numerical Stability](1955-1970%20--%20Penrose%2C%20Golub%20and%20Reinsch%20--%20Least%20Squares%2C%20Pseudoinverses%20and%20Numerical%20Stability%20--%20Concept%20Note.md) for sensitivity and stable computation, or [Random Projection and the Johnson-Lindenstrauss Guarantee](2003%20--%20Dasgupta%20and%20Gupta%20--%20Random%20Projection%20and%20the%20Johnson-Lindenstrauss%20Guarantee%20--%20Study%20Note.md) for what geometry can survive dimensionality reduction.

## Primary reading and locators

- Axler, Linear Algebra Done Right, 4th edition: chapters 1–3; §§6A–6C on inner products, orthonormal bases and minimization; §§7E–7F on SVD and approximation. The bibliographic edition is 2024; the author periodically corrects the online PDF.
- Boyd and Vandenberghe, Introduction to Applied Linear Algebra (2018): §§1.4, 3.4 and 5.3–5.4; §10.4 introduces QR. These locators were checked against the full source's contents.
- Provenance: [Axler's author site](https://linear.axler.net/) and [Boyd's Stanford book page](https://stanford.edu/~boyd/vmls/) provide the complete books with stated web-distribution permission.
