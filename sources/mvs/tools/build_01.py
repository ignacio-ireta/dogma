"""Generate 01_linear_algebra_and_covariance_geometry.ipynb."""

from nbbuild import (SETUP_CELL, answer_key_banner, build, close, code, header,
                     learn, md, practice, predict, retrieve)

cells = []

cells.append(header(
    title="01 · Linear Algebra and Covariance Geometry",
    est_time="90–120 min",
    objectives=[
        "Read matrix shapes, transpose, and multiplication as operations on data (rows = observations).",
        "Centre and standardize data, and build covariance and correlation matrices from the formula.",
        "Interpret eigenvectors/eigenvalues, positive definiteness, quadratic forms, and Mahalanobis distance geometrically.",
        "Use the SVD for projection, rank, and low-rank reconstruction — the engine under PCA.",
    ],
    prereq="00 (method map). High-school matrix arithmetic; basic mean/variance.",
    mvp="Do 🔁 Retrieve, the **covariance-from-scratch** cell, and the **covariance ellipse + eigenvectors** cell. "
        "Those are the geometric facts every later method reuses.",
    extra="\n\nThis is the **toolbox** notebook. Each object below has a forward link — a margin note saying "
          "*which later method uses it*. If a later notebook feels mysterious, the missing piece is almost always here.",
))

cells.append(code(SETUP_CELL))

# ---------------------------------------------------------------- Retrieve
cells += retrieve([
    "From 00: which two of the five methods are supervised?",
    "From 00: a researcher wants new axes capturing maximum variance of many correlated variables — which method, and what is its goal-verb?",
    "Pre-req: write the formula for the sample variance of a column x (with n values and mean x̄).",
    "Pre-req: if every value of a variable is multiplied by 10, what happens to its variance?",
])

# ---------------------------------------------------------------- Predict
cells += predict([
    "A covariance matrix S is always **symmetric**. Predict one more property it must have (think about what variances along any direction must be).",
    "You standardize each column of X (mean 0, variance 1) and then compute its covariance matrix. What special matrix do you predict you get?",
    "Mahalanobis distance is 'distance that accounts for covariance'. Predict: relative to plain Euclidean distance, will it stretch or shrink directions of high variance?",
    "PCA, regression, clustering, discriminant analysis. Predict which one is **not** built directly on the covariance matrix.",
])

# ---------------------------------------------------------------- Learn
cells.append(learn("Learn the system — the objects and what they become later"))

cells.append(md(r"""
Think of every dataset as a matrix $X$ of shape $n \times p$: **$n$ rows = observations**, **$p$ columns
= variables**. Almost everything is built from three operations on it.

| Object | Definition | Geometric meaning | **Used later in** |
|---|---|---|---|
| **Centring** | $X_c = X - \bar x$ | move the cloud's centre to the origin | PCA, FA, regression |
| **Standardizing** | $Z = X_c / s$ (per column) | rescale each axis to unit spread | PCA, clustering, discriminant |
| **Covariance** | $S = \tfrac{1}{n-1} X_c^\top X_c$ | shape/orientation of the cloud | PCA, FA, LDA, Mahalanobis |
| **Correlation** | $R = D^{-1} S D^{-1}$, $D=\mathrm{diag}(s)$ | covariance of standardized data | correlation-PCA, FA |
| **Eigenvectors of $S$** | $S v = \lambda v$ | axes of the covariance ellipse | **PCA directions**, LDA |
| **Eigenvalues** $\lambda$ | variance along each axis | length² of ellipse semi-axes | variance explained |
| **Quadratic form** | $q(w)=w^\top S w$ | variance of the projection $Xw$ | PCA objective |
| **Mahalanobis** | $d^2=(x-\mu)^\top S^{-1}(x-\mu)$ | Euclidean distance *after* whitening | clustering, LDA, outliers |
| **SVD** | $X_c = U\Sigma V^\top$ | rotate–stretch–rotate | **PCA engine**, low-rank approx |

**Positive (semi)definiteness.** Since $w^\top S w = \operatorname{Var}(Xw) \ge 0$ for every $w$, a
covariance matrix is **positive semi-definite**: all eigenvalues $\ge 0$. It is **positive definite**
(all eigenvalues $>0$, hence invertible) exactly when no linear combination of the variables is constant —
which is why perfect collinearity breaks methods that need $S^{-1}$ (Mahalanobis, LDA, OLS).

**Rank** = number of nonzero singular values = number of truly independent directions. A rank-deficient
$X$ has redundant variables (a forward warning for multicollinearity in regression).
"""))

cells.append(md(r"""
#### Visual 1 — a matrix *is* a transformation of space

A 2×2 matrix $A$ sends every point $g$ to $Ag$. Watch a square grid get sheared/stretched. PCA, in
reverse, *finds* the transformation that turned a round cloud into your data.
"""))
cells.append(code(r"""
g = np.array([[x, y] for x in np.linspace(-2, 2, 9) for y in np.linspace(-2, 2, 9)])
A = np.array([[2.0, 1.0],
              [0.0, 1.0]])
gt = g @ A.T

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.5, 4.2))
a1.scatter(*g.T, s=10, color=P.ORIGINAL); a1.set_title("before:  unit grid")
a2.scatter(*gt.T, s=10, color=P.COMPONENT); a2.set_title("after:  g · Aᵀ  (sheared + stretched)")
for a in (a1, a2):
    a.set_aspect("equal"); a.axhline(0, color=P.NEUTRAL, lw=.6); a.axvline(0, color=P.NEUTRAL, lw=.6)
plt.suptitle("Q: what does multiplying by a matrix DO to space?"); plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""
#### Visual 2 — covariance ellipse and its eigenvectors

The covariance matrix's eigenvectors are the **axes** of the data ellipse; the square-roots of its
eigenvalues are the **semi-axis lengths**. This single picture is the geometric heart of PCA, Mahalanobis
distance, and LDA.
"""))
cells.append(code(r"""
S_true = np.array([[4.0, 2.2],
                   [2.2, 2.0]])
Xc_demo = RNG.multivariate_normal([0, 0], S_true, size=400)
S = np.cov(Xc_demo, rowvar=False)
vals, vecs = np.linalg.eigh(S); order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]

fig, ax = plt.subplots(figsize=(5.2, 5))
ax.scatter(*Xc_demo.T, s=7, color=P.ORIGINAL, alpha=0.4)
plots.cov_ellipse([0, 0], S, ax, n_std=2, edgecolor=P.HIGHLIGHT)
for i in range(2):
    d = vecs[:, i] * 2 * np.sqrt(vals[i])
    ax.annotate("", xy=d, xytext=(0, 0), arrowprops=dict(color=P.COMPONENT, width=2, headwidth=9))
    ax.text(*d * 1.15, f"v{i+1}\nλ={vals[i]:.1f}", color=P.COMPONENT, weight="bold")
ax.set_aspect("equal"); ax.set_title("Q: how do eigenvectors of S relate to the cloud's shape?")
plt.show()
"""))

# ---------------------------------------------------------------- Practice
cells.append(practice())

cells.append(md(r"""
**P1 (Compute · covariance from scratch, code-completion).** Build the covariance matrix from the formula
$S=\frac{1}{n-1}X_c^\top X_c$ and confirm it matches `np.cov`.
"""))
cells.append(code(r"""
X = np.array([[2.0, 1.0],
              [4.0, 0.0],
              [6.0, 2.0],
              [8.0, 5.0]])     # 4 obs, 2 variables

# TODO: centre X, then form the covariance matrix by hand.
# HINT 1 (concept): centre = subtract each column's mean.
# HINT 2 (formula): S = (Xc.T @ Xc) / (n - 1)
# HINT 3 (code): Xc = X - X.mean(0); S_hand = Xc.T @ Xc / (len(X) - 1)
S_hand = None   # <- replace

def _v():
    assert_close(S_hand, np.cov(X, rowvar=False), name="covariance from scratch")
selfcheck(S_hand is not None, _v)
"""))

cells.append(md(r"""
**P2 (Compute · standardize, then correlation).** Standardize `X`, recompute covariance, and check it
equals the **correlation** matrix of the original `X` (your Predict-#2 hypothesis).
"""))
cells.append(code(r"""
# TODO: standardize each column of X (subtract mean, divide by std), call it Z.
# HINT 1: per-column. HINT 2: use ddof=1 std to match np.corrcoef. HINT 3: Z = (X - X.mean(0)) / X.std(0, ddof=1)
Z = None   # <- replace

def _v():
    assert_close(np.cov(Z, rowvar=False), np.corrcoef(X, rowvar=False),
                 tol=1e-8, name="cov(standardized) == correlation")
    print("Confirmed: standardizing first turns covariance into correlation.")
selfcheck(Z is not None, _v)
"""))

cells.append(md(r"""
**P3 (Compute · quadratic form by hand).** With $S=\begin{pmatrix}4&2\\2&3\end{pmatrix}$ and
$w=(1,1)^\top$, the projected variance is $w^\top S w$. Compute it by hand, then verify.

*Hand:* $Sw=(4{+}2,\,2{+}3)=(6,5)$; $w^\top S w = 6+5 = 11$.
"""))
cells.append(code(r"""
S2 = np.array([[4.0, 2.0], [2.0, 3.0]]); w = np.array([1.0, 1.0])
# TODO: enter the scalar value of wᵀ S w you computed by hand.
qf_by_hand = None   # <- replace (a single number)
def _v():
    assert_close(qf_by_hand, w @ S2 @ w, name="quadratic form wᵀSw")
    print("This number is the variance of the projection X·w — exactly what PCA maximizes over w.")
selfcheck(qf_by_hand is not None, _v)
"""))

cells.append(md(r"""
**P4 (Interpret · Mahalanobis distance).** Two points can be equally far in Euclidean terms but very
different in Mahalanobis terms, because covariance reshapes "far". Run and answer.
"""))
cells.append(code(r"""
mu = np.array([0.0, 0.0])
Sm = np.array([[4.0, 0.0], [0.0, 1.0]])     # x-variance 4, y-variance 1
Sinv = np.linalg.inv(Sm)
a = np.array([2.0, 0.0])     # along high-variance axis
b = np.array([0.0, 2.0])     # along low-variance axis

def mahalanobis(x):
    d = x - mu
    return float(np.sqrt(d @ Sinv @ d))

print(f"Euclidean   : a={np.linalg.norm(a):.2f}, b={np.linalg.norm(b):.2f}  (equal)")
print(f"Mahalanobis : a={mahalanobis(a):.2f}, b={mahalanobis(b):.2f}")

th = np.linspace(0, 2*np.pi, 200)
fig, ax = plt.subplots(figsize=(4.8, 4.6))
plots.cov_ellipse(mu, Sm, ax, n_std=1, edgecolor=P.HIGHLIGHT)
plots.cov_ellipse(mu, Sm, ax, n_std=2, edgecolor=P.NEUTRAL)
ax.scatter(*a, color=P.FIT, s=60, label="a (along wide axis)")
ax.scatter(*b, color=P.RESIDUAL, s=60, label="b (along narrow axis)")
ax.set_aspect("equal"); ax.legend(fontsize=8)
ax.set_title("Q: why is b 'farther' than a in Mahalanobis terms?"); plt.show()
"""))
cells.append(md(r"""
**YOUR ANSWER (P4):** In one sentence — why does Mahalanobis call `b` farther than `a` even though they
are the same Euclidean distance from the centre? (Name which later methods rely on this idea.)
"""))

cells.append(md(r"""
**P5 (Interpret · SVD = singular values are retained information).** Low-rank reconstruction keeps the
top-$k$ singular directions. The drop in singular values tells you how compressible the data is — this
*is* PCA's explained variance, viewed through the SVD.
"""))
cells.append(code(r"""
from sklearn.datasets import load_iris
iris = load_iris()
Xi = iris.data - iris.data.mean(0)         # centre
U, sv, Vt = np.linalg.svd(Xi, full_matrices=False)

def rank_k(k):
    return (U[:, :k] * sv[:k]) @ Vt[:k]

ks = range(1, 5)
err = [np.linalg.norm(Xi - rank_k(k)) / np.linalg.norm(Xi) for k in ks]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
a1.bar(range(1, 5), sv, color=P.COMPONENT); a1.set_title("Q: singular values of iris"); a1.set_xlabel("direction")
a2.plot(list(ks), err, "-o", color=P.RESIDUAL); a2.set_title("Q: relative reconstruction error vs k")
a2.set_xlabel("rank k kept"); a2.set_ylabel("‖X−X̂‖ / ‖X‖"); plt.tight_layout(); plt.show()
print("singular values:", np.round(sv, 2), "-> first one dwarfs the rest, so rank-1 already captures most structure.")
"""))

cells.append(md(r"""
**P6 (Recognize · positive definiteness).** For each matrix, predict positive-definite / semi-definite /
neither *before* running. The eigenvalues confirm it.
"""))
cells.append(code(r"""
mats = {
    "A": np.array([[2.0, 0.0], [0.0, 3.0]]),
    "B": np.array([[1.0, 1.0], [1.0, 1.0]]),     # singular
    "C": np.array([[1.0, 2.0], [2.0, 1.0]]),     # indefinite
}
for name, M in mats.items():
    ev = np.linalg.eigvalsh(M)
    kind = ("positive definite" if (ev > 1e-9).all()
            else "positive semi-definite" if (ev >= -1e-9).all()
            else "indefinite (NOT a covariance matrix)")
    print(f"{name}: eigenvalues {np.round(ev,2)}  ->  {kind}")
"""))
cells.append(md(r"""
**YOUR ANSWER (P6):** Which of A, B, C could be a real covariance matrix, and which one would make
Mahalanobis distance impossible to compute (and why)?
"""))

# ---------------------------------------------------------------- Close
cells.append(close())

cells.append(md(r"""
**Exam-style closing (5 sentences).** Explain, to a classmate, why the eigen-decomposition of the
covariance matrix is "the same calculation" that PCA performs — mention covariance, eigenvectors,
eigenvalues, variance, and orthogonality.
"""))
cells.append(md("**YOUR 5 SENTENCES:**\n\n"))

cells.append(md(r"""
#### 🃏 Foundations card — fill from memory

| Object | Formula | What it becomes later |
|---|---|---|
| Covariance matrix |  |  |
| Eigenvector of S |  |  |
| Eigenvalue of S |  |  |
| Mahalanobis distance |  |  |
| SVD of centred X |  |  |
"""))

cells.append(md(r"""
**Transfer question.** In regression you will project the target $y$ onto the column space of the
predictors $X$. Using only this notebook's ideas, what does "project onto the column space" mean
geometrically, and what is the *residual* in that picture? (One or two sentences.)
"""))
cells.append(md("**YOUR ANSWER:**\n\n"))

cells.append(md(r"""
**Suggested Anki prompts**
- Front: *Why is every covariance matrix positive semi-definite?* → Back: wᵀSw = Var(Xw) ≥ 0 for all w.
- Front: *Standardize then take covariance = ?* → Back: the correlation matrix.
- Front: *Eigenvalues of S mean…?* → Back: variance along each principal axis (ellipse semi-axis lengths²).
- Front: *Mahalanobis distance formula?* → Back: √((x−μ)ᵀ S⁻¹ (x−μ)) — Euclidean distance after whitening.
- Front: *Rank of X = ?* → Back: number of nonzero singular values = independent directions.
"""))

# ---------------------------------------------------------------- Answer key
cells.append(answer_key_banner())
cells.append(md(r"""
**🔁 Retrieve**
1. Multiple regression and discriminant/logistic analysis.
2. PCA; goal-verb = reduce.
3. $s^2=\frac{1}{n-1}\sum_i (x_i-\bar x)^2$.
4. It is multiplied by $10^2=100$.

**🔮 Predict** (model answers)
1. It must be positive semi-definite (no direction can have negative variance).
2. The identity-like correlation matrix (1's on the diagonal); for standardized data covariance *is* correlation.
3. It **shrinks** distances along high-variance directions (and stretches low-variance ones) — it divides by the spread.
4. Clustering's *core* doesn't need S (it can run on raw Euclidean distance), though Mahalanobis/standardized variants do use it; PCA/LDA/regression all lean on S directly. (Either clustering or "regression uses XᵀX not Cov per se" earns credit if argued.)

**🛠️ Practice**
- *P1:* `Xc = X - X.mean(0); S_hand = Xc.T @ Xc / (n-1)` — matches `np.cov`.
- *P2:* `Z = (X - X.mean(0)) / X.std(0, ddof=1)`; cov(Z) == corrcoef(X).
- *P3:* $w^\top S w = 11$.
- *P4:* `b` points along the *narrow* (low-variance, σ²=1) axis, so 2 units there is "many standard deviations", while `a`'s 2 units lie along the *wide* (σ²=4) axis = only 1 sd. Mahalanobis divides by spread per direction. Used by clustering (standardized/Mahalanobis distance), LDA, and outlier detection.
- *P5:* iris singular values ≈ [25.1, 6.0, 3.4, 1.9]; the first dominates, so rank-1/2 reconstructs iris with small error — the same fact PCA reports as "PC1 explains most variance."
- *P6:* A positive definite, B positive semi-definite (one eigenvalue 0 → singular), C indefinite. A and B are valid covariance matrices; **B** makes Mahalanobis impossible because it is singular (no $S^{-1}$). C is not a covariance matrix at all.

**Transfer:** Projecting $y$ onto the column space of $X$ drops a perpendicular from $y$ onto the flat
(sub)space spanned by the predictor columns; the fitted values are the foot of that perpendicular and the
**residual** is the perpendicular segment $y-\hat y$, orthogonal to every predictor.
"""))

build("01_linear_algebra_and_covariance_geometry.ipynb", cells)
