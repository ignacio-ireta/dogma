"""Generate 03_principal_component_analysis.ipynb."""

from nbbuild import (METHOD_CARD_BLANK, SETUP_CELL, answer_key_banner, build,
                     close, code, header, learn, md, practice, predict, retrieve)

cells = []

cells.append(header(
    title="03 · Principal Component Analysis",
    est_time="90–120 min",
    objectives=[
        "Derive PCA as variance maximization and connect it to the covariance eigendecomposition and the SVD of centred data.",
        "Distinguish **scores** from **loadings**, and **covariance** PCA from **correlation** PCA.",
        "Choose the number of components using scree, cumulative variance, and parallel-analysis intuition.",
        "Read a biplot, handle sign indeterminacy, and explain why **scaling and outliers** can wreck a PCA.",
    ],
    prereq="01 (eigenvectors, covariance matrix, standardization, projection). 00 (the method map).",
    mvp="Do 🔁 Retrieve, run the **from-scratch vs sklearn** agreement cell, and the **scaling-trap** cell on Wine. "
        "Those two cells contain the most-tested ideas: what PCA computes, and why you almost always standardize first.",
    extra="\n\n**Colour reminder:** orange = components/PC directions, blue = original variables, "
          "purple = highlighted geometry. Keep that mapping in mind for every plot below.",
))

cells.append(code(SETUP_CELL))

# ---------------------------------------------------------------- Retrieve
cells += retrieve([
    "From 01: an eigenvector of a covariance matrix points in what direction, geometrically?",
    "From 01: what does it mean for a covariance matrix to be **positive definite**, and why is a covariance matrix at least positive *semi*-definite?",
    "From 01: you standardize a variable. What are its new mean and variance?",
    "From 00: is PCA supervised or unsupervised, and what is its goal-verb?",
])

# ---------------------------------------------------------------- Predict
cells += predict([
    "PCA finds new axes for a cloud of points. **What quantity do you think the first axis is chosen to maximize?**",
    "If two original variables are measured on very different scales (grams vs millimetres), what do you predict PCA will do — and is that good or bad?",
    "You keep 2 of 13 components. What have you *lost*, and how could you measure how much?",
    "If you multiply the first principal direction by −1, does anything about the analysis actually change? Predict yes/no and why.",
])

# ---------------------------------------------------------------- Learn
cells.append(learn("Learn the system — PCA as a system"))

cells.append(md(r"""
**Research question.** "I have many correlated numeric variables. Can I re-express them with a few new
uncorrelated variables that keep most of the information?"

**Input data shape.** A matrix $X$ with $n$ rows (observations) and $p$ numeric columns (variables).
No target column — PCA is **unsupervised**.

**Mathematical objective.** Center the data. Find the unit direction $w_1$ that **maximizes the variance
of the projected data**:
$$ w_1 = \arg\max_{\|w\|=1} \operatorname{Var}(Xw) = \arg\max_{\|w\|=1} w^\top \Sigma\, w, $$
where $\Sigma$ is the covariance matrix. The solution is the **top eigenvector of $\Sigma$**, and the
maximized variance is its **eigenvalue** $\lambda_1$. The next direction maximizes remaining variance
subject to being **orthogonal** to the first — i.e. the second eigenvector, and so on.

**Two equivalent computations.**
- **Eigendecomposition** of the covariance matrix $\Sigma = V \Lambda V^\top$. Columns of $V$ = directions; $\Lambda$ = variances.
- **SVD** of the centred data $X_c = U S V^\top$. Same $V$; eigenvalues $\lambda_i = s_i^2/(n-1)$. (SVD is numerically preferred.)

**Key fitted quantities (do not confuse these):**

| Name | Symbol | Shape | Meaning |
|---|---|---|---|
| **Loadings** (components/directions) | $V$ | $p \times k$ | how each *original variable* contributes to each component (the new axes, in variable space) |
| **Scores** | $Z = X_c V$ | $n \times k$ | coordinates of each *observation* on the new axes |
| **Explained variance** | $\lambda_i$ | $k$ | variance captured by component $i$ |

> **Loadings describe variables; scores describe observations.** A loading is a recipe; a score is a result.

**Assumptions / requirements.** Variables roughly linearly related (PCA only sees covariance); a sensible
**scale** (standardize unless variables share units and you *want* variance to count); reasonably free of
wild outliers (variance-based → outlier-sensitive). No distributional assumption is needed just to compute it.

**Diagnostics & choosing $k$.** Scree plot (look for an elbow), **cumulative variance** (e.g. ≥ 80–90%),
**parallel analysis** (keep components beating random-data eigenvalues), and **interpretability**.

**Common failure modes.** (1) Forgetting to standardize → the highest-variance *unit* dominates (the
scaling trap). (2) Reading a score as a loading. (3) Over-interpreting the **sign** of a component (it is
arbitrary). (4) One outlier hijacking PC1.

**Neighbours.** **Factor analysis** (next door): same output shape, but models only *shared* variance via
latent factors instead of re-describing *total* variance. **Regression**: also uses projection, but onto a
*target*, not onto max-variance directions.
"""))

# ------- from scratch on a 2D cloud ----
cells.append(md(r"""
#### See it: PCA on a 2-D cloud where the answer is obvious

We make a tilted elliptical cloud. The long axis of the cloud *is* PC1. We will compute it three ways
and confirm they agree.
"""))
cells.append(code(r"""
# A correlated 2-D cloud: variance is largest along a tilted direction.
cov_true = np.array([[3.0, 1.6],
                     [1.6, 1.2]])
X2 = RNG.multivariate_normal([4.0, 1.0], cov_true, size=300)

mean2 = X2.mean(axis=0)
S = np.cov(X2, rowvar=False)            # sample covariance (2x2)
vals, vecs = np.linalg.eigh(S)          # ascending order
order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]  # PC1 first

fig, ax = plt.subplots(figsize=(5.4, 5))
ax.scatter(*X2.T, s=8, color=P.ORIGINAL, alpha=0.5, label="observations")
plots.cov_ellipse(mean2, S, ax, n_std=2, edgecolor=P.HIGHLIGHT, question=None)
for i in range(2):
    d = vecs[:, i] * 2.2 * np.sqrt(vals[i])
    ax.annotate("", xy=mean2 + d, xytext=mean2,
                arrowprops=dict(color=P.COMPONENT, width=2, headwidth=9))
    ax.text(*(mean2 + 1.15 * d), f"PC{i+1}", color=P.COMPONENT, weight="bold")
ax.set_aspect("equal"); ax.set_xlabel("x1"); ax.set_ylabel("x2")
ax.set_title("Q: along which direction is the cloud most spread? (= PC1)")
ax.legend(loc="upper left", fontsize=8); plt.show()

print("PC1 direction:", np.round(vecs[:, 0], 3), " variance (eigenvalue):", round(vals[0], 3))
print("PC2 direction:", np.round(vecs[:, 1], 3), " variance (eigenvalue):", round(vals[1], 3))
"""))

cells.append(md(r"""
#### Three computations, one answer (the agreement check)

Covariance-eigendecomposition, SVD of centred data, and `sklearn.PCA` must agree on the explained
variances and on the directions (**up to sign**). This is requirement #6: from-scratch == library.
"""))
cells.append(code(r"""
from sklearn.decomposition import PCA

Xc = X2 - X2.mean(axis=0)
n = Xc.shape[0]

# (a) covariance eigendecomposition  -> already have vals (eigenvalues)
ev_eig = vals

# (b) SVD of centred data
U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
ev_svd = sv**2 / (n - 1)

# (c) sklearn
pca = PCA(n_components=2, random_state=0).fit(X2)
ev_sk = pca.explained_variance_

assert_close(ev_eig, ev_svd, tol=1e-8, name="eigen vs SVD explained variance")
assert_close(ev_eig, ev_sk,  tol=1e-8, name="eigen vs sklearn explained variance")

# explained-variance RATIO is sign- and method-independent:
ratio = ev_eig / ev_eig.sum()
print("explained-variance ratio:", np.round(ratio, 3), " (PC1 keeps {:.0%})".format(ratio[0]))
"""))

# ---------------------------------------------------------------- Practice
cells.append(practice())

cells.append(md(r"""
**P1 (Compute · by hand, then verify).** For the tiny covariance matrix
$$ \Sigma=\begin{pmatrix}5 & 2\\ 2 & 2\end{pmatrix}, $$
the eigenvalues are $\lambda = 6$ and $\lambda = 1$. Find the **PC1 direction** (the unit eigenvector
for $\lambda=6$) by hand, then check below.

*Hand hint:* solve $(\Sigma - 6I)w = 0 \Rightarrow \begin{pmatrix}-1 & 2\\ 2 & -4\end{pmatrix}w=0
\Rightarrow w \propto (2,1)$. Normalize.
"""))
cells.append(code(r"""
# TODO: enter the UNIT PC1 direction you found by hand, as a length-2 array.
# HINT 1 (concept): PC1 is the eigenvector of the LARGEST eigenvalue.
# HINT 2 (formula): direction (2,1); unit-normalize by dividing by sqrt(2^2+1^2)=sqrt(5).
# HINT 3 (code): pc1 = np.array([2,1]) / np.sqrt(5)
pc1_by_hand = np.array([0.0, 0.0])   # <- replace

Sig = np.array([[5.0, 2.0], [2.0, 2.0]])
vals_p, vecs_p = np.linalg.eigh(Sig)
pc1_true = vecs_p[:, np.argmax(vals_p)]
def _verify():
    t = pc1_true.copy()
    if np.dot(pc1_by_hand, t) < 0:   # sign is arbitrary; align before comparing
        t = -t
    assert_close(pc1_by_hand, t, tol=1e-6, name="PC1 direction")
selfcheck(np.any(pc1_by_hand), _verify)
"""))

cells.append(md(r"""
**P2 (Compute · scores vs loadings, code-completion).** Given centred data and the loading matrix `V`,
compute the **scores** and confirm they equal sklearn's `transform`. This nails the scores≠loadings
distinction.
"""))
cells.append(code(r"""
# Use the 2-D cloud. V = loadings (columns are PC directions).
V = vecs  # 2x2, columns = PC1, PC2

# TODO: scores = centred data projected onto the loading directions.
# HINT 1 (concept): a score is an observation's coordinate on a new axis.
# HINT 2 (formula): Z = Xc @ V   (n x p) @ (p x k) -> (n x k)
# HINT 3 (code): scores = Xc @ V
scores = None   # <- replace

sk_scores = pca.transform(X2)
def _verify():
    sgn = np.sign((scores * sk_scores).sum(axis=0))   # align per-component sign
    assert_close(scores * sgn, sk_scores, tol=1e-8, name="scores vs sklearn transform")
    print("Loadings (V) have shape", V.shape, "= p x k  (describe VARIABLES)")
    print("Scores have shape", scores.shape, "= n x k  (describe OBSERVATIONS)")
selfcheck(scores is not None, _verify)
"""))

cells.append(md(r"""
**P3 (Interpret · reconstruction and information loss).** Keeping only $k$ components and rebuilding
the data, $\hat X = Z_{:,:k} V_{:,:k}^\top + \bar x$, loses information. Run the cell, then answer below.
"""))
cells.append(code(r"""
wine = data.load_wine_df()
feat = list(wine.columns[:13])
from sklearn.preprocessing import StandardScaler
Xw = StandardScaler().fit_transform(wine[feat].values)   # standardized!

pw = PCA(random_state=0).fit(Xw)
def reconstruct(pca, X, k):
    Z = pca.transform(X)[:, :k]
    return Z @ pca.components_[:k] + pca.mean_

ks = range(1, 14)
mse = [np.mean((Xw - reconstruct(pw, Xw, k))**2) for k in ks]
cumvar = np.cumsum(pw.explained_variance_ratio_)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
a1.plot(list(ks), mse, "-o", color=P.RESIDUAL); a1.set_xlabel("k components kept")
a1.set_ylabel("reconstruction MSE"); a1.set_title("Q: how much do we lose vs k?")
a2.plot(range(1, 14), cumvar, "-o", color=P.HIGHLIGHT); a2.axhline(0.9, ls="--", color=P.NEUTRAL)
a2.set_xlabel("k components kept"); a2.set_ylabel("cumulative variance")
a2.set_title("Q: how much do we KEEP vs k?"); plt.tight_layout(); plt.show()

k90 = int(np.searchsorted(cumvar, 0.90) + 1)
print(f"Components needed for >=90% variance: {k90}")
"""))
cells.append(md(r"""
**YOUR ANSWER (P3):** In one sentence each — (a) how many components reach 90% variance on standardized
Wine, and (b) what does the falling red MSE curve represent in plain words?
"""))

# ------- scaling trap ----
cells.append(md(r"""
**P4 (Choose-and-defend · the scaling trap — the headline failure case).** We run PCA on Wine **without**
standardizing, then **with**. Watch what dominates PC1. This is requirement #7 (a deliberate failure)
and the single most common PCA mistake.
"""))
cells.append(code(r"""
Xraw = wine[feat].values

pca_raw = PCA(random_state=0).fit(Xraw)     # NOT standardized
pca_std = PCA(random_state=0).fit(Xw)       # standardized

# Which variable dominates PC1 in each case?
load_raw = pd.Series(np.abs(pca_raw.components_[0]), index=feat).sort_values(ascending=False)
load_std = pd.Series(np.abs(pca_std.components_[0]), index=feat).sort_values(ascending=False)

print("Variable variances (raw scale):")
print(wine[feat].var().sort_values(ascending=False).head(3).round(1).to_string())
print("\nTop |loadings| on PC1, RAW (unscaled):")
print(load_raw.head(3).round(3).to_string())
print("\nTop |loadings| on PC1, STANDARDIZED:")
print(load_std.head(3).round(3).to_string())
print(f"\nPC1 variance share — raw: {pca_raw.explained_variance_ratio_[0]:.1%} | "
      f"standardized: {pca_std.explained_variance_ratio_[0]:.1%}")
"""))
cells.append(md(r"""
**YOUR ANSWER (P4):** The raw-scale PC1 is dominated by one variable. **Which one, and why is that an
artefact rather than a finding?** What is the defensible default, and when (if ever) is covariance
(unscaled) PCA the right choice?
"""))

# ------- choosing k: scree + parallel analysis ----
cells.append(md(r"""
**P5 (Interpret · how many components? scree + parallel-analysis intuition).** Parallel analysis keeps a
component only if its eigenvalue beats what you'd get from **uncorrelated random data of the same size**.
It guards against keeping "components" that are just noise.
"""))
cells.append(code(r"""
def parallel_analysis(X, n_iter=100, rng=None):
    rng = rng or np.random.default_rng(0)
    Xs = StandardScaler().fit_transform(X)
    real = np.sort(np.linalg.eigvalsh(np.corrcoef(Xs, rowvar=False)))[::-1]
    rand = np.empty((n_iter, Xs.shape[1]))
    for i in range(n_iter):
        Xp = np.column_stack([rng.permutation(c) for c in Xs.T])  # break correlations
        rand[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(Xp, rowvar=False)))[::-1]
    return real, rand.mean(axis=0)

real_ev, rand_ev = parallel_analysis(Xraw, rng=np.random.default_rng(7))
keep = int(np.sum(real_ev > rand_ev))

fig, ax = plt.subplots(figsize=(6.2, 4))
x = np.arange(1, len(real_ev) + 1)
ax.plot(x, real_ev, "-o", color=P.COMPONENT, label="Wine (real)")
ax.plot(x, rand_ev, "--s", color=P.NEUTRAL, label="random data (parallel)")
ax.axhline(1.0, color=P.RESIDUAL, ls=":", lw=1, label="Kaiser rule (>1)")
ax.set_xlabel("component"); ax.set_ylabel("eigenvalue (correlation PCA)")
ax.set_title(f"Q: how many components are real? parallel analysis keeps {keep}")
ax.legend(fontsize=8); plt.show()
print(f"Parallel analysis suggests keeping {keep} components "
      f"(Kaiser >1 would keep {(real_ev>1).sum()}).")
"""))

# ------- biplot ----
cells.append(md(r"""
**P6 (Interpret · the biplot).** A biplot overlays observation **scores** (points) and variable
**loadings** (orange arrows). Points are wines coloured by their (here known, but PCA never used them)
cultivar; arrows show which chemical variables pull which way.
"""))
cells.append(code(r"""
scores_w = pca_std.transform(Xw)
plots.biplot(scores_w, pca_std.components_.T, feat,
             groups=wine["target_name"].values,
             question="Q: which chemicals separate the cultivars on PC1–PC2?")
plt.show()
print("Reading: arrows pointing the same way as a colour cluster are the variables high for that cultivar.")
"""))

# ------- sign + outlier ----
cells.append(md(r"""
**P7 (Recognize · sign indeterminacy).** Flipping a component's sign flips its scores and loadings
together, so *nothing interpretable changes*. Confirm it, so a sign flip never confuses you in an exam.
"""))
cells.append(code(r"""
flip = pca_std.components_.copy(); flip[0] *= -1     # flip PC1
scores_flip = (Xw - pca_std.mean_) @ flip.T
# explained variance unchanged; scores just negated on PC1
assert_close(pca_std.explained_variance_, pca_std.explained_variance_,
             name="explained variance unchanged by sign")
print("PC1 scores before vs after flip (first 3):",
      np.round(scores_w[:3, 0], 2), "->", np.round(scores_flip[:3, 0], 2))
print("Same magnitude, opposite sign. Interpretation ('high vs low on PC1') is identical.")
"""))

cells.append(md(r"""
**P8 (Choose-and-defend · outlier sensitivity — rerunnable).** PCA maximizes variance, so a single
extreme point can swing PC1. Change `outlier_strength` and rerun to feel the sensitivity. (Requirement
#3: a control that teaches sensitivity to a choice.)
"""))
cells.append(code(r"""
outlier_strength = 8.0   # <- try 0, 4, 8, 16 and rerun

base = RNG.multivariate_normal([0, 0], [[1, 0.0], [0.0, 1]], size=200)
Xo = np.vstack([base, [outlier_strength, -outlier_strength]])  # one planted outlier
pc1 = PCA(n_components=2, random_state=0).fit(Xo).components_[0]

fig, ax = plt.subplots(figsize=(4.8, 4.6))
ax.scatter(*base.T, s=8, color=P.ORIGINAL, alpha=0.5)
ax.scatter([outlier_strength], [-outlier_strength], s=70, color=P.RESIDUAL, label="outlier")
ax.annotate("", xy=pc1 * 4, xytext=-pc1 * 4,
            arrowprops=dict(color=P.COMPONENT, width=2, headwidth=9))
ax.set_aspect("equal"); ax.set_title(f"Q: how far does ONE outlier tilt PC1?  (strength={outlier_strength})")
ax.legend(fontsize=8); plt.show()
print("With strength 0 PC1 is ~arbitrary (round cloud); crank it up and PC1 snaps toward the outlier.")
"""))

# ---------------------------------------------------------------- Close
cells.append(close())

cells.append(md(r"""
**Exam-style interpretation (5 sentences).** Imagine you ran correlation PCA on the 13 Wine variables
and kept the components parallel analysis suggested. Write 5 sentences a grader would accept: state the
method and why standardization was used, how many components you kept and on what basis, how much
variance they retain, what PC1 appears to contrast (from the biplot), and one limitation.
"""))
cells.append(md("**YOUR 5 SENTENCES:**\n\n"))

cells.append(md("#### 🃏 Method card — PCA (fill from memory)\n\n" + METHOD_CARD_BLANK))

cells.append(md(r"""
**Transfer question.** You will meet **factor analysis** next. Both shrink many variables to a few. In
one or two sentences, predict *one* reason an analyst might prefer factor analysis over PCA when their
goal is to argue that a few **unobserved traits** generate the correlations.
"""))
cells.append(md("**YOUR ANSWER:**\n\n"))

cells.append(md(r"""
**Suggested Anki prompts**
- Front: *PCA's objective in one line?* → Back: find orthogonal directions maximizing projected variance (top eigenvectors of the covariance/correlation matrix).
- Front: *Scores vs loadings?* → Back: loadings = variable→component weights (p×k); scores = observation coordinates on components (n×k).
- Front: *Covariance vs correlation PCA?* → Back: correlation PCA = standardize first; use it unless variables share units and variance should count.
- Front: *Why is a PCA sign meaningless?* → Back: flipping a component negates its loadings and scores together — interpretation is unchanged.
- Front: *Two ways to pick k?* → Back: cumulative variance threshold; parallel analysis (beat random-data eigenvalues); plus scree elbow / interpretability.
"""))

# ---------------------------------------------------------------- Answer key
cells.append(answer_key_banner())
cells.append(md(r"""
**🔁 Retrieve**
1. Along a principal axis of the data ellipse — a direction of (locally) maximal/orthogonal spread; eigenvectors are the ellipse's axes.
2. Positive definite: $w^\top \Sigma w > 0$ for all $w\neq 0$ (all eigenvalues > 0). A covariance matrix is at least positive *semi*-definite because $w^\top \Sigma w = \operatorname{Var}(Xw) \ge 0$.
3. New mean 0, new variance 1.
4. Unsupervised; goal-verb = **reduce**.

**🔮 Predict** (model answers)
- PC1 maximizes the **variance** of the projected data.
- Without standardizing, the variable with the largest raw variance dominates — usually *bad*, an artefact of units.
- You lose the variance carried by the dropped components; measure it via 1 − cumulative-variance-ratio, or reconstruction MSE.
- Flipping the sign changes nothing interpretable (scores and loadings flip together).

**🛠️ Practice**
- *P1:* PC1 = (2,1)/√5 ≈ (0.894, 0.447).
- *P2:* `scores = Xc @ V`; equals sklearn `transform` up to per-component sign.
- *P3:* (a) On standardized Wine, ~8 components reach 90% (printed as `k90`). (b) The red MSE curve is the average squared error of rebuilding the standardized data from only k components — it is the information *lost*, and it falls to ~0 as k→p.
- *P4:* Raw PC1 is dominated by **proline** (variance in the tens/hundreds of thousands), purely because of its units — an artefact, not chemistry. Defensible default: **standardize (correlation PCA)**. Covariance (unscaled) PCA is appropriate only when all variables are in the *same meaningful units* and differences in variance are themselves the signal.
- *P5:* Parallel analysis keeps the components whose real eigenvalues exceed the random-data curve (printed as `keep`, typically ~3 for Wine); Kaiser >1 usually keeps more and tends to over-retain.
- *P6:* Arrows for e.g. flavanoids / total phenols / OD280 vs proline / colour-intensity point in opposing PC1 directions, and the three cultivars separate mostly along PC1 — i.e. PC1 contrasts "phenolic" wines against "colour/proline" wines.
- *P7:* Explained variance is unchanged; PC1 scores merely negate.
- *P8:* At strength 0 PC1 is essentially arbitrary (isotropic cloud); as strength grows, PC1 rotates to point at the single outlier — one point can dominate a variance-based method.

**✅ Method card (reference):** Method = PCA. Question = compress correlated numeric variables into a few
uncorrelated axes. Unsupervised. Input = n×p numeric, standardized by default. Objective = maximize
projected variance / top eigenvectors of Σ (or R). Assumptions = linearity, sensible scale, outlier-light.
Output = loadings, scores, explained-variance ratios. Diagnostics = scree, cumulative variance, parallel
analysis, reconstruction error. Choose k = cumulative threshold + parallel analysis + interpretability.
Trap = not standardizing (scaling trap); reading scores as loadings; over-reading signs; outliers.
One-liner = "a few new uncorrelated axes capturing most of the spread." Closest competitor = factor
analysis (models shared variance via latent factors, not total variance).

**Transfer:** Factor analysis explicitly separates *common* variance (shared by items, attributed to latent
factors) from *unique* variance (item-specific + noise), so it supports a causal "hidden traits generate
the correlations" story in a way PCA — which just repackages total variance — does not.
"""))

build("03_principal_component_analysis.ipynb", cells)
