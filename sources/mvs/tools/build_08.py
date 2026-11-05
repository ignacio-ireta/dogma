"""Generate 08_exam_practical_simulation.ipynb (NO solutions inline)."""

from nbbuild import SETUP_CELL, build, code, md

cells = []

cells.append(md(r"""
# 08 · Practical Exam Simulation  ⏱️

**Time budget:** 2.5 hours. **Closed-book-ish:** notebooks 00–07 allowed as reference, but work the
problems yourself. **Do NOT open** `08_exam_practical_simulation_solutions.ipynb` until you have attempted
every section.

**How it is graded (rubric, 5 axes):** correctness · method justification · interpretation · diagnostics ·
communication. The *choose-and-defend* answers carry the most weight.

**Sections**
1. Method selection (Recognize)            — ~20 min
2. Matrix / covariance computation (Compute) — ~25 min
3. Regression output interpretation        — ~25 min
4. PCA / factor output interpretation      — ~25 min
5. Clustering or discrimination practical  — ~30 min
6. Concise report (Choose-and-defend)      — ~25 min

Run the setup cell, then go top to bottom. Write answers in the **YOUR ANSWER** cells.
"""))

cells.append(code(SETUP_CELL))

# ----------------------------------------------------------- Section 1
cells.append(md(r"""
## Section 1 — Method selection  *(Recognize)*

For each scenario, name **the** method, say supervised/unsupervised, and justify in one sentence.

1. A winery has 13 chemical measurements on 200 bottles (no labels) and wants 2 indices for a quality dashboard.
2. A hospital has 6 blood markers and a known diagnosis (disease / no disease) and wants to classify new patients.
3. A sociologist has 30 survey items and believes 5 latent attitudes generate the responses.
4. An economist wants to explain household spending (in pesos) from 8 numeric indicators.
5. A retailer has 5 behavioural metrics per customer (no labels) and wants to find customer segments.
"""))
cells.append(md("**YOUR ANSWERS (Section 1):**\n\n1. \n2. \n3. \n4. \n5. "))

# ----------------------------------------------------------- Section 2
cells.append(md(r"""
## Section 2 — Matrix / covariance computation  *(Compute)*

Given the covariance matrix
$$ S=\begin{pmatrix} 6 & 2 \\ 2 & 3 \end{pmatrix}. $$

Compute, by hand then verify in code: (a) its eigenvalues, (b) the proportion of total variance captured by
the **first** principal component, and (c) the Mahalanobis distance of the point $x=(3,1)^\top$ from the
origin (mean 0). The self-check confirms your numbers without revealing them.
"""))
cells.append(code(r"""
S = np.array([[6.0, 2.0], [2.0, 3.0]])
x = np.array([3.0, 1.0])

# TODO: fill these three answers (compute by hand first, then you may verify with numpy).
# HINT (a): eigenvalues solve det(S - λI)=0  ->  λ² - 7λ + 14? recompute the trace(=9) and det(=14).
# HINT (b): pc1 share = largest eigenvalue / sum of eigenvalues.
# HINT (c): d = sqrt(xᵀ S⁻¹ x).
eigvals_desc = None   # <- length-2 array, largest first
pc1_share = None      # <- a number in [0,1]
maha = None           # <- a number

def _v():
    ev = np.sort(np.linalg.eigvalsh(S))[::-1]
    assert_close(np.sort(eigvals_desc)[::-1], ev, tol=1e-6, name="eigenvalues")
    assert_close(pc1_share, ev[0]/ev.sum(), tol=1e-6, name="PC1 variance share")
    assert_close(maha, float(np.sqrt(x @ np.linalg.inv(S) @ x)), tol=1e-6, name="Mahalanobis distance")
    print("All three numeric answers check out.")
selfcheck(eigvals_desc is not None and pc1_share is not None and maha is not None, _v)
"""))

# ----------------------------------------------------------- Section 3
cells.append(md(r"""
## Section 3 — Regression output interpretation

The table below is a real OLS fit (a small constructed dataset). **Read it** and answer underneath — do not
refit anything.
"""))
cells.append(code(r"""
import statsmodels.api as sm
n = 120
x1 = RNG.normal(size=n)
x2 = x1 + RNG.normal(scale=0.05, size=n)      # nearly a copy of x1
x3 = RNG.normal(size=n)
y = 3 + 2.0*x1 - 1.5*x3 + RNG.normal(scale=1.0, size=n)
Xs = sm.add_constant(np.column_stack([x1, x2, x3]))
m = sm.OLS(y, Xs).fit()
print(m.summary(xname=["const", "x1", "x2", "x3"]).tables[1])
print(f"\nR2 = {m.rsquared:.3f}   adjusted R2 = {m.rsquared_adj:.3f}")

from statsmodels.stats.outliers_influence import variance_inflation_factor
print("VIFs:", {nm: round(variance_inflation_factor(Xs, i), 1)
                for i, nm in enumerate(["const", "x1", "x2", "x3"])})
"""))
cells.append(md(r"""
**YOUR ANSWERS (Section 3):**
1. Which predictors are statistically significant at 0.05, and which are not?
2. `x1` and `x2` have huge standard errors and unstable signs despite the model fitting well. What is going on, and which diagnostic number proves it?
3. Is `x3`'s effect on `y` positive or negative, and roughly how big per one-unit change?
4. One sentence of advice to fix the model.
"""))

# ----------------------------------------------------------- Section 4
cells.append(md(r"""
## Section 4 — PCA / factor output interpretation

Below is a correlation-PCA of the Wine data (standardized). Read the explained variance and the PC1
loadings, then answer.
"""))
cells.append(code(r"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
wine = data.load_wine_df(); feat = list(wine.columns[:13])
Z = StandardScaler().fit_transform(wine[feat].values)
pca = PCA(random_state=0).fit(Z)

evr = pca.explained_variance_ratio_
print("explained variance ratio (first 6):", np.round(evr[:6], 3))
print("cumulative                        :", np.round(np.cumsum(evr)[:6], 3))
pc1 = pd.Series(pca.components_[0], index=feat).sort_values()
print("\nPC1 loadings (most negative to most positive):")
print(pc1.round(2).to_string())
"""))
cells.append(md(r"""
**YOUR ANSWERS (Section 4):**
1. How many components would you keep to exceed 70% cumulative variance? Show the number.
2. From the PC1 loadings, describe in words the *contrast* PC1 represents (which variables are high vs low).
3. Why was the data standardized before this PCA, and what would change if it were not?
4. A colleague says "PC1 is the most important *factor*". Correct or amend that statement in one sentence.
"""))

# ----------------------------------------------------------- Section 5
cells.append(md(r"""
## Section 5 — Clustering or discrimination practical  *(Compute + interpret)*

Choose **one** of 5A / 5B. Complete the small code task (self-contained) and interpret.
"""))
cells.append(md(r"""
**5A — Discrimination.** Classify Wine cultivars with LDA, cross-validated. Fill the two blanks.
"""))
cells.append(code(r"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

Xw = wine[feat].values; yw = wine["target"].values
# TODO: build a pipeline (standardize -> LDA) and get the 5-fold CV mean accuracy.
# HINT: pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()); cv = cross_val_score(pipe, Xw, yw, cv=5)
pipe = None
cv_acc = None     # <- the mean CV accuracy
def _v():
    assert cv_acc is not None and 0.9 <= cv_acc <= 1.0, "expected a high CV accuracy (~0.98)"
    print(f"CV accuracy = {cv_acc:.3f}")
selfcheck(cv_acc is not None, _v)
"""))
cells.append(md(r"""
**5B — Clustering.** Cluster standardized Wine with k-means (k=3) and compare to the true cultivars with the
adjusted Rand index. Fill the blank.
"""))
cells.append(code(r"""
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

Zw = StandardScaler().fit_transform(wine[feat].values)
# TODO: fit KMeans(k=3, n_init=10, random_state=7) and compute adjusted Rand vs wine['target'].
labels = None
ari = None
def _v():
    assert ari is not None, "compute the adjusted Rand index"
    print(f"adjusted Rand index vs true cultivars = {ari:.2f}")
selfcheck(ari is not None, _v)
"""))
cells.append(md(r"""
**YOUR ANSWERS (Section 5):** For the option you chose — (a) report the headline number, (b) interpret it in
one sentence, (c) name one assumption or limitation of the method *as applied here*.
"""))

# ----------------------------------------------------------- Section 6
cells.append(md(r"""
## Section 6 — Concise report  *(Choose-and-defend — highest weight)*

Pick **one** analysis from Sections 3–5 and write a **5–7 sentence report** for a non-statistician manager:
the question, the method and why it fits, the main result with one number, one diagnostic/validation point,
and one limitation. Plain language; no LaTeX required.

**YOUR REPORT:**
"""))

cells.append(md(r"""
---
⏱️ **Time's up?** Check you have: 5 method choices, 3 computed numbers (Section 2), 4 regression answers,
4 PCA answers, one practical (5A or 5B) with interpretation, and a report. Then — and only then — open
**`08_exam_practical_simulation_solutions.ipynb`** and grade yourself against the rubric.
"""))

build("08_exam_practical_simulation.ipynb", cells)
