"""Generate 08_exam_practical_simulation_solutions.ipynb."""

from nbbuild import SETUP_CELL, build, code, md

cells = []

cells.append(md(r"""
# 08 · Practical Exam — Solutions & Rubric

Worked answers for `08_exam_practical_simulation.ipynb`. Use this **after** your attempt. For each section,
grade yourself 0–2 on each rubric axis (correctness, justification, interpretation, diagnostics,
communication) — a strong pass is ~80% of the available points with the *choose-and-defend* items done well.
"""))

cells.append(code(SETUP_CELL))

# ----------------------------------------------------------- Section 1
cells.append(md(r"""
## Section 1 — Method selection

1. **PCA** (unsupervised) — reduce 13 correlated chemical variables to 2 dashboard indices.
2. **Discriminant analysis / logistic** (supervised) — known disease labels → classify new patients.
3. **Factor analysis** (unsupervised) — latent attitudes generating item responses (a causal/latent claim, not mere compression).
4. **Multiple regression** (supervised) — numeric outcome (spending) explained by numeric predictors.
5. **Cluster analysis** (unsupervised) — discover customer segments with no predefined labels.

*Rubric:* full marks require the method **and** a one-line justification using the goal-verb /
supervised–unsupervised distinction.
"""))

# ----------------------------------------------------------- Section 2
cells.append(md("## Section 2 — Matrix / covariance computation"))
cells.append(code(r"""
S = np.array([[6.0, 2.0], [2.0, 3.0]]); x = np.array([3.0, 1.0])
ev = np.sort(np.linalg.eigvalsh(S))[::-1]
print("(a) eigenvalues: trace=9, det=14  ->  λ = (9 ± sqrt(81-56))/2 = (9 ± 5)/2 = 7 and 2")
print("    numpy:", np.round(ev, 4))
print(f"(b) PC1 variance share = 7/(7+2) = {ev[0]/ev.sum():.4f}  (~0.778)")
Sinv = np.linalg.inv(S)   # = (1/14) * [[3, -2], [-2, 6]]
d2 = float(x @ Sinv @ x)
print(f"(c) S⁻¹·x = {np.round(Sinv @ x, 4)};  xᵀS⁻¹x = {d2:.4f};  d = sqrt = {np.sqrt(d2):.4f}")
"""))
cells.append(md("Eigenvalues **7 and 2**; PC1 share **= 7/9 ≈ 0.778**; Mahalanobis distance **= √1.5 ≈ 1.22**."))

# ----------------------------------------------------------- Section 3
cells.append(md("## Section 3 — Regression output interpretation"))
cells.append(code(r"""
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
n = 120
x1 = RNG.normal(size=n); x2 = x1 + RNG.normal(scale=0.05, size=n); x3 = RNG.normal(size=n)
y = 3 + 2.0*x1 - 1.5*x3 + RNG.normal(scale=1.0, size=n)
Xs = sm.add_constant(np.column_stack([x1, x2, x3]))
m = sm.OLS(y, Xs).fit()
print(m.summary(xname=["const", "x1", "x2", "x3"]).tables[1])
print("VIFs:", {nm: round(variance_inflation_factor(Xs, i), 1) for i, nm in enumerate(["const","x1","x2","x3"])})
"""))
cells.append(md(r"""
1. **const** and **x3** are clearly significant; **x1** and **x2** individually are *not* (large p-values)
   even though together they carry the real signal — a hallmark of collinearity.
2. **x2 is essentially a copy of x1** (built as x1 + tiny noise). The model cannot separate their effects, so
   it splits the coefficient between them with huge, unstable, opposite values and inflated standard errors.
   The proof is the **VIF** for x1 and x2 (in the hundreds/thousands).
3. **x3 is negative**, about **−1.5** per unit (matches the data-generating coefficient).
4. **Fix:** drop one of x1/x2 (or combine them / use regularization or PCA) — they are redundant.
"""))

# ----------------------------------------------------------- Section 4
cells.append(md("## Section 4 — PCA / factor output interpretation"))
cells.append(code(r"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
wine = data.load_wine_df(); feat = list(wine.columns[:13])
Z = StandardScaler().fit_transform(wine[feat].values)
pca = PCA(random_state=0).fit(Z)
cum = np.cumsum(pca.explained_variance_ratio_)
print("cumulative variance:", np.round(cum[:6], 3))
print("components to exceed 70%:", int(np.searchsorted(cum, 0.70) + 1))
print("\nPC1 loadings:")
print(pd.Series(pca.components_[0], index=feat).sort_values().round(2).to_string())
"""))
cells.append(md(r"""
1. **4 components** exceed 70% cumulative variance (printed above).
2. PC1 **contrasts phenolic/flavonoid richness against colour/alcohol/proline**: variables like flavanoids,
   total_phenols, OD280 load strongly one way, while colour_intensity, alcohol, proline load the other. PC1
   orders wines from "phenol-rich, lighter" to "deep-coloured, high-proline".
3. Standardizing puts all 13 variables on a common scale; **without it, proline (variance in the tens of
   thousands) would dominate PC1**, an artefact of units rather than chemistry.
4. **Amend:** PC1 is the highest-variance *component*, not a *factor* — PCA repackages variance and makes no
   latent-cause claim; calling it a "factor" imports factor-analysis meaning it doesn't have.
"""))

# ----------------------------------------------------------- Section 5
cells.append(md("## Section 5 — Clustering or discrimination practical"))
cells.append(code(r"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import make_pipeline

Xw = wine[feat].values; yw = wine["target"].values
pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
print(f"5A  LDA 5-fold CV accuracy = {cross_val_score(pipe, Xw, yw, cv=5).mean():.3f}")

Zw = StandardScaler().fit_transform(Xw)
lab = KMeans(n_clusters=3, n_init=10, random_state=7).fit_predict(Zw)
print(f"5B  k-means adjusted Rand vs cultivars = {adjusted_rand_score(yw, lab):.2f}")
"""))
cells.append(md(r"""
**5A (LDA).** CV accuracy ≈ **0.97**. The three cultivars are almost perfectly separable from the 13
standardized chemical variables. *Limitation:* LDA assumes equal group covariances and ~normal predictors;
with 13 variables and 178 wines the pooled covariance is estimable but the assumption should be checked
(QDA/logistic as robustness).

**5B (k-means).** Adjusted Rand ≈ **0.90** — clusters recovered on *standardized* data align well with the
true cultivars. *Limitation:* clustering is unsupervised, so the match to cultivars is luck of good
separation, not a guarantee; on raw (unscaled) data the agreement collapses because proline dominates distance.
"""))

# ----------------------------------------------------------- Section 6 + rubric
cells.append(md(r"""
## Section 6 — Concise report (model answer)

> *Example (Section 5A).* We asked whether a wine's cultivar can be identified from its chemistry. Because the
> cultivars were **known**, this is a supervised discrimination problem, so we used **linear discriminant
> analysis** on the 13 standardized chemical measurements. Validated by 5-fold cross-validation, the classifier
> reached **≈97% accuracy**, with the only minor confusion between the two chemically similar cultivars.
> Flavanoids, proline and colour intensity contributed most to the separation. LDA's equal-covariance
> assumption looked approximately reasonable, but we would confirm with QDA. The main limitation is that these
> wines come from one region and three cultivars, so the rule may not generalize.

## Rubric (score each section 0–2 per axis)

| Axis | 0 | 1 | 2 |
|---|---|---|---|
| **Correctness** | wrong method/number | minor error | correct |
| **Justification** | none | asserted | tied to goal-verb / assumptions |
| **Interpretation** | restates output | partial meaning | clear real-world meaning |
| **Diagnostics** | none | mentioned | checked + acted on |
| **Communication** | unclear | wordy/jargon | concise, plain, honest |

**Pass guide:** ~80% overall, *with* the Section-6 report and the choose-and-defend items done well, indicates
you are ready. Re-study any unit where you lost correctness points.
"""))

build("08_exam_practical_simulation_solutions.ipynb", cells)
