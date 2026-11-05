"""Generate 07_integrated_case_study.ipynb."""

from nbbuild import (SETUP_CELL, answer_key_banner, build, close, code, header,
                     md, predict, retrieve)

cells = []

cells.append(header(
    title="07 · Integrated Case Study — Palmer Penguins",
    est_time="2–3 hours (a full mini-analysis)",
    objectives=[
        "Run the entire workflow once, end to end, on a single real dataset: question → data → assumptions → fit → diagnostics → interpretation → report.",
        "Choose a method from a bounded menu and *justify* it before fitting — the most heavily graded skill.",
        "Check the assumptions your chosen method actually requires, and report honestly including limitations.",
    ],
    prereq="Notebooks 01–06. This is where they come together.",
    mvp="Pick **one** research question from the menu, do the 8 workflow steps for it at a basic level, and write the "
        "5-sentence report. A complete simple analysis beats a half-finished complicated one.",
    extra="\n\n**This notebook does NOT tell you the method.** That is deliberate — on the exam, choosing and "
          "defending the method is the point. You get a bounded menu and a dataset; the justification is yours.",
))

cells.append(code(SETUP_CELL))

# ---------------------------------------------------------------- Retrieve
cells += retrieve([
    "From 02: name one diagnostic you would check before trusting a regression's coefficients.",
    "From 03/04: state the one-line difference between PCA and factor analysis.",
    "From 05: why must you standardize before a distance-based clustering of penguins?",
    "From 06: what does cross-validation protect you from, and what is data leakage?",
])

# ---------------------------------------------------------------- The brief + data
cells.append(md(r"""
## 📋 The brief

You are handed the **Palmer Penguins** dataset: body measurements for penguins of three species
(Adelie, Chinstrap, Gentoo) on three Antarctic islands. A colleague asks you to "find something
statistically defensible and write it up." Your job is to turn that vague ask into a real analysis.

First, **inspect the data** (always step 1). Run the cells below before deciding anything.
"""))
cells.append(code(r"""
pen = data.load_penguins_df()
feat = data.PENGUIN_FEATURES
print("shape:", pen.shape)
print("\ncolumns & types:"); print(pen.dtypes.to_string())
print("\nspecies counts:"); print(pen["species"].value_counts().to_string())
pen.head()
"""))
cells.append(code(r"""
print("numeric summary:")
print(pen[feat].describe().round(1).to_string())
print("\nNote the scales: body_mass_g is in thousands, the *_mm features are tens. (Scaling will matter.)")
"""))
cells.append(code(r"""
import seaborn as sns
g = sns.pairplot(pen[feat + ["species"]], hue="species",
                 palette=[P.group_color(i) for i in range(3)], height=1.6, corner=True)
g.figure.suptitle("Q: which measurements separate species? where are the correlations?", y=1.02)
plt.show()
"""))

# ---------------------------------------------------------------- Choose (predict)
cells.append(md(r"""
## 🔮 Choose your question and method (bounded — pick ONE)

Each question below naturally calls for a different method. **Pick one**, then in the answer cell name the
method and justify it in 2–3 sentences (goal-verb, supervised/unsupervised, why it fits *these* data).

| # | Research question | Hint (do not skip the justification) |
|---|---|---|
| **A** | "Can we **predict** a penguin's body mass from its other measurements, and which matter most?" | numeric outcome → … |
| **B** | "Can we **summarize** the 4 correlated measurements into 1–2 axes for a dashboard?" | reduce, no target → … |
| **C** | "Do penguins fall into **natural groups**, and do those groups match species?" | discover groups, unsupervised → … |
| **D** | "Given **known species**, which measurements best separate them and can we classify a new bird?" | known labels → … |

> Only one deliberate difficulty at a time: start with the question you feel most confident about.
"""))
cells.append(md("**YOUR CHOICE & JUSTIFICATION:**\n\nI choose question ___ . The method is ___ because ...\n"))

# ---------------------------------------------------------------- Work the case
cells.append(md(r"""
## 🛠️ Work the case — the 8 steps

Fill each step for **your** chosen question. The markdown cells are where the marks are; the code cell is a
scratchpad to actually fit your method (it is self-contained — leaving it blank will not break the notebook).
"""))

cells.append(md(r"""
**Step 1 — Research question.** Restate your question precisely (what is the outcome / what is being grouped / what is being reduced?).

**YOUR ANSWER:**
"""))
cells.append(md(r"""
**Step 2 — Inspect & preprocess.** What did the inspection above tell you? State your decisions: missing
values (we dropped NA rows in the loader), scaling (needed? why?), which columns you will actually use, and
any encoding (e.g. species as a label vs. a predictor).

**YOUR ANSWER:**
"""))
cells.append(md(r"""
**Step 3 — Choose & justify the method.** (Copy/expand your choice above.) Name the method, its goal-verb,
and one method you considered and rejected — and why.

**YOUR ANSWER:**
"""))
cells.append(md(r"""
**Step 4 — Assumptions to check.** List the assumptions *your* method needs (e.g. regression: linearity,
no severe multicollinearity, well-behaved residuals; LDA: roughly equal group covariances; clustering:
standardized features + a sensible distance). You will check them in Step 6.

**YOUR ANSWER:**
"""))
cells.append(code(r"""
# Scratchpad — fit YOUR chosen method here. Self-contained: blank is fine.
# Reusable pieces you already know:
#   from sklearn.preprocessing import StandardScaler
#   import statsmodels.api as sm                              # regression coefficient table
#   from sklearn.decomposition import PCA                     # reduce
#   from sklearn.cluster import KMeans                        # group  (+ silhouette_score)
#   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # discriminate
#   from sklearn.model_selection import cross_val_score, train_test_split # honest evaluation
X = pen[feat].values
# ... your analysis ...
print("Scratchpad ready — X has shape", X.shape)
"""))
cells.append(md(r"""
**Step 5 — Fit.** Briefly: what did you run, and with what key settings (k, n_components, standardized?, priors?)?

**YOUR ANSWER:**
"""))
cells.append(md(r"""
**Step 6 — Diagnostics / validation.** Report the numbers that judge your fit (R²/RMSE, or explained
variance, or silhouette, or cross-validated accuracy + confusion matrix) **and** whether the Step-4
assumptions held.

**YOUR ANSWER:**
"""))
cells.append(md(r"""
**Step 7 — Interpret.** Translate the output into sentences about penguins (not about coefficients in the
abstract). Which variables mattered, and what does the result *mean*?

**YOUR ANSWER:**
"""))
cells.append(md(r"""
**Step 8 — Limitations.** Name two honest limitations (e.g. observational data, three species only, clusters
≠ species ground truth, possible collinearity between flipper length and body mass).

**YOUR ANSWER:**
"""))

# ---------------------------------------------------------------- Close
cells.append(close())
cells.append(md(r"""
**Final report (5–8 sentences).** Combine Steps 1–8 into a short report a non-statistician could read:
question, method + why, what you found, how good the fit is, and limitations. This is exactly the exam
deliverable.

**YOUR REPORT:**
"""))
cells.append(md(r"""
**Self-assessment (rate yourself 1–3 on each, the exam rubric):**
correctness · method justification · interpretation · diagnostics · communication.

**YOUR SELF-SCORES:**
"""))

# ---------------------------------------------------------------- Answer key (2 worked analyses)
cells.append(answer_key_banner())
cells.append(md(r"""
Below are **two** fully worked, defensible analyses — one **supervised regression** (question A) and one
**supervised discriminant** (question D). Yours may differ; compare the *shape* of the reasoning, not the
exact wording. (Questions B and C are left for you, but the methods are PCA from notebook 03 and k-means
from notebook 05 applied to these same `feat` columns.)
"""))

cells.append(md("**Worked analysis A — predict body mass (multiple regression).**"))
cells.append(code(r"""
import statsmodels.api as sm
pred_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]
Xa = sm.add_constant(pen[pred_cols].values)
ya = pen["body_mass_g"].values
mA = sm.OLS(ya, Xa).fit()
print(mA.summary().tables[1])
print(f"R2={mA.rsquared:.3f}  adjR2={mA.rsquared_adj:.3f}  RMSE={np.sqrt(np.mean(mA.resid**2)):.0f} g")

# standardized coefficients for fair importance comparison
from sklearn.preprocessing import StandardScaler
Xz = StandardScaler().fit_transform(pen[pred_cols].values)
mz = sm.OLS((ya-ya.mean())/ya.std(), sm.add_constant(Xz)).fit()
print("\nstandardized coefficients:",
      dict(zip(pred_cols, np.round(mz.params[1:], 2))))
"""))
cells.append(md(r"""
*Report A (model answer).* We modelled body mass on three measurements (OLS). The fit is strong
(R² ≈ 0.76, RMSE ≈ 390 g). **Flipper length** is by far the dominant predictor (largest standardized
coefficient, highly significant); bill dimensions add less. Residuals are reasonably well-behaved, though
species is a lurking variable — Gentoos are both larger and heavier, so part of the "flipper" effect is
really "species". *Limitations:* observational data (no causal claim); adding `species` as a predictor would
change the coefficients (a Simpson's-paradox risk); three species only.
"""))

cells.append(md("**Worked analysis D — classify species (LDA, cross-validated).**"))
cells.append(code(r"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

Xd = pen[feat].values
yd = pen["species"].values
pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
cv = cross_val_score(pipe, Xd, yd, cv=5)
print(f"5-fold CV accuracy: {cv.mean():.3f} ± {cv.std():.3f}")

Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.3, random_state=7, stratify=yd)
pipe.fit(Xtr, ytr)
plots.confusion(yte, pipe.predict(Xte), labels=["Adelie", "Chinstrap", "Gentoo"],
                question="Q: which species get confused?")
plt.show()

lda = pipe.named_steps["lineardiscriminantanalysis"]
contrib = pd.Series(np.abs(lda.coef_).sum(0), index=feat).sort_values(ascending=False)
print("variable contribution to separation:"); print(contrib.round(2).to_string())
"""))
cells.append(md(r"""
*Report D (model answer).* To classify species we used LDA on the four standardized measurements, validated
by 5-fold cross-validation (accuracy ≈ 0.98). The confusion matrix shows the only meaningful confusion is
Adelie↔Chinstrap (similar size); Gentoo is essentially always separated. **Bill depth and body mass**
contribute most to the separation (largest standardized discriminant weights). LDA's equal-covariance
assumption is approximately reasonable here; QDA or
logistic would be sensible robustness checks. *Limitations:* three species from one archipelago — the rule
may not transfer; class sizes are unequal; measurement error is ignored.

**🔁 Retrieve answers:** (1) residuals-vs-fitted / VIF / Cook's distance. (2) PCA repackages total variance;
FA models shared variance via latent factors. (3) body mass (grams) would otherwise dominate the distance.
(4) CV protects against over-optimistic, overfit performance estimates; leakage = letting test information
into training (e.g. scaling/feature-selecting on all data before splitting).
"""))

build("07_integrated_case_study.ipynb", cells)
