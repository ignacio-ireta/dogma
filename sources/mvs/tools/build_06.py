"""Generate 06_discriminant_analysis_and_logistic_bridge.ipynb."""

from nbbuild import (METHOD_CARD_BLANK, SETUP_CELL, answer_key_banner, build,
                     close, code, header, learn, md, practice, predict, retrieve)

cells = []

cells.append(header(
    title="06 · Discriminant Analysis and the Logistic Bridge",
    est_time="110–140 min",
    objectives=[
        "Build a two-group LDA discriminant function from pooled covariance and read variable contributions.",
        "Compare LDA, QDA, and logistic boundaries, and connect them to their assumptions (equal vs unequal covariance).",
        "Evaluate a classifier with a confusion matrix, sensitivity/specificity/precision/recall, and honest cross-validation.",
        "Explain data leakage and why discrimination (supervised) differs from clustering (unsupervised).",
    ],
    prereq="01 (covariance geometry, Mahalanobis, pooled covariance/inverse). 02 (logistic as regression's cousin). 05 (clusters ≠ known groups).",
    mvp="Do 🔁 Retrieve, the **from-scratch LDA vs sklearn** cell, and the **confusion-matrix metrics** cell. "
        "Then read the LDA-vs-QDA assumption comparison.",
    extra="\n\nUnit 5. The exam names *logistic discrimination* explicitly, so we treat it as a first-class method "
          "alongside LDA — not an afterthought. QDA is enrichment: useful, but not the central subtopic.",
))

cells.append(code(SETUP_CELL))

# ---------------------------------------------------------------- Retrieve
cells += retrieve([
    "From 01: what does the pooled covariance matrix estimate, and why must it be invertible for LDA?",
    "From 05: state the one-line difference between clustering and discriminant analysis.",
    "From 02: logistic regression replaces multiple regression when the outcome is what kind of variable?",
    "From 01: Mahalanobis distance — LDA classifies a point to the nearest group centre in *which* distance?",
])

# ---------------------------------------------------------------- Predict
cells += predict([
    "LDA assumes the groups share the **same** covariance matrix. Predict the *shape* of the boundary that assumption produces (straight or curved?).",
    "QDA lets each group have its **own** covariance. Predict how its boundary differs from LDA's.",
    "A classifier is 97% accurate, but the disease it detects affects 3% of people. Predict why accuracy could be a misleading score here.",
    "You standardize the whole dataset, then split into train/test and cross-validate. Predict whether anything is subtly wrong.",
])

# ---------------------------------------------------------------- Learn
cells.append(learn("Learn the system — discriminant analysis as a system"))

cells.append(md(r"""
**Research question.** "Given observations from **known** groups, what combination of variables separates
them, and how do I classify a new case?" **Supervised** (contrast: clustering discovers groups).

**Input shape.** $X$: $n\times p$ numeric predictors; $y$: known class label. Output = a classification rule
+ discriminant axes.

**LDA — the model.** Assume each group is multivariate normal with **its own mean** $\mu_g$ but a **shared
covariance** $\Sigma$ (estimated by the *pooled* covariance $S_p$). Then the boundary between two groups is
**linear**, and the discriminant direction for two groups is
$$ w = S_p^{-1}(\mu_1-\mu_2). $$
You classify $x$ to the group whose centre is nearest in **Mahalanobis** distance using $S_p$ (equivalently,
by the sign of $w^\top x$ minus a threshold set by the class midpoint and priors).

**Contribution by variable.** On standardized data, the magnitude of each discriminant coefficient says how
much that variable contributes to separation. **Multiclass LDA** finds up to $\min(p,\,g-1)$ discriminant
axes (so 3 groups → at most 2 axes — handy for plotting).

**QDA (enrichment).** Drop the equal-covariance assumption: each group gets its own $\Sigma_g$. Boundary
becomes **quadratic** (curved). More flexible, more parameters, needs more data.

**Logistic discrimination (syllabus).** Model the *probability* of a class directly:
$\log\frac{P(y{=}1)}{P(y{=}0)} = \beta_0+\beta^\top x$. No normality assumption on $x$; coefficients are
log-odds effects. Often the robust default for two-group problems.

**Evaluation (read these every time).** Build the **confusion matrix**; from it:

| Metric | Formula | Question it answers |
|---|---|---|
| Sensitivity / recall (TPR) | TP/(TP+FN) | of true positives, how many caught? |
| Specificity (TNR) | TN/(TN+FP) | of true negatives, how many cleared? |
| Precision | TP/(TP+FP) | of predicted positives, how many right? |
| Accuracy | (TP+TN)/all | overall — **misleading under class imbalance** |

Always estimate these by **cross-validation**, never on the training data, and beware **data leakage**
(letting test information into training — e.g. scaling or feature-selecting on the full dataset before splitting).

**Common traps.** (1) Using LDA when covariances are very unequal (→ use QDA/logistic). (2) Reporting
accuracy on imbalanced classes. (3) **Leakage** inflating cross-validated scores. (4) Reading a raw (unstandardized)
discriminant coefficient as "importance".

**Neighbours.** Clustering (05) = unsupervised cousin. Logistic ↔ multiple regression (02) for categorical outcomes.
PCA (03) can pre-reduce predictors before discrimination.
"""))

# ---------------------------------------------------------------- Practice
cells.append(practice())

cells.append(md(r"""
**P1 (Compute · two-group LDA from scratch).** Build the pooled covariance, the discriminant direction
$w=S_p^{-1}(\mu_1-\mu_2)$, and classify by the midpoint threshold. Confirm your labels match
`sklearn`'s LDA (equal priors) on data that satisfies the equal-covariance assumption.
"""))
cells.append(code(r"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

Sigma = np.array([[1.0, 0.4], [0.4, 1.0]])
g0 = RNG.multivariate_normal([0, 0], Sigma, 150)
g1 = RNG.multivariate_normal([2.4, 1.8], Sigma, 150)     # same covariance, different mean
X = np.vstack([g0, g1]); ylab = np.r_[np.zeros(150), np.ones(150)].astype(int)

mu0, mu1 = g0.mean(0), g1.mean(0)
S0, S1 = np.cov(g0, rowvar=False), np.cov(g1, rowvar=False)
n0, n1 = len(g0), len(g1)

# TODO: pooled covariance, discriminant direction w, and midpoint threshold c.
# HINT 1 (concept): pool the two within-group covariances weighted by (n-1).
# HINT 2 (formula): Sp = ((n0-1)S0 + (n1-1)S1)/(n0+n1-2); w = Sp^{-1}(mu1-mu0); c = w·(mu0+mu1)/2
# HINT 3 (code): Sp = ((n0-1)*S0 + (n1-1)*S1)/(n0+n1-2); w = np.linalg.solve(Sp, mu1-mu0); c = w @ (mu0+mu1)/2
Sp = None; w = None; c = None   # <- replace
def _v():
    pred_hand = (X @ w > c).astype(int)
    sk = LinearDiscriminantAnalysis(priors=[0.5, 0.5]).fit(X, ylab)
    agree = np.mean(pred_hand == sk.predict(X))
    assert agree > 0.98, f"only {agree:.0%} agreement — check Sp, w, or threshold"
    print(f"from-scratch LDA agrees with sklearn on {agree:.0%} of points.")
selfcheck(Sp is not None and w is not None and c is not None, _v)
"""))

cells.append(md(r"""
**P2 (Interpret · project onto the discriminant axis).** LDA's whole point is a 1-D axis (for two groups)
that separates them. Plot the two groups' scores along it.
"""))
cells.append(code(r"""
lda = LinearDiscriminantAnalysis().fit(X, ylab)
z = lda.transform(X).ravel()        # discriminant scores
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.hist(z[ylab == 0], bins=20, alpha=0.6, color=P.group_color(0), label="group 0")
ax.hist(z[ylab == 1], bins=20, alpha=0.6, color=P.group_color(1), label="group 1")
ax.set_xlabel("LDA discriminant score  w·x"); ax.set_ylabel("count")
ax.set_title("Q: how well does ONE discriminant axis separate the groups?"); ax.legend(fontsize=8)
plt.show()
"""))

cells.append(md(r"""
**P3 (Choose-and-defend · LDA vs QDA vs logistic boundaries — the assumption comparison / failure case).**
We make groups with **unequal** covariance, violating LDA's core assumption, and compare all three boundaries
and their cross-validated accuracy.
"""))
cells.append(code(r"""
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# group A: tight & round; group B: wide & tilted -> covariances differ a lot
A = RNG.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], 250)
B = RNG.multivariate_normal([1.5, 1.5], [[3.0, 2.2], [2.2, 3.0]], 250)
Xu = np.vstack([A, B]); yu = np.r_[np.zeros(250), np.ones(250)].astype(int)

models = {"LDA (assumes equal cov)": LinearDiscriminantAnalysis(),
          "QDA (own cov per group)": QuadraticDiscriminantAnalysis(),
          "Logistic": LogisticRegression()}
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (name, mdl) in zip(axes, models.items()):
    mdl.fit(Xu, yu)
    acc = cross_val_score(mdl, Xu, yu, cv=5).mean()
    plots.decision_boundary(mdl, Xu, yu, ax=ax, question=f"{name}\nCV acc={acc:.2f}")
plt.tight_layout(); plt.show()
"""))
cells.append(md(r"""
**YOUR ANSWER (P3):** (a) Which model's boundary is forced to be straight, and why does that hurt here?
(b) Which handles the unequal covariance best, and at what cost? (c) When *would* LDA be the better choice?
"""))

cells.append(md(r"""
**P4 (Interpret · multiclass LDA on Wine + contribution by variable).** Three cultivars → LDA finds 2
discriminant axes. Plot them, and rank variables by their standardized contribution to separation.
"""))
cells.append(code(r"""
from sklearn.preprocessing import StandardScaler
wine = data.load_wine_df(); feat = list(wine.columns[:13])
Xw = StandardScaler().fit_transform(wine[feat].values)
yw = wine["target"].values

lda3 = LinearDiscriminantAnalysis(n_components=2).fit(Xw, yw)
Zw = lda3.transform(Xw)
fig, ax = plt.subplots(figsize=(5.6, 4.6))
for g in np.unique(yw):
    m = yw == g
    ax.scatter(Zw[m, 0], Zw[m, 1], s=16, color=P.group_color(g), label=wine["target_name"][m].iloc[0])
ax.set_xlabel("LD1"); ax.set_ylabel("LD2"); ax.legend(fontsize=8, title="cultivar")
ax.set_title("Q: do 2 discriminant axes separate 3 cultivars?"); plt.show()

contrib = pd.Series(np.abs(lda3.coef_).sum(0), index=feat).sort_values(ascending=False)
print("Variables contributing most to separation (|standardized LDA weight|):")
print(contrib.head(5).round(2).to_string())
"""))

cells.append(md(r"""
**P5 (Interpret · logistic + confusion-matrix metrics).** Two-group penguin problem (Adelie vs Gentoo) with
logistic discrimination. Read the confusion matrix and compute the four core metrics.
"""))
cells.append(code(r"""
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

pen = data.load_penguins_df()
two = pen[pen["species"].isin(["Adelie", "Gentoo"])].copy()
Xp = two[data.PENGUIN_FEATURES].values
yp = (two["species"] == "Gentoo").astype(int).values     # 1 = Gentoo (the "positive" class)

Xtr, Xte, ytr, yte = train_test_split(Xp, yp, test_size=0.3, random_state=7, stratify=yp)
clf = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xtr, ytr)
pred = clf.predict(Xte)

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
sens = tp / (tp + fn); spec = tn / (tn + fp); prec = tp / (tp + fp); acc = (tp + tn) / len(yte)
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"sensitivity/recall = {sens:.2f}   specificity = {spec:.2f}   "
      f"precision = {prec:.2f}   accuracy = {acc:.2f}")
plots.confusion(yte, pred, labels=[0, 1], question="Q: where do the errors fall? (1=Gentoo)")
plt.show()
"""))
cells.append(md("**YOUR ANSWER (P5):** In words, what do *sensitivity* and *specificity* mean for this penguin "
                "classifier? If missing a Gentoo were very costly, which metric would you optimize?"))

cells.append(md(r"""
**P6 (Diagnose · data leakage — the silent score-inflator).** With pure-noise features and a random label,
*selecting the best features using all the data* before cross-validating fakes high accuracy. Doing the
selection **inside** the CV folds tells the truth.
"""))
cells.append(code(r"""
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

n, p = 200, 1000
Xrand = RNG.normal(size=(n, p)); yrand = RNG.integers(0, 2, size=n)   # no real signal

# WRONG: pick the 20 "best" features on ALL data, THEN cross-validate
keep = SelectKBest(f_classif, k=20).fit(Xrand, yrand).get_support()
leaked = cross_val_score(LogisticRegression(max_iter=500), Xrand[:, keep], yrand, cv=5).mean()

# RIGHT: selection happens inside each fold (no peeking at the held-out part)
honest = cross_val_score(
    Pipeline([("sel", SelectKBest(f_classif, k=20)), ("clf", LogisticRegression(max_iter=500))]),
    Xrand, yrand, cv=5).mean()

print(f"LEAKED  CV accuracy (select on all data): {leaked:.2f}   <- looks predictive, but isn't")
print(f"HONEST  CV accuracy (select inside folds): {honest:.2f}   <- ~0.5 = truth, no signal")
"""))

# ---------------------------------------------------------------- Close
cells.append(close())

cells.append(md(r"""
**Exam-style interpretation (5 sentences).** Report the Wine LDA: state the question, the method and its key
assumption, how well 2 discriminant axes separate the 3 cultivars, the 2–3 variables driving separation, and
how you validated (cross-validation) plus one assumption you would check.
"""))
cells.append(md("**YOUR 5 SENTENCES:**\n\n"))

cells.append(md("#### 🃏 Method card — Discriminant analysis (fill from memory)\n\n" + METHOD_CARD_BLANK))

cells.append(md(r"""
**Transfer question.** You have 3 known groups *and* 60 predictors with heavy multicollinearity. Combining
ideas from notebooks 03 and 06, sketch a two-step plan and name the leakage pitfall to avoid. (Two sentences.)
"""))
cells.append(md("**YOUR ANSWER:**\n\n"))

cells.append(md(r"""
**Suggested Anki prompts**
- Front: *Two-group LDA direction?* → Back: w = Sp⁻¹(μ₁−μ₀), pooled covariance Sp; classify by nearest Mahalanobis centre.
- Front: *LDA vs QDA boundary?* → Back: LDA assumes equal covariance → linear boundary; QDA allows per-group covariance → quadratic boundary.
- Front: *Sensitivity vs specificity?* → Back: sensitivity = TP/(TP+FN) (positives caught); specificity = TN/(TN+FP) (negatives cleared).
- Front: *Why is accuracy misleading?* → Back: under class imbalance a trivial majority guesser scores high; use recall/precision/AUC.
- Front: *Data leakage?* → Back: using test information in training (scaling/feature-selecting on full data) → optimistic, fake scores.
"""))

# ---------------------------------------------------------------- Answer key
cells.append(answer_key_banner())
cells.append(md(r"""
**🔁 Retrieve**
1. Pooled covariance estimates the common within-group covariance Σ (averaged over groups); LDA needs Sp⁻¹, so Sp must be invertible (predictors not collinear, enough data).
2. Clustering discovers unknown groups (unsupervised); discriminant analysis uses known groups (supervised).
3. A categorical (here binary) outcome.
4. Mahalanobis distance (using the pooled covariance).

**🔮 Predict** (model answers)
1. Straight / linear boundary.
2. Curved / quadratic; it can wrap around a group with different spread.
3. A model predicting "healthy" for everyone is 97% accurate yet useless — accuracy ignores the rare positives; use sensitivity/precision.
4. Yes — standardizing on the *whole* dataset before splitting leaks test information into training; scale inside a pipeline/CV fold.

**🛠️ Practice**
- *P1:* `Sp = ((n0-1)*S0+(n1-1)*S1)/(n0+n1-2)`, `w = np.linalg.solve(Sp, mu1-mu0)`, `c = w@(mu0+mu1)/2`; matches sklearn (>98%).
- *P2:* The two histograms along w·x overlap only a little — one axis separates the equal-covariance groups well.
- *P3:* (a) **LDA** is forced straight by the equal-covariance assumption, which is false here, so it misclassifies the spread-out group's tails. (b) **QDA** fits the curved boundary best, at the cost of more parameters (its own covariance per group) and needing more data; logistic is in between. (c) LDA is better when covariances really are similar and/or data is limited (fewer parameters → lower variance).
- *P4:* The 3 cultivars separate cleanly in LD1–LD2; top contributors typically include flavanoids, proline, colour intensity, OD280, alcohol.
- *P5:* Sensitivity = fraction of true Gentoos correctly flagged; specificity = fraction of Adelies correctly cleared. If missing a Gentoo is costly, maximize **sensitivity (recall)**.
- *P6:* Leaked ≈ 0.6–0.7 (fake), honest ≈ 0.5 (truth). Feature selection must live *inside* cross-validation.

**Transfer:** Step 1 — reduce the 60 collinear predictors with **PCA** (or regularized LDA); Step 2 — run LDA
on the components and validate by cross-validation. Pitfall: fit the PCA **inside** the CV pipeline, never on
the full dataset before splitting, or you leak test information.
"""))

build("06_discriminant_analysis_and_logistic_bridge.ipynb", cells)
