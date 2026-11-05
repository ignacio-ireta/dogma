"""Generate 02_multiple_regression.ipynb."""

from nbbuild import (METHOD_CARD_BLANK, SETUP_CELL, answer_key_banner, build,
                     close, code, header, learn, md, practice, predict, retrieve)

cells = []

cells.append(header(
    title="02 · Multiple Regression",
    est_time="100–130 min",
    objectives=[
        "Compute least-squares coefficients from the normal equations and read least squares as a projection.",
        "Read a real `statsmodels` coefficient table: estimates, standard errors, t, p-values, confidence intervals.",
        "Judge fit with R², adjusted R², and RMSE, and tell train fit from genuine predictive ability.",
        "Diagnose multicollinearity (VIF), leverage/influence, and residual problems — and the danger of blind stepwise selection.",
    ],
    prereq="01 (projection, covariance, positive definiteness/invertibility). 00 (method map).",
    mvp="Do 🔁 Retrieve, the **normal-equations vs statsmodels** agreement cell, and read the **coefficient table** cell. "
        "Then skim the two failure cases (high-R² illusion, multicollinearity).",
    extra="\n\nUnit 1 of the syllabus. The exam asks you to *interpret real output and write a short report* — so we "
          "spend real time reading a coefficient table, not just fitting.",
))

cells.append(code(SETUP_CELL))

# ---------------------------------------------------------------- Retrieve
cells += retrieve([
    "From 01: 'project y onto the column space of X' — what is the residual in that picture, and to what is it orthogonal?",
    "From 01: the normal equations need (XᵀX)⁻¹. What property of XᵀX must hold for that inverse to exist?",
    "From 00: is regression supervised or unsupervised, and what is its goal-verb?",
    "Pre-req: in simple regression y = a + bx, what does the slope b mean in words?",
])

# ---------------------------------------------------------------- Predict
cells += predict([
    "Least squares picks coefficients to minimize *what* quantity? Write the objective in words.",
    "You add a useless, random predictor to a regression. Predict what happens to (a) R² and (b) adjusted R².",
    "Two predictors are nearly perfectly correlated. Predict what happens to their estimated coefficients and standard errors.",
    "You get R² = 0.98 on your data. Predict one reason this could still be a bad model.",
])

# ---------------------------------------------------------------- Learn
cells.append(learn("Learn the system — multiple regression as a system"))

cells.append(md(r"""
**Research question.** "How does a **numeric** outcome $y$ depend on several predictors $x_1,\dots,x_p$,
and how well can we predict/explain it?"

**Input shape.** $X$: $n\times p$ numeric predictors (plus a column of 1's for the intercept); $y$: length-$n$ numeric target. **Supervised**.

**Objective (least squares).** Choose $\beta$ to minimize the sum of squared residuals
$$\hat\beta=\arg\min_\beta \|y-X\beta\|^2 \;\Rightarrow\; \boxed{X^\top X\,\hat\beta = X^\top y}\;\Rightarrow\;\hat\beta=(X^\top X)^{-1}X^\top y.$$
Geometrically, $\hat y=X\hat\beta$ is the **orthogonal projection** of $y$ onto the column space of $X$;
the residual $e=y-\hat y$ is perpendicular to every predictor (that perpendicularity *is* the normal equations).

**Fitted quantities.** coefficients $\hat\beta$, fitted values $\hat y$, residuals $e$, and their standard
errors $\Rightarrow$ t-statistics and p-values.

**Assumptions (for the inference, not just the fit).** linearity; errors independent, constant-variance
(homoscedastic), roughly normal; predictors not perfectly collinear; no high-leverage point distorting everything.

**Goodness of fit.**

| Measure | Formula | Reading |
|---|---|---|
| $R^2$ | $1-\frac{\text{SSE}}{\text{SST}}$ | fraction of variance explained (never decreases when you add predictors) |
| Adjusted $R^2$ | $1-\frac{\text{SSE}/(n-p-1)}{\text{SST}/(n-1)}$ | penalizes useless predictors (can go down) |
| RMSE | $\sqrt{\text{SSE}/n}$ | typical error, in the units of $y$ |

**Diagnostics.** residuals-vs-fitted (curvature/heteroscedasticity), QQ-plot (normality), **VIF** > ~5–10
(multicollinearity), leverage $h_{ii}$ and **Cook's distance** (influential points), train-vs-test gap (overfit).

**Common traps.** (1) Adding predictors to chase $R^2$ → overfit; use adjusted $R^2$ / a test set. (2)
**Multicollinearity** → unstable, uninterpretable coefficients with huge SEs. (3) **Blind stepwise**
selection → inflated significance, non-reproducible models. (4) Reading a coefficient "holding others fixed"
when the others *can't* be held fixed (collinearity).

**Neighbours.** Logistic regression (06) = regression for a categorical outcome. PCA (03) can pre-compress
collinear predictors (principal-component regression).
"""))

# ---------------------------------------------------------------- Practice
cells.append(practice())

cells.append(md(r"""
**P1 (Compute · normal equations from scratch).** Solve $\hat\beta=(X^\top X)^{-1}X^\top y$ yourself and
confirm it matches `statsmodels` on the Diabetes data.
"""))
cells.append(code(r"""
import statsmodels.api as sm
dia = data.load_diabetes_df()
predictors = [c for c in dia.columns if c != "target"]
Xmat = sm.add_constant(dia[predictors].values)   # adds the intercept column of 1s
y = dia["target"].values

# TODO: solve the normal equations for beta_hat.
# HINT 1 (concept): beta = (XᵀX)⁻¹ Xᵀy.
# HINT 2 (numerics): prefer np.linalg.solve(XtX, Xty) over an explicit inverse.
# HINT 3 (code): XtX = Xmat.T @ Xmat; Xty = Xmat.T @ y; beta_hat = np.linalg.solve(XtX, Xty)
beta_hat = None   # <- replace

ols = sm.OLS(y, Xmat).fit()
def _v():
    assert_close(beta_hat, ols.params, tol=1e-6, name="normal-equations vs statsmodels coefficients")
selfcheck(beta_hat is not None, _v)
"""))

cells.append(md(r"""
**P2 (Interpret · read the real coefficient table).** This is the exam skill. Run it, then answer below.
The columns: `coef` (estimate), `std err`, `t`, `P>|t|` (p-value), and the 95% CI.
"""))
cells.append(code(r"""
print(ols.summary().tables[1])     # the coefficient table only (not the whole dump)
print(f"\nR2 = {ols.rsquared:.3f}   adjusted R2 = {ols.rsquared_adj:.3f}   "
      f"F p-value = {ols.f_pvalue:.2e}")
print(f"RMSE = {np.sqrt(np.mean(ols.resid**2)):.2f}  (target ranges {y.min():.0f}–{y.max():.0f})")
"""))
cells.append(md(r"""
**YOUR ANSWER (P2):** (a) Name the two or three predictors that are clearly significant (p < 0.05).
(b) For `bmi`, write one sentence interpreting its coefficient *holding the others fixed*. (c) Does a big
coefficient always mean a big effect here — why or why not? (Hint: are these predictors on the same scale?)
"""))

cells.append(md(r"""
**P3 (Recognize · R², adjusted R², RMSE from scratch).** Compute all three from residuals and check
against statsmodels. This cements what each number is.
"""))
cells.append(code(r"""
yhat = Xmat @ ols.params
resid = y - yhat
n, p = len(y), len(predictors)
SSE = np.sum(resid**2); SST = np.sum((y - y.mean())**2)

# TODO: compute R2, adjusted R2, RMSE from SSE/SST.
# HINT: R2 = 1 - SSE/SST ; adjR2 = 1 - (SSE/(n-p-1))/(SST/(n-1)) ; RMSE = sqrt(SSE/n)
R2 = None; adjR2 = None; RMSE = None   # <- replace all three
def _v():
    assert_close([R2, adjR2], [ols.rsquared, ols.rsquared_adj], tol=1e-6, name="R2 and adjusted R2")
    assert_close(RMSE, np.sqrt(SSE/n), name="RMSE")
selfcheck(R2 is not None, _v)
"""))

cells.append(md(r"""
**P4 (Diagnose · the high-R² illusion — failure case #1).** We regress a random target on many random
predictors with few observations. Watch in-sample R² soar while *test* R² collapses. This is why R²
alone never proves a good model.
"""))
cells.append(code(r"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

n_obs, n_noise = 60, 40
Xn = RNG.normal(size=(n_obs, n_noise))           # pure noise predictors
yn = RNG.normal(size=n_obs)                      # target unrelated to X
Xtr, Xte, ytr, yte = train_test_split(Xn, yn, test_size=0.4, random_state=7)

lin = LinearRegression().fit(Xtr, ytr)
print(f"TRAIN R2 = {lin.score(Xtr, ytr):.3f}   <- looks great")
print(f"TEST  R2 = {lin.score(Xte, yte):.3f}   <- the truth: no real signal")
print("\nLesson: a high in-sample R2 with many predictors and few rows is an artefact, not skill.")
"""))
cells.append(md(r"""
**YOUR ANSWER (P4):** In one sentence — what single number exposed the illusion, and what cheap practice
would have protected the analyst?
"""))

cells.append(md(r"""
**P5 (Diagnose · multicollinearity — failure case #2).** We add a predictor that is almost a copy of an
existing one. The fit barely changes, but the two coefficients become wild and their standard errors
explode. VIF flags it.
"""))
cells.append(code(r"""
from statsmodels.stats.outliers_influence import variance_inflation_factor

base = dia[["bmi", "bp", "s5"]].copy()
base["bmi_copy"] = base["bmi"] + RNG.normal(scale=1e-3, size=len(base))   # near-duplicate of bmi
Xc = sm.add_constant(base.values)
m_collin = sm.OLS(y, Xc).fit()

vif = [variance_inflation_factor(Xc, i) for i in range(Xc.shape[1])]
tab = pd.DataFrame({"variable": ["const"] + list(base.columns),
                    "coef": np.round(m_collin.params, 1),
                    "std_err": np.round(m_collin.bse, 1),
                    "VIF": np.round(vif, 1)})
print(tab.to_string(index=False))
print("\nbmi and bmi_copy: enormous opposite coefficients, huge std errors, VIF in the thousands.")
"""))
cells.append(md(r"""
**YOUR ANSWER (P5):** (a) Why do `bmi` and `bmi_copy` get huge, opposite coefficients? (b) What does a VIF
of, say, 2000 tell you? (c) Give one sensible fix.
"""))

cells.append(md(r"""
**P6 (Interpret · residual & influence diagnostics).** Two standard diagnostic plots on the real model.
"""))
cells.append(code(r"""
infl = ols.get_influence()
cooks = infl.cooks_distance[0]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
a1.scatter(yhat, resid, s=12, color=P.RESIDUAL, alpha=0.6); a1.axhline(0, color=P.NEUTRAL)
a1.set_xlabel("fitted ŷ"); a1.set_ylabel("residual"); a1.set_title("Q: any curvature / fan shape?")
a2.stem(cooks, linefmt=P.HIGHLIGHT, markerfmt=" ", basefmt=" ")
a2.axhline(4/len(y), color=P.NEUTRAL, ls="--", label="rule-of-thumb 4/n")
a2.set_xlabel("observation"); a2.set_ylabel("Cook's distance")
a2.set_title("Q: any single influential point?"); a2.legend(fontsize=8)
plt.tight_layout(); plt.show()
print(f"Largest Cook's distance: {cooks.max():.3f} at observation {cooks.argmax()}.")
"""))

cells.append(md(r"""
**P7 (Choose-and-defend · standardized coefficients).** To compare predictor *importance* on a common
scale, refit on standardized predictors. Then answer.
"""))
cells.append(code(r"""
from sklearn.preprocessing import StandardScaler
Xz = StandardScaler().fit_transform(dia[predictors].values)
mz = sm.OLS((y - y.mean()) / y.std(), sm.add_constant(Xz)).fit()
std_betas = pd.Series(mz.params[1:], index=predictors).sort_values(key=np.abs, ascending=False)
print("Standardized coefficients (largest |effect| first):")
print(std_betas.round(3).to_string())
"""))
cells.append(md(r"""
**YOUR ANSWER (P7):** Using the standardized coefficients, which 2–3 predictors matter most, and why is
comparing *standardized* coefficients fairer than comparing the raw ones from P2?
"""))

# ---------------------------------------------------------------- Close
cells.append(close())

cells.append(md(r"""
**Exam production — write the report (this is graded on the real exam).** In **5 sentences**, report the
Diabetes regression: state the question, name the significant predictors and the direction/size of the
strongest effect, give the fit (R²/adjusted R²/RMSE), note one diagnostic concern, and state one
limitation. Write for a non-statistician colleague.
"""))
cells.append(md("**YOUR 5-SENTENCE REPORT:**\n\n"))

cells.append(md("#### 🃏 Method card — Multiple regression (fill from memory)\n\n" + METHOD_CARD_BLANK))

cells.append(md(r"""
**Transfer question.** Your target is now *binary* (disease: yes/no) instead of numeric. Which neighbouring
method replaces multiple regression, and name one thing that breaks if you force ordinary least squares on a
0/1 outcome. (One or two sentences.)
"""))
cells.append(md("**YOUR ANSWER:**\n\n"))

cells.append(md(r"""
**Suggested Anki prompts**
- Front: *Least-squares normal equations?* → Back: XᵀXβ̂ = Xᵀy ⇒ β̂ = (XᵀX)⁻¹Xᵀy; ŷ is the projection of y onto col(X).
- Front: *R² vs adjusted R²?* → Back: R² never decreases adding predictors; adjusted R² penalizes useless ones and can fall.
- Front: *VIF > 5–10 means?* → Back: serious multicollinearity — unstable coefficients, inflated standard errors.
- Front: *High train R², low test R² means?* → Back: overfitting; the model memorized noise, no real predictive skill.
- Front: *Why standardize coefficients?* → Back: to compare predictor importance on a common (per-σ) scale.
"""))

# ---------------------------------------------------------------- Answer key
cells.append(answer_key_banner())
cells.append(md(r"""
**🔁 Retrieve**
1. Residual $e=y-\hat y$; it is orthogonal (perpendicular) to the column space of X — to every predictor.
2. XᵀX must be invertible (positive definite) — i.e. predictors not perfectly collinear / X full column rank.
3. Supervised; goal-verb = predict.
4. b = expected change in y for a one-unit increase in x.

**🔮 Predict** (model answers)
1. The sum of squared residuals ‖y − Xβ‖².
2. (a) R² rises (or stays equal); (b) adjusted R² typically falls.
3. Coefficients become large, unstable, often opposite in sign; standard errors blow up.
4. Overfitting / too many predictors for too few rows; or leakage; or extrapolation — high R² ≠ predictive skill.

**🛠️ Practice**
- *P1:* `beta_hat = np.linalg.solve(Xmat.T@Xmat, Xmat.T@y)` — matches statsmodels.
- *P2:* Typically **bmi**, **s5** (and often **bp**, **s1/s3**) are significant. bmi coef ≈ +X means: per one-unit rise in (already-scaled) bmi, predicted progression rises by X, holding others fixed. A bigger raw coefficient is **not** a bigger effect because the predictors are on different scales — compare standardized coefficients (P7).
- *P3:* `R2 = 1-SSE/SST`, `adjR2 = 1-(SSE/(n-p-1))/(SST/(n-1))`, `RMSE = sqrt(SSE/n)`; all match.
- *P4:* The **test R²** (≈ 0 or negative) exposed it; a simple **train/test split** (or cross-validation) protects you.
- *P5:* (a) The model can trade coefficient mass between two near-identical columns arbitrarily, so it picks a huge +β on one and −β on the other. (b) VIF ≈ 2000 means that predictor's variance is ~2000× inflated by collinearity — essentially redundant. (c) Drop one of the pair, combine them, or use regularization / PCA.
- *P6:* Residuals should look like a structureless band around 0 (no funnel/curve); Cook's distances should mostly sit below 4/n with no single dominating spike.
- *P7:* Usually **s5, bmi, bp** dominate. Standardized coefficients are per-standard-deviation, so they are comparable across predictors measured in different units; raw coefficients are not.

**Transfer:** Use **logistic regression** (Unit 5). OLS on a 0/1 target can predict values below 0 or above 1
(not probabilities) and violates the constant-variance/normal-error assumptions, so inference is invalid.
"""))

build("02_multiple_regression.ipynb", cells)
