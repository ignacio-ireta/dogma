"""Generate 04_factor_analysis.ipynb."""

from nbbuild import (METHOD_CARD_BLANK, SETUP_CELL, answer_key_banner, build,
                     close, code, header, learn, md, practice, predict, retrieve)

cells = []

cells.append(header(
    title="04 · Factor Analysis",
    est_time="100–130 min",
    objectives=[
        "State the latent-variable model x = Λf + ε and split variance into common (communality) and unique.",
        "Read loadings, communalities, uniquenesses; run KMO/Bartlett adequacy checks.",
        "Choose the number of factors and apply a rotation; explain orthogonal vs oblique.",
        "Articulate exactly how factor analysis differs from PCA, and name factors defensibly (not as discovered truths).",
    ],
    prereq="03 (scores vs loadings, choosing components, scaling). 01 (correlation matrix).",
    mvp="Do 🔁 Retrieve, the **recover-known-factors** synthetic cell, and the **PCA-vs-FA** comparison cell. "
        "Those carry the two ideas the exam tests: what FA estimates, and how it differs from PCA.",
    extra="\n\n**Honest framing up front:** factors are *modelling choices*, not objects the data hands you. "
          "Two analysts can defensibly extract different factors. Naming a factor is an argument from its loadings "
          "plus domain knowledge — never 'the computer found it'.",
))

cells.append(code(SETUP_CELL))

cells.append(code(
    "import warnings\n"
    "# factor_analyzer 0.5 calls a deprecated scikit-learn argument; the FutureWarning\n"
    "# is harmless library noise, so hide it (real convergence warnings still show).\n"
    "warnings.filterwarnings('ignore', category=FutureWarning)"))

# ---------------------------------------------------------------- Retrieve
cells += retrieve([
    "From 03: define a **loading** and a **score** in one line each.",
    "From 03: name two ways to choose the number of components, and which guards against keeping noise.",
    "From 01: what is on the diagonal of a **correlation** matrix, and what does an off-diagonal entry measure?",
    "From 00: PCA and factor analysis share the same goal-verb — which one — yet differ in purpose. State the purpose difference in a phrase.",
])

# ---------------------------------------------------------------- Predict
cells += predict([
    "FA assumes a few hidden factors *cause* the correlations among observed variables. Predict: what part of each variable's variance is left *unexplained* by the factors called?",
    "PCA components are built to maximize variance. Predict what factor-analysis loadings are built to reproduce instead.",
    "A 'rotation' leaves the model's fit unchanged but changes the loadings. Predict why an analyst would bother rotating.",
    "You force 8 factors onto 6 variables. Predict what could go wrong with the estimated uniquenesses.",
])

# ---------------------------------------------------------------- Learn
cells.append(learn("Learn the system — factor analysis as a system"))

cells.append(md(r"""
**Research question.** "Are these many correlated variables driven by a *small number of unobserved
factors*? What are those factors and how strongly does each variable reflect them?"

**Input shape.** $X$: $n\times p$ numeric variables (usually standardized). **Unsupervised**.

**The model.** Each standardized variable is a linear combination of $m$ common factors plus its own unique part:
$$ x = \Lambda f + \varepsilon,\qquad \operatorname{Cov}(f)=I,\ \operatorname{Cov}(\varepsilon)=\Psi\ (\text{diagonal}). $$
This implies the correlation matrix splits as
$$ R \approx \Lambda\Lambda^\top + \Psi. $$
- $\Lambda$ ($p\times m$) = **loadings** (variable ↔ factor).
- $\operatorname{diag}(\Lambda\Lambda^\top)_i = h_i^2$ = **communality** (variance of $x_i$ explained by the common factors).
- $\Psi_{ii} = 1-h_i^2$ = **uniqueness** (variable-specific variance + noise).

> **The defining contrast with PCA.** PCA reproduces the **whole** correlation matrix (diagonal included) by
> repackaging *total* variance. FA models only the **off-diagonal** (shared) structure and deliberately sets
> aside each variable's unique variance $\Psi$. PCA components are exact linear combinations of the data; FA
> factors are *latent* and only estimated.

**Fitting.** Estimate $\Lambda,\Psi$ so $\Lambda\Lambda^\top+\Psi$ best reproduces $R$ (principal-axis /
minimum-residual / maximum-likelihood). **Factor scores** (each subject's estimated $f$) are then *estimated*,
not computed exactly.

**Rotation (identifiability).** $\Lambda$ is only defined up to rotation: $\Lambda$ and $\Lambda Q$ fit
equally well. So we rotate toward **simple structure** (each variable loads strongly on one factor).
*Orthogonal* (varimax) keeps factors uncorrelated; *oblique* (promax/oblimin) lets them correlate — often
more realistic for real traits.

**Adequacy checks (do them first).** **Bartlett's sphericity** (is $R$ different from identity at all?) and
**KMO** (is there enough common variance to factor? KMO > ~0.6 acceptable).

**Choosing $m$.** scree / eigenvalues > 1 on the correlation matrix, **parallel analysis**, variance
explained, and — decisively — **interpretability**.

**Common traps.** (1) Treating factors as objectively real. (2) Extracting too many factors → **Heywood
cases** (communality > 1 / negative uniqueness = impossible variance). (3) Naming a factor from one big
loading while ignoring the rest. (4) Running FA when KMO says there's nothing to factor.
"""))

# ---------------------------------------------------------------- Practice
cells.append(practice())

cells.append(md(r"""
**P1 (Compute · communalities from a known Λ, from scratch).** Given a loading matrix, the communality of
a variable is the sum of its squared loadings, and (for standardized data) uniqueness = 1 − communality.
"""))
cells.append(code(r"""
Lam = np.array([[0.8, 0.1],
                [0.7, 0.0],
                [0.0, 0.75],
                [0.1, 0.65]])     # 4 variables, 2 factors

# TODO: communalities h2 (per row) and uniquenesses psi for standardized variables.
# HINT 1 (concept): communality = shared variance = sum of squared loadings in that row.
# HINT 2 (formula): h2_i = sum_j Lam[i,j]**2 ;  psi_i = 1 - h2_i
# HINT 3 (code): h2 = (Lam**2).sum(axis=1); psi = 1 - h2
h2 = None; psi = None   # <- replace
def _v():
    R_model = Lam @ Lam.T + np.diag(psi)
    assert_close(np.diag(R_model), np.ones(4), tol=1e-9, name="model variances sum to 1")
    assert_close(h2, (Lam**2).sum(1), name="communalities")
    print("communalities:", np.round(h2, 3), " uniquenesses:", np.round(psi, 3))
selfcheck(h2 is not None and psi is not None, _v)
"""))

cells.append(md(r"""
**P2 (Compute · recover known factors — mechanism obvious).** We *build* data from a known 2-factor
structure (items 1–3 load factor A, items 4–6 load factor B), then check FA recovers it. Because we know
the truth, this is the cleanest possible test of what FA estimates.
"""))
cells.append(code(r"""
from factor_analyzer import FactorAnalyzer

Lam_true = np.array([[0.80, 0.00],
                     [0.70, 0.00],
                     [0.75, 0.00],
                     [0.00, 0.80],
                     [0.00, 0.70],
                     [0.00, 0.75]])
h2_true = (Lam_true**2).sum(1)
nobs = 600
f = RNG.normal(size=(nobs, 2))
eps = RNG.normal(size=(nobs, 6)) * np.sqrt(1 - h2_true)
Xsyn = f @ Lam_true.T + eps
cols = [f"item{i+1}" for i in range(6)]
Xsyn = pd.DataFrame(Xsyn, columns=cols)

fa = FactorAnalyzer(n_factors=2, rotation="varimax", method="principal")
fa.fit(Xsyn)
rec = pd.DataFrame(np.round(fa.loadings_, 2), index=cols, columns=["F1", "F2"])
print("Recovered loadings (sign/column order may differ from truth):")
print(rec.to_string())
print("\nItems 1-3 should load one factor, items 4-6 the other — the block structure we built in.")
"""))
cells.append(md(r"""
**YOUR ANSWER (P2):** Look at the recovered loadings. Which items group together, and does that match the
block structure we planted? (Sign and which-column-is-which are arbitrary — focus on the *pattern*.)
"""))

cells.append(md(r"""
**P3 (Interpret · adequacy then extraction on real data — Wine).** Before factoring, check KMO and
Bartlett. Then extract factors and read communalities.
"""))
cells.append(code(r"""
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

wine = data.load_wine_df(); feat = list(wine.columns[:13])
Xw = wine[feat]

chi2, bart_p = calculate_bartlett_sphericity(Xw)
kmo_per, kmo_model = calculate_kmo(Xw)
print(f"Bartlett sphericity: chi2={chi2:.0f}, p={bart_p:.2e}  -> {'factorable' if bart_p < .05 else 'NOT'}")
print(f"KMO (overall)      : {kmo_model:.2f}  -> {'adequate' if kmo_model > .6 else 'weak'}")

fa3 = FactorAnalyzer(n_factors=3, rotation="varimax", method="principal").fit(Xw)
comm = pd.Series(fa3.get_communalities(), index=feat)
print("\nCommunalities (variance explained by the 3 common factors):")
print(comm.round(2).sort_values(ascending=False).to_string())
"""))

cells.append(md(r"""
**P4 (Interpret · rotation makes interpretation clearer — success case).** Same 3-factor solution, with and
without rotation. Rotation does not change fit; it concentrates each variable's loading onto one factor.
"""))
cells.append(code(r"""
fa_none = FactorAnalyzer(n_factors=3, rotation=None, method="principal").fit(Xw)
fa_vari = FactorAnalyzer(n_factors=3, rotation="varimax", method="principal").fit(Xw)

def simplicity(L):   # mean of max-squared-loading share per variable; higher = simpler structure
    sq = L**2
    return float(np.mean(sq.max(1) / sq.sum(1)))

print(f"simple-structure score  unrotated: {simplicity(fa_none.loadings_):.2f}")
print(f"simple-structure score  varimax  : {simplicity(fa_vari.loadings_):.2f}  (higher = easier to name)")

L = pd.DataFrame(np.round(fa_vari.loadings_, 2), index=feat, columns=["F1", "F2", "F3"])
print("\nVarimax loadings (|·|>0.4 are the interpretable ones):")
print(L.to_string())
"""))
cells.append(md(r"""
**YOUR ANSWER (P4):** Pick **one** factor and propose a name based on the variables that load > 0.4 on it.
Justify the name in one sentence — and state why your name is an *argument*, not a fact.
"""))

cells.append(md(r"""
**P5 (Diagnose · too many factors → over-extraction / Heywood — failure case).** A true *Heywood case* is a
communality > 1 (impossible: a variable can't have >100% of its variance explained). Software like
`factor_analyzer` *clamps* communalities at 1, so the symptom you actually see is **uniquenesses driven to
~0** (variables implausibly claimed as 100% common variance) while each extra factor explains less and less.
Watch both as `m` grows.
"""))
cells.append(code(r"""
print(f"{'m':>3} {'max communality':>16} {'#uniqueness<0.01':>17} {'last-factor var%':>17}")
for m in [2, 3, 5, 8, 11]:
    fa_m = FactorAnalyzer(n_factors=m, rotation=None, method="principal").fit(Xw)
    comm = fa_m.get_communalities(); uniq = 1 - comm
    _, prop_var, _ = fa_m.get_factor_variance()
    clamped = int(np.sum(uniq < 0.01))     # clamped Heywood cases (uniqueness pinned at 0)
    print(f"{m:>3} {comm.max():>16.3f} {clamped:>17d} {prop_var[-1]*100:>16.2f}%")
print("\nAt m=11 (near 13 variables): 7 variables hit ~0 uniqueness (clamped Heywood), and the last")
print("factor explains <2% — you are extracting noise. Fewer, interpretable factors beat 'more'.")
"""))

cells.append(md(r"""
**P6 (Choose-and-defend · PCA vs FA on the same data).** The exam loves this comparison. We fit both with
3 dimensions on Wine and contrast what each *claims*.
"""))
cells.append(code(r"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
Xz = StandardScaler().fit_transform(Xw)
pca = PCA(n_components=3, random_state=0).fit(Xz)

print("PCA  : reproduces TOTAL variance; components are exact linear combos of variables.")
print(f"       3 components keep {pca.explained_variance_ratio_.sum():.0%} of total variance.")
print("FA   : reproduces SHARED (off-diagonal) structure; sets aside unique variance Ψ.")
print(f"       mean communality with 3 factors = {fa_vari.get_communalities().mean():.2f} "
      f"(rest is each variable's uniqueness).")
print("\nSame output shape (variables x 3), different claim: compression (PCA) vs latent causes (FA).")
"""))
cells.append(md(r"""
**YOUR ANSWER (P6):** Your supervisor says "PCA and factor analysis are the same thing." Give two concrete
differences, and name one research goal for which you would specifically choose factor analysis.
"""))

# ---------------------------------------------------------------- Close
cells.append(close())

cells.append(md(r"""
**Exam-style interpretation (5 sentences).** Report the Wine 3-factor solution: state the question and why
FA (not PCA) suits a "hidden structure" claim, give the adequacy checks (KMO/Bartlett), say how many factors
and on what basis, name one factor from its loadings, and state one interpretive caution.
"""))
cells.append(md("**YOUR 5 SENTENCES:**\n\n"))

cells.append(md("#### 🃏 Method card — Factor analysis (fill from memory)\n\n" + METHOD_CARD_BLANK))

cells.append(md(r"""
**Transfer question.** Next you study **clustering**, which also finds 'hidden structure' but among
*observations* rather than variables. In one sentence, contrast what FA groups vs what clustering groups.
"""))
cells.append(md("**YOUR ANSWER:**\n\n"))

cells.append(md(r"""
**Suggested Anki prompts**
- Front: *Factor model equation?* → Back: x = Λf + ε, with Cov(f)=I, Cov(ε)=Ψ diagonal ⇒ R ≈ ΛΛᵀ + Ψ.
- Front: *Communality vs uniqueness?* → Back: communality = shared variance from common factors (Σ loadings²); uniqueness = 1 − communality.
- Front: *PCA vs FA, one line?* → Back: PCA repackages total variance (exact combos); FA models shared variance via latent factors, setting aside uniqueness.
- Front: *Why rotate?* → Back: same fit, simpler structure — each variable loads mainly on one factor → nameable.
- Front: *Heywood case?* → Back: communality > 1 / negative uniqueness — sign of over-extraction (too many factors).
"""))

# ---------------------------------------------------------------- Answer key
cells.append(answer_key_banner())
cells.append(md(r"""
**🔁 Retrieve**
1. Loading = weight of a variable on a component/factor (variable→axis). Score = an observation's coordinate on a component/factor.
2. Cumulative variance threshold and parallel analysis (also scree elbow, Kaiser); **parallel analysis** guards against noise.
3. Diagonal of a correlation matrix = 1; an off-diagonal entry = Pearson correlation between two variables.
4. Both 'reduce'; purpose differs: PCA = compression of total variance; FA = model latent factors generating shared variance.

**🔮 Predict** (model answers)
1. The **uniqueness** (unique/specific variance + noise), Ψ.
2. The **correlations** among variables (the off-diagonal of R), not maximal variance.
3. To reach simple structure so factors are interpretable/nameable (fit is unchanged).
4. Uniquenesses can go negative / communalities exceed 1 — impossible variances (Heywood cases).

**🛠️ Practice**
- *P1:* `h2 = (Lam**2).sum(1)` → [0.65, 0.49, 0.5625, 0.4325]; `psi = 1 - h2`.
- *P2:* Items 1–3 load one factor, items 4–6 the other — recovering the planted blocks (up to sign/column swap).
- *P3:* Bartlett p ≈ 0 (factorable), KMO around 0.7–0.8 (adequate); flavanoid/phenol-type variables get high communalities.
- *P4:* Varimax raises the simple-structure score. Example name: a factor loading flavanoids, total_phenols, OD280, proanthocyanins → "phenolic richness". It is an *argument* because the label is our interpretation of a loading pattern, not a measured quantity.
- *P5:* As m increases the last factor's variance share collapses (≈19%→<2%) and, by m=11, ~7 variables have uniqueness pinned at ~0 — clamped Heywood cases. A real Heywood case would push communality >1; the software clamps it at 1, but the over-extraction is just as diagnostic.
- *P6:* (1) FA has uniquenesses/error; PCA does not. (2) PCA components are exact linear combinations of the variables; FA factors are latent and only estimated (so are FA scores). Choose FA when the goal is to argue for/measure unobserved constructs (e.g. psychological traits).

**Transfer:** Factor analysis groups **variables** (which items share a latent cause); clustering groups
**observations** (which subjects resemble each other).
"""))

build("04_factor_analysis.ipynb", cells)
