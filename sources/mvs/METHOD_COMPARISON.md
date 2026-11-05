# Method comparison sheet — last-minute review

One page that collates the five method cards. Cover the right-hand columns and recite them; by exam day you
should be able to rebuild this from memory.

## Quick decision

```
NUMERIC outcome to explain/predict ............... MULTIPLE REGRESSION   (U1)
KNOWN groups → separate / classify ............... DISCRIMINANT / LOGISTIC (U5)
No target, re-express VARIABLES:
   • maximize variance, new axes ................. PCA                   (U2)
   • model hidden factors behind correlations .... FACTOR ANALYSIS       (U3)
No target, group OBSERVATIONS .................... CLUSTER ANALYSIS      (U4)
```

| | Supervised? | Goal verb | Core objective |
|---|---|---|---|
| **Multiple regression** | supervised | predict | minimize ‖y − Xβ‖² (project y onto col(X)) |
| **PCA** | unsupervised | reduce | maximize projected variance (top eigenvectors of Σ/R) |
| **Factor analysis** | unsupervised | find latent | reproduce correlations as ΛΛᵀ + Ψ |
| **Cluster analysis** | unsupervised | group | minimize within-cluster distance (or merge by linkage) |
| **Discriminant analysis** | supervised | discriminate | separate known groups (Mahalanobis / log-odds) |

---

## 1 · Multiple regression (Unit 1)
- **Question:** how does numeric y depend on predictors X; which matter, how well do they predict?
- **Input:** n×p numeric X + numeric y (+ intercept).
- **Objective:** β̂ = (XᵀX)⁻¹Xᵀy; ŷ is the orthogonal projection of y onto the predictors.
- **Assumptions:** linearity; independent, constant-variance, ~normal errors; no perfect collinearity; no dominating influential point.
- **Key output:** coefficients + std errors → t, p-values, CIs; R²/adjusted R²/RMSE.
- **Diagnostics:** residuals-vs-fitted, QQ-plot, **VIF**, leverage/Cook's distance, train-vs-test gap.
- **Choose complexity:** adjusted R² / cross-validation; avoid blind stepwise.
- **Trap:** chasing R² (overfit); multicollinearity → unstable coefficients; reading raw coefficients as importance.
- **One-liner:** "a weighted recipe of predictors for a number, with honest uncertainty."
- **Closest competitor:** logistic regression (categorical outcome).

## 2 · Principal component analysis (Unit 2)
- **Question:** can many correlated variables be re-expressed by a few uncorrelated axes?
- **Input:** n×p numeric, **standardized** by default (correlation PCA).
- **Objective:** orthogonal directions maximizing projected variance = top eigenvectors of the covariance/correlation matrix (= SVD of centred data).
- **Assumptions:** linearity; sensible scaling; outlier-light. (No distribution needed to compute.)
- **Key output:** loadings (variable→component), scores (observation coordinates), explained-variance ratios.
- **Diagnostics:** scree plot, cumulative variance, **parallel analysis**, reconstruction error.
- **Choose complexity (k):** cumulative-variance threshold + parallel analysis + interpretability.
- **Trap:** not standardizing (scaling trap); confusing scores with loadings; over-reading component **signs**; outliers hijacking PC1.
- **One-liner:** "a few new uncorrelated axes capturing most of the spread."
- **Closest competitor:** factor analysis (models shared variance via latent factors, not total variance).

## 3 · Factor analysis (Unit 3)
- **Question:** do a few **unobserved factors** generate the correlations among observed variables?
- **Input:** n×p numeric (standardized); check **KMO/Bartlett** first.
- **Objective:** estimate Λ, Ψ so that R ≈ ΛΛᵀ + Ψ (model the off-diagonal / shared variance).
- **Assumptions:** a low-rank common structure + diagonal uniqueness; enough common variance (KMO); ~linear relations.
- **Key output:** loadings, **communalities** (shared variance), **uniquenesses**; rotated for simple structure.
- **Diagnostics:** KMO, Bartlett, communalities in [0,1], % variance, residual correlations.
- **Choose complexity (m):** eigenvalues>1 / parallel analysis / variance / **interpretability**.
- **Trap:** treating factors as discovered truths; over-extraction → Heywood (uniqueness→0 / communality>1); naming from one loading.
- **One-liner:** "a small set of latent factors that reproduce the observed correlations."
- **Closest competitor:** PCA (compression of total variance; components are exact, not latent).

## 4 · Cluster analysis (Unit 4)
- **Question:** do observations fall into natural groups not defined in advance?
- **Input:** n×p numeric, **standardized**; choice of distance matters.
- **Objective:** k-means → minimize within-cluster sum of squares; hierarchical → merge by linkage into a tree.
- **Assumptions:** a meaningful distance/scale; (k-means) roughly convex, comparable-size clusters.
- **Key output:** cluster labels, centroids/dendrogram, per-cluster profiles.
- **Diagnostics:** elbow (inertia), **silhouette**, stability across seeds, domain usefulness.
- **Choose complexity (k):** elbow + silhouette + interpretability; dendrogram cut height.
- **Trap:** not scaling (largest-unit variable dominates); labels are **nominal not ordinal**; clusters ≠ ground truth; outliers.
- **One-liner:** "data-driven groups of similar observations — discovered, not given."
- **Closest competitor:** discriminant analysis (supervised: groups known in advance).

## 5 · Discriminant analysis & logistic (Unit 5)
- **Question:** given **known** groups, what separates them and how do I classify a new case?
- **Input:** n×p numeric X + known class y.
- **Objective:** LDA → w = Sₚ⁻¹(μ₁−μ₂), classify by nearest Mahalanobis centre; logistic → model log-odds β₀+βᵀx.
- **Assumptions:** LDA: multivariate-normal groups with **equal covariance** (QDA drops equal-cov → quadratic boundary); logistic: no normality assumption.
- **Key output:** discriminant axes/coefficients, decision boundary, **confusion matrix** + sensitivity/specificity/precision/recall.
- **Diagnostics:** **cross-validated** accuracy; check covariance equality; guard against **data leakage**.
- **Choose complexity:** #discriminant axes ≤ min(p, g−1); LDA vs QDA vs logistic by assumptions + CV.
- **Trap:** LDA under very unequal covariances; accuracy under class imbalance; leakage inflating CV scores; raw-coefficient "importance".
- **One-liner:** "a validated rule that separates and labels known groups."
- **Closest competitor:** cluster analysis (unsupervised cousin); logistic ↔ regression for categorical outcomes.

---

### Foundations behind all five (Notebook 01)
covariance matrix Σ = XcᵀXc/(n−1) · eigenvectors = axes of the data ellipse · eigenvalues = variance per axis ·
positive (semi)definite (wᵀΣw ≥ 0) · Mahalanobis d = √((x−μ)ᵀΣ⁻¹(x−μ)) · SVD Xc = UΣVᵀ (PCA engine, low-rank
reconstruction) · projection = drop a perpendicular onto a subspace (regression & PCA both do this).
