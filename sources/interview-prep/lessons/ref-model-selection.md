# Model Selection — Decision Framework for Consulting Case Interviews

*The right model is the simplest one that solves the business problem. The evidence backs this up more than you'd expect.*

---

## 🚨 QUICK REFERENCE — The Decision Tree (Memorize This)

When the interviewer describes a problem, run these questions **in order**:

| #   | Question                        | → Answer                                                                                                                                          |
| --- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **What's the target?**          | Continuous → regression. Binary → classification. Count → Poisson GLM. Time-to-event → survival. No target → unsupervised. Time → forecasting.    |
| 2   | **Must the client EXPLAIN it?** | Yes (board/regulator) → linear/logistic regression, GAMs. Somewhat → Random Forest + SHAP. No → gradient boosting.                                |
| 3   | **How much labeled data?**      | <1K → regularized regression, Bayesian. 1K-10K → gradient boosting with careful CV. 10K+ → gradient boosting. 1M+ → deep learning becomes viable. |
| 4   | **What kind of data?**          | Tabular → GBDTs. Images → CNNs. Text → transformers. Sequences → transformers/LSTMs. Graph → GNNs.                                                |
| 5   | **Is there a time component?**  | Forecast → ARIMA baseline → Prophet → GBDTs with temporal features. Panel → fixed/random effects.                                                 |
| 6   | **Linear or nonlinear?**        | Mostly linear → regularized linear models (faster, comparable accuracy). Nonlinear/unknown → tree-based.                                          |
| 7   | **Deployment constraints?**     | Real-time (<10ms) → linear models, small trees. Batch → anything. Quick baseline → CatBoost defaults.                                             |

> **🔑 The one-liner that wins:** "I'd start with a simple baseline to establish a performance floor, then increase complexity only if the business case justifies it."

---

## 📋 MODEL CARDS

Each card follows the same structure: **When → Why → Failure Modes → VP Version → Interview Phrasing**

Three depth levels per card:
- 🟢 **Minimum viable answer** — say this if your brain is frozen
- 🟡 **Strong answer** — this is your target
- 🔴 **Impressive answer** — only if you're feeling sharp and the interviewer is pushing

---

### Linear Regression

**When:** Continuous target, ~linear relationships, need coefficient interpretation.

**Why:** Coefficients map directly to business levers. "A 1-unit increase in X drives $Y revenue."

**Failure modes:**
- Omitted variable bias → biased coefficients, misleading causal claims (the most dangerous one)
- Multicollinearity → inflates standard errors but predictions are fine
- Heteroscedasticity → confidence intervals unreliable, point estimates still unbiased
- Non-linearity → add polynomial/interaction terms (still "linear" regression!)

**VP version:** "We built a model that tells you exactly how much each factor contributes to the outcome. Turn this dial, get this result."

**Interview phrasing:**
- 🟢 "I'd start with linear regression as a baseline because I need interpretable coefficients."
- 🟡 "Linear regression gives us direct coefficient interpretation — 'a 1-unit increase in X drives $Y.' I'd check residual plots for violations and VIF for multicollinearity."
- 🔴 "Linear regression can actually handle nonlinear relationships via feature transformations — it's linear in *parameters*, not necessarily in features. I'd also distinguish between using it for prediction vs. causal inference, since the assumptions that matter differ."


---

### Logistic Regression

**When:** Binary classification with probability estimates and interpretable odds ratios. The default starting model for any classification task in consulting.

**Why:** Outputs probabilities AND explanations. "Customers with feature X have 3x the odds of churning."

**Failure modes:**
- Separation (a predictor perfectly predicts outcome) → infinite coefficients. Fix with Firth's penalized likelihood.
- Confusing odds ratios with risk ratios — high odds may = low absolute risk
- Needs ~10-20 events per predictor. 5 predictors + 10% event rate = need ≥500 observations.
- Linearity of log-odds assumption — verify with Box-Tidwell test

**VP version:** "We can tell you the probability each customer churns AND which factors matter most, in plain language."

**Interview phrasing:**
- 🟢 "Logistic regression — it gives me probabilities and interpretable odds ratios."
- 🟡 "I'd start with logistic regression because the client needs to understand WHY customers churn, not just which ones. Odds ratios translate directly to business actions."
- 🔴 "I'd check for separation, verify linearity of log-odds, and confirm we have enough events per predictor — rule of thumb is 10-20 EPV. If the client confuses odds ratios with risk ratios, I'd present absolute risk differences instead."


---

### GLMs (Generalized Linear Models)

**When:** Your target variable doesn't fit normal regression assumptions.

| Data type | GLM | Example |
|-----------|-----|---------|
| Counts | Poisson | Customer complaints, insurance claims |
| Overdispersed counts | Negative Binomial | When Poisson variance assumption fails |
| Right-skewed continuous | Gamma | Claim amounts, revenue per customer |
| Bounded proportions | Beta | Conversion rates, market share |

**Why:** Match the model to the data's natural distribution instead of forcing transformations.

**VP version:** "Instead of awkwardly transforming your data to fit a standard model, we used a model designed for this type of data. Better predictions, more honest uncertainty estimates."

**Interview phrasing:**
- 🟢 "The target is count data, so I'd use a Poisson GLM instead of forcing linear regression."
- 🟡 "Revenue per customer is right-skewed with a long tail — I'd use a Gamma GLM with a log link rather than log-transforming the target, which distorts the error structure."
- 🔴 "If the Poisson's equidispersion assumption fails — and it usually does with real data — I'd switch to negative binomial to handle overdispersion."

---

### Regularized Regression (LASSO, Ridge, Elastic Net)

**When:** Many features, multicollinearity, or you want automatic feature selection.

| Method | What it does | Use when |
|--------|-------------|----------|
| **Ridge (L2)** | Shrinks all coefficients toward zero | Most predictors contribute, multicollinearity exists |
| **LASSO (L1)** | Sets some coefficients to exactly zero | You want sparse, interpretable models |
| **Elastic Net** | Both L1 + L2 | **When in doubt, start here** — subsumes both |

**Key gotcha:** LASSO arbitrarily picks one among correlated features and drops the rest. If two features are highly correlated and both matter, LASSO lies to you. Ridge or Elastic Net handles this.

**Interview phrasing:**
- 🟢 "With 100+ features, I'd use regularized regression — probably Elastic Net to get both feature selection and multicollinearity handling."
- 🟡 "I'd use `ElasticNetCV` to cross-validate both the regularization strength and the L1/L2 balance. It subsumes LASSO and Ridge as special cases."
- 🔴 "LASSO is tempting for feature selection, but with correlated predictors it arbitrarily drops features. Elastic Net gives me sparsity without that instability."

---

### Decision Trees

**When:** You need perfect interpretability — you can literally draw the decision path for a stakeholder.

**Why they're fragile:** High variance (small data changes → completely different tree), overfit easily, generally underperform ensembles.

**Real use:** Initial exploration, generating business rules, and as the building block you explain BEFORE introducing ensembles.

**Interview phrasing:**
- 🟢 "A single tree is great for explanation but too fragile for production. I'd use it to build intuition, then move to an ensemble."

---

### Random Forest

**When:** You need a reliable model fast and slight accuracy loss vs. boosting is acceptable.

**How:** Ensemble of trees, each on a bootstrap sample with random feature subsets. Prediction = majority vote (clf) or average (reg).

**Strengths:** Robust out-of-box, handles mixed types/missing values, built-in feature importance, parallelizable.

**Weaknesses:** Less accurate than gradient boosting on most tabular tasks. Can't extrapolate beyond training data range. Large memory footprint.

**Interview phrasing:**
- 🟢 "Random Forest — reliable, handles messy data, hard to overfit."
- 🟡 "I'd start with Random Forest for robustness. It handles mixed features and missing values without preprocessing. If accuracy isn't sufficient, I'd move to gradient boosting."
- 🔴 "RF reduces variance through bagging and random feature subsets, but it can't extrapolate. If the business question requires predicting outside the training range — say, revenue at a price point we've never offered — a parametric model might be necessary."

---

### Gradient Boosting (XGBoost / LightGBM / CatBoost)

**When:** Tabular data and you want the best predictive performance. This is the workhorse.

| Library | Superpower | Best for |
|---------|-----------|----------|
| **XGBoost** | Most tunable, largest community | Maximum control, well-documented |
| **LightGBM** | Fastest training (histogram-based) | Large datasets (100K+ rows) |
| **CatBoost** | Best defaults, native categorical handling | Quick strong baseline, many categoricals |

**Failure modes:**
- Overfitting with too many rounds → **always use early stopping**
- Sensitive to noisy labels
- Sequential training limits parallelism
- Less interpretable than linear models — but SHAP bridges this gap

**VP version:** "We built a model that learns from its own mistakes. Each iteration focuses on the cases it got wrong before. It's the most accurate approach for this type of data."

**Interview phrasing:**
- 🟢 "XGBoost or LightGBM — they dominate tabular data."
- 🟡 "I'd use gradient boosting with early stopping. For a quick baseline, CatBoost's defaults are excellent. For maximum speed on large data, LightGBM. I'd use SHAP values to make it interpretable for stakeholders."
- 🔴 "Gradient boosting sequentially fits trees to residuals, so each tree corrects the previous ensemble's errors. The learning rate controls how aggressively we trust each new tree. I'd tune via Bayesian optimization rather than grid search — much more efficient in high-dimensional hyperparameter space."


---

### Survival Analysis

**When:** The question is "how long until X happens?" — churn timing, equipment failure, time to conversion.

**Cox Proportional Hazards:** Semi-parametric, outputs hazard ratios, standard for time-to-event. Critical assumption: proportional hazards (ratio stays constant over time). Test with Schoenfeld residuals.

**When Cox fails:** Switch to Accelerated Failure Time (AFT) models — they require a distributional assumption (Weibull, log-logistic) but output interpretable time ratios.

**VP version:** "Instead of just predicting WHO will churn, we can predict WHEN — so you can time your interventions."

**Interview phrasing:**
- 🟢 "Churn is a time-to-event problem — I'd consider survival analysis, specifically Cox Proportional Hazards."
- 🟡 "Cox PH gives us hazard ratios: 'customers with feature X have 2.3x the hazard of churning at any time point.' I'd verify the proportional hazards assumption with Schoenfeld residuals — if it fails, I'd consider AFT or time-varying covariates."


---

### Bayesian Methods

**When:** Small samples, need principled uncertainty, have domain knowledge to incorporate, or need full posterior distributions for decision-making.

**Tools (2025):** PyMC v5 (JAX backend, fast), NumPyro, Bambi (R-like formula syntax).

**VP version:** "We combined what we learned from the data with expert knowledge to make better predictions, especially where data is limited. And we can tell you exactly how confident we are."

**Interview phrasing:**
- 🟢 "With only 200 samples, frequentist estimates will be unstable — I'd use Bayesian regression with informative priors."
- 🟡 "Bayesian methods give us credible intervals instead of confidence intervals — they directly answer 'what's the probability the true effect is in this range?' which is what the client actually wants to know."

---

### K-Means

**When:** Well-separated, roughly equal-sized clusters. You know (or can estimate) K.

**How:** Initialize K centroids → assign points → update centroids → repeat.

**Choosing K:** Elbow method (inertia vs K), silhouette score.

**Failure modes:** Assumes spherical equal-sized clusters, sensitive to initialization (use k-means++), sensitive to outliers.

**Interview phrasing:**
- 🟢 "K-means for initial customer segmentation. I'd use the elbow method and silhouette scores to pick K."
- 🟡 "K-means is fast and intuitive but assumes spherical clusters. If the segments have irregular shapes or varying density, I'd switch to HDBSCAN."

---

### DBSCAN / HDBSCAN

**When:** Unknown K, irregular cluster shapes, noise/outliers in data.

**DBSCAN:** Discovers arbitrary shapes, identifies outliers, doesn't need K. But two tricky parameters (epsilon, min_samples) and struggles with varying-density clusters.

**HDBSCAN:** Extends DBSCAN to handle varying densities. More robust to parameters. In sklearn since v1.3.

**The modern pipeline:** **UMAP → HDBSCAN** is the standard for high-dimensional clustering. UMAP reduces dimensions while preserving both local and global structure, then HDBSCAN finds clusters.

**Interview phrasing:**
- 🟢 "DBSCAN — it finds clusters of any shape and handles outliers automatically."
- 🟡 "For high-dimensional customer data, I'd use UMAP for dimensionality reduction followed by HDBSCAN for clustering. UMAP preserves both local and global structure, and HDBSCAN handles varying-density clusters without needing to specify K."

---

### Gaussian Mixture Models (GMMs)

**When:** You need **soft clustering** — probabilistic membership rather than hard assignments.

**VP version:** "Instead of forcing each customer into one box, we can say 'this customer is 70% budget-conscious and 30% premium-seeking.' That's more realistic and lets you personalize more precisely."

**Interview phrasing:**
- 🟢 "If customers can belong to multiple segments, I'd use a Gaussian Mixture Model for soft clustering."
- 🟡 "GMMs give probabilistic cluster assignments — useful when the segments overlap, like a customer who's both price-sensitive AND brand-loyal. I'd use BIC to select the number of components."

---

### Dimensionality Reduction (PCA / t-SNE / UMAP)

| Method | Preserves | Use for |
|--------|-----------|---------|
| **PCA** | Global structure (linear) | Feature engineering for downstream models, denoising |
| **t-SNE** | Local neighborhoods | Visualization ONLY (don't use for features) |
| **UMAP** | Local + global | Visualization AND preprocessing (better than t-SNE for most uses) |

**Key insight:** Only PCA is suitable for feature engineering in downstream models. t-SNE and UMAP are for exploration/visualization.

---

### Anomaly Detection

| Method | When to Use |
|--------|-------------|
| **Isolation Forest** | Tabular data. "Anomalies are easier to isolate, so they have shorter path lengths." |
| **Autoencoder** | Complex patterns. "Trains on normal data, flags high reconstruction error." |
| **Statistical (Z-score, IQR)** | Simple, univariate baseline. "Start simple." |

---

### Time Series

| Scenario | Reach for... | Why |
|----------|-------------|-----|
| Quick baseline | ARIMA / Seasonal Naive | Classical, interpretable, fast |
| Business series with holidays/seasonality | Prophet | Interpretable decomposition (trend + seasonality + holidays) |
| Need exogenous variables | LightGBM with temporal features | Gradient boosting handles mixed inputs well |
| Awareness mention (if asked about cutting edge) | Time Series Foundation Models (Chronos-Bolt, TimesFM) | Zero-shot forecasting, 15-55% error reduction in benchmarks |

**Interview phrasing:**
- 🟢 "I'd start with ARIMA as a baseline, then try Prophet if there are multiple seasonalities."
- 🟡 "ARIMA requires stationarity — I'd check with an ADF test. Prophet handles holidays and missing data automatically. But honestly, LightGBM with good temporal features often beats both."

---

### Recommendation Systems

| Method | When to Use |
|--------|-------------|
| **Collaborative Filtering** | User-item interaction data exists. "Users who bought X also bought Y." |
| **Content-Based** | Item features available. Cold-start friendly. |
| **Hybrid** | Best of both — collaborative for engaged users, content-based for new users. |

---

### Optimization

| Method | When to Use |
|--------|-------------|
| **Linear Programming** | Linear constraints. "Minimize cost subject to capacity constraints." |
| **Constraint Programming** | Complex discrete decisions. Scheduling, assignment. |
| **Heuristics / Genetic** | Problem too complex for exact solutions. "Good-enough answer fast." |

---

### Neural Networks (When and ONLY When)

**Use for:** Images (CNNs), text (transformers), sequences with massive data (LSTMs/transformers). NOT for tabular data unless you have a specific reason.

**The tabular verdict (NeurIPS 2024, 176 datasets):** The "neural network vs GBDT" debate is overemphasized. For most tabular tasks, light hyperparameter tuning on GBDTs matters more than algorithm choice.

**Interview phrasing:**
- 🟢 "Neural networks shine for unstructured data — images, text, sequences. For tabular data, gradient boosting is simpler and usually wins."
- 🟡 "The largest tabular benchmark found that tuning GBDTs matters more than switching to neural nets. I'd only go to deep learning for tabular data if I had multimodal inputs or needed incremental online updates."

---

## 🎯 THE JUSTIFICATION TEMPLATE

Every time you name a model, use this structure:

> "I'd use **[model]** because:
> 1. **It fits the problem** — [type] on [data type]
> 2. **It matches the constraints** — [interpretability/speed/accuracy]
> 3. **It handles the data** — [mixed features/missing values/imbalance/etc.]
>
> I considered **[alternative]** but rejected it because **[reason]**.
> If this doesn't perform, I'd escalate to **[fallback]**."

---

## 🧠 THE EVIDENCE: SIMPLER MODELS WIN MORE THAN YOU THINK

This isn't just a heuristic — it's empirically validated:

- **Forecasting meta-analysis** (Journal of Business Research): Simpler methods were as or more accurate in the vast majority of comparisons, reducing error by up to 43%.
- **Cancer risk prediction** (13 cancer types): Logistic regression slightly outperformed ML on average. 10 of 13 comparisons showed no statistical difference.
- **M-series competitions** (most rigorous time series benchmarks): Simple statistical methods consistently matched or outperformed ML.

**Why this matters in consulting:** Consulting values **speed to value**. A model deployed in 2 weeks that captures 80% of value beats one deployed in 6 months capturing 95%. The firm's motto: "Think Big, Start Small, Grow Fast."

**The interpretability-accuracy tradeoff is THE consulting tradeoff.** In regulated industries, an interpretable model that's 2% less accurate than a black box will almost always be preferred. A logistic regression a VP can read in a slide deck delivers more organizational value than a marginally better gradient boosting model that nobody trusts.

---

## ❌ ANTI-PATTERNS — What NOT to Do

| Don't Say | Why | Say Instead |
|-----------|-----|-------------|
| "I'd use deep learning" (tabular) | Overkill. Consulting values pragmatism. | "GBDTs dominate tabular data. I'd start with XGBoost." |
| "I'd try everything and see" | No structure. No hypothesis. | "I'd start with [model] because [reason]. If underperforms, [alternative]." |
| "This model is the best" | No model is universally best. | "This model fits THIS problem because [reasons]." |
| "I'd use accuracy" | Misleading with imbalanced classes. | "Precision-recall or F1 given the class imbalance." |
| (Name a model without explaining it) | They WILL ask you to go deeper. | Always be ready for "why?" and "how does it work?" |
| "I'd use a transformer" (for tabular) | Shows you're chasing hype. | "Transformers excel at unstructured data. For tabular, I'd use gradient boosting." |

---

## 🔬 FAILURE MODES — Know These Cold

Being able to name failure modes signals real experience. Interviewers test this.

| Model | Primary Failure Mode | What to Say |
|-------|---------------------|-------------|
| Linear regression | Omitted variable bias | "Biased coefficients if we miss a confounding variable — the most dangerous violation." |
| Logistic regression | Separation | "If a predictor perfectly predicts the outcome, coefficients explode to infinity. Fix with penalized likelihood." |
| LASSO | Correlated feature arbitrage | "LASSO arbitrarily picks one among correlated features and drops the rest. Use Elastic Net." |
| Decision trees | High variance | "Small data changes → completely different tree. That's why we ensemble." |
| Random forests | Can't extrapolate | "Predictions are bounded by training data range. Can't predict outside what it's seen." |
| Gradient boosting | Overfitting without early stopping | "Always use early stopping. Monitor validation loss, stop when it stops improving." |
| K-means | Spherical assumption | "Assumes round, equal-sized clusters. Real customer segments rarely look like that." |
| Neural networks | Data hunger + overfitting | "Need large data AND regularization (dropout, weight decay, early stopping). Otherwise they memorize noise." |
| ARIMA | Stationarity requirement | "Requires stationarity — real business data often has trends and structural breaks." |
| All time series | Regime shifts | "Every time series model struggles when the underlying process fundamentally changes." |

---

## 💬 THREE PHRASES THAT SIGNAL CONSULTING-READY THINKING

Memorize these. Use them naturally.

1. **"Before choosing a model, I want to understand what decision this model needs to support and how the output will be used."**

2. **"I'd start with a simple baseline — logistic regression or CatBoost with defaults — to establish a performance floor, then increase complexity only if the business case justifies it."**

3. **"The model accuracy matters less than whether the organization can actually act on the predictions. A simpler model the operations team trusts and uses will outperform a complex model that sits unused."**

---

## 🧪 RETRIEVAL PROMPTS — Test Yourself

*Don't read the answers above. Try from memory first. Struggle is the point.*

1. A client has customer complaint COUNT data. What model family? Why not linear regression?
2. You have 500 labeled samples and strong domain knowledge. What approach?
3. The client's board needs to understand every prediction. You want the best accuracy. What's the tradeoff and what do you recommend?
4. You built an XGBoost model. The client asks "why not a neural network?" What do you say?
5. You're clustering customers but the segments have weird shapes and varying density. K-means isn't working. What's next?
6. A churn model asks "which customers will churn?" vs "when will they churn?" — different question, different model. What's the second one?
7. You used LASSO for feature selection and got 5 features. A colleague points out some of the dropped features were highly correlated with the selected ones. What's the problem?
8. Name the primary failure mode for: (a) logistic regression, (b) gradient boosting, (c) ARIMA.
9. The VP asks you to explain your gradient boosting model in one sentence. Go.
10. Why might a simpler model actually deliver MORE business value than a complex one?

---


*Start simple. Prove value. Iterate. That's consulting. That's also just good engineering.* 🏔️
