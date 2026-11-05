# Model Failure Modes — Cheat Sheet

Know these cold. Interviewers test whether you understand WHERE models break.

---

| Model | Primary Failure Mode | What to Say |
|-------|---------------------|-------------|
| **Linear Regression** | Omitted variable bias | "If we miss a confounding variable, coefficients are biased — the most dangerous violation because it looks fine on surface metrics." |
| **Logistic Regression** | Separation | "If a predictor perfectly predicts the outcome, coefficients explode to infinity. Fix with Firth's penalized likelihood or drop the feature." |
| **LASSO** | Correlated feature arbitrage | "LASSO picks one among correlated features and drops the rest arbitrarily. Use Elastic Net if you suspect multicollinearity." |
| **Decision Trees** | High variance | "Small data changes → completely different tree. That's why we ensemble — RF or boosting." |
| **Random Forest** | Can't extrapolate | "Predictions are bounded by training data range. If the business question requires predicting outside historical range, a parametric model might be needed." |
| **Gradient Boosting** | Overfitting without early stopping | "Always use early stopping on a validation set. Without it, boosting will memorize noise." |
| **K-Means** | Spherical cluster assumption | "Assumes round, equal-sized clusters. Real customer segments rarely look like that. DBSCAN or HDBSCAN for arbitrary shapes." |
| **Neural Networks** | Data hunger + overfitting | "Needs large data AND regularization (dropout, weight decay, early stopping). On small tabular datasets, GBDTs almost always win." |
| **ARIMA** | Stationarity requirement | "Requires stationarity — real business data often has trends and structural breaks. Differencing helps, but regime shifts break it." |
| **All time series** | Regime shifts | "Every time series model assumes the future resembles the past. Structural breaks (COVID, policy changes) violate this fundamentally." |

---

## The Master Principle

**Bias-Variance Tradeoff:**
- High-bias models (linear regression, naive Bayes) → underfit, miss real patterns
- High-variance models (deep nets, unpruned trees) → overfit, capture noise
- Adding complexity only helps when you have enough signal AND enough data

**The evidence says simpler wins more than expected:**
- Meta-analyses show simpler methods as or more accurate in the vast majority of forecasting comparisons
- Logistic regression matched or slightly beat ML in cancer risk prediction across 13 cancer types
- The M-series competitions (most rigorous benchmarks) consistently showed simple statistical methods matching ML

**Interview translation:** "I'm aware that the evidence shows simpler models outperform more often than most practitioners expect. That's why I always start with a baseline and only add complexity when the data justifies it."

---

*Knowing failure modes is what separates "I read the docs" from "I've shipped models." Show you know WHERE things break, not just how to build them.* 🏔️
