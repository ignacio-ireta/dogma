# Code Comprehension — What Does This Code Do?

**Format:** Show this code. Ask the questions below. Conversational — no coding required.

```python
def build_credit_score(df, features, target='defaulted'):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    clean = df[features + [target]].dropna()
    X = clean[features]
    y = clean[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    importance = dict(zip(features, model.coef_[0]))

    return model, scaler, auc, importance
```

## Questions to Ask

1. **"Walk me through what this function does."**
   - Expected: Builds a logistic regression credit scoring model. Drops rows with missing data, splits into train/test, scales features, trains with balanced class weights (to handle default/non-default imbalance), evaluates with AUC, and returns the model + feature importance via coefficients.

2. **"What's the issue with `dropna()` on the full feature+target set?"**
   - Expected: If ANY feature has a missing value, the entire row is dropped. With multiple features having 5-8% missing, you could lose 20-30% of your data. Better approach: impute missing values, or use a model that handles them natively (like gradient boosting).

3. **"The function uses `class_weight='balanced'`. What does that do, and when would you NOT want it?"**
   - Expected: It adjusts class weights inversely proportional to class frequency — giving more weight to the minority class (defaulters). Useful when you care about detecting defaults. But in lending, the COST of a false positive (rejecting a good applicant) and false negative (approving a defaulter) are very different. A cost-sensitive approach with explicit cost ratios might be better than simple balanced weights.
