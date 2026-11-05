# Python DS Syntax Cheatsheet — CodeSignal Quick Reference

*The 20% of pandas/sklearn you'll use 80% of the time.*

---

## ⚠️ CodeSignal Platform Gotchas

```python
# 1. DISK SPACE ERROR — You MUST delete input files before saving output
import os
os.remove('input_file_1.csv')
os.remove('input_file_2.csv')
# NOW you can save
df.to_csv('output.csv', index=False)

# 2. Copy-paste is DISABLED. You must type everything from memory.
# 3. No tab switching. No Google. Only official docs (but too slow to use).
# 4. Use print() statements to debug — no IDE debugger available.
print(df.shape)          # check dimensions after merge
print(df.dtypes)         # check types
print(df.head())         # sanity check
print(df.isnull().sum()) # check nulls before they bite you
```

---

## Pandas — Loading Multiple Files (Module 3 Pattern)

```python
import glob
import os

# Load all CSVs from a directory and concatenate
csv_files = glob.glob('data/*.csv')  # or use os.listdir()
dfs = [pd.read_csv(f) for f in csv_files]
combined = pd.concat(dfs, ignore_index=True)

# Alternative with os.listdir()
import os
folder = 'data/'
files = [f for f in os.listdir(folder) if f.endswith('.csv')]
dfs = [pd.read_csv(os.path.join(folder, f)) for f in files]
combined = pd.concat(dfs, ignore_index=True)
```

---

## Pandas — Data Cleaning

```python
# Shape & info
df.shape                          # (rows, cols)
df.info()                         # dtypes + nulls
df.describe()                     # summary stats
df.isnull().sum()                 # null count per column
df.duplicated().sum()             # total duplicate rows
df.duplicated(subset="col").sum() # duplicates on specific column

# Drop duplicates
df = df.drop_duplicates(subset="col", keep="first")

# Fill nulls
df["col"].fillna(value)                          # scalar
df["col"].fillna(df["col"].median())             # median
df["col"].fillna(df.groupby("g")["col"].transform("median"))  # grouped

# Replace values
df["col"] = df["col"].replace({"old": "new"})
df.loc[df["col"] < 0, "col"] = np.nan           # conditional replace

# String operations
df["col"].str.strip().str.upper()                # clean + uppercase
df["col"].str.title()                            # "basic" -> "Basic"
df["col"].str.contains("pattern", na=False)      # regex search

# Type conversion
pd.to_datetime(df["col"], errors="coerce")       # bad dates -> NaT
df["col"].astype(int)                            # type cast
pd.to_numeric(df["col"], errors="coerce")        # bad numbers -> NaN
```

## Pandas — Querying & Aggregation

```python
# Filter
df[df["col"] > 5]
df[df["col"].isin(["A", "B"])]
df[(df["a"] > 5) & (df["b"] < 10)]              # multiple conditions

# GroupBy
df.groupby("city")["salary"].mean()              # single agg
df.groupby("city").agg(                          # multiple aggs
    avg_salary=("salary", "mean"),
    total=("id", "count"),
    max_spend=("spend", "max"),
)
# GroupBy with dictionary style (also common)
df.groupby("city").agg({"salary": "mean", "id": "nunique", "spend": ["sum", "max"]})

# Transform (returns same-size Series — great for group-level features)
df["city_avg"] = df.groupby("city")["salary"].transform("mean")
df["rank"] = df.groupby("city")["salary"].transform("rank", ascending=False)
df["prev_value"] = df.groupby("id")["value"].shift(1)  # lag

# Mode (most frequent)
df["col"].mode().iloc[0]

# Merge / Join
pd.merge(left, right, on="key", how="inner")     # inner join
pd.merge(left, right, on="key", how="left")      # left join
pd.merge(left, right, on=["key1", "key2"], how="left")  # multi-column join
pd.merge(left, right, left_on="a", right_on="b") # different column names

# ⚠️ ALWAYS deduplicate + handle NaNs BEFORE merging!
# Bad merge on dirty data → Cartesian explosion → all downstream results wrong
left = left.drop_duplicates(subset="key")
right = right.drop_duplicates(subset="key")
merged = pd.merge(left, right, on="key", how="left")

# Concat (stacking DataFrames)
pd.concat([df1, df2, df3], ignore_index=True)     # stack vertically
pd.concat([df1, df2], axis=1)                      # stack horizontally

# Pivot / Value counts
df["col"].value_counts()                          # frequency table
df["col"].value_counts(normalize=True)            # proportions
pd.crosstab(df["a"], df["b"])                     # contingency table

# Sort
df.sort_values("col", ascending=False)
df.sort_values(["col1", "col2"], ascending=[True, False])  # multi-column

# New columns
df["new"] = df["a"] / (df["b"] + 1)              # avoid div by zero
df["flag"] = (df["income"] > threshold).astype(int)

# Quantiles
df["col"].quantile(0.75)

# Rolling window
df["rolling_avg"] = df["value"].rolling(window=7).mean()
df["cumsum"] = df["value"].cumsum()
```

## Pandas — Binning & Mixed-Format Handling (Module 4 Pattern)

```python
# Discretize continuous → categorical
df["age_bin"] = pd.cut(df["age"], bins=[0, 18, 35, 50, 100],
                       labels=["youth", "young_adult", "middle", "senior"])

# Equal-frequency bins
df["quartile"] = pd.qcut(df["income"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

# Map mixed-format strings to numbers (CONFIRMED CodeSignal pattern)
# e.g., "satisfaction" column has: "5 points", "very good", "3", "average"
mapping = {
    "very bad": 1, "bad": 2, "average": 3, "good": 4, "very good": 5,
    "1 point": 1, "2 points": 2, "3 points": 3, "4 points": 4, "5 points": 5,
}
# For string values
df["satisfaction_clean"] = df["satisfaction"].map(mapping)
# For values that might already be numeric
df["satisfaction_clean"] = df["satisfaction_clean"].fillna(
    pd.to_numeric(df["satisfaction"], errors="coerce")
)

# .apply() for custom transformations
df["clean"] = df["messy_col"].apply(lambda x: parse_value(x))

# .map() for dictionary-based replacement
df["encoded"] = df["category"].map({"low": 0, "medium": 1, "high": 2})
```

## Pandas — Saving Output

```python
# Always use index=False unless told otherwise
df.to_csv("output.csv", index=False)

# Rounding before saving (if required)
df["value"] = df["value"].round(3)

# Creating a results DataFrame from calculated values
results = pd.DataFrame({
    "insight_type": ["avg_price", "pct_sunroof", "avg_views_30d"],
    "value": [avg_price, pct_sunroof, avg_views]
})
results["value"] = results["value"].round(2)
results.to_csv("analysis_results.csv", index=False)
```

## Pandas — Method Chaining (Speed Technique)

```python
# Instead of 5 intermediate variables, chain operations:
result = (
    df
    .drop_duplicates(subset="id")
    .merge(other_df, on="id", how="left")
    .fillna({"col": 0})
    .assign(new_col=lambda x: x["a"] / x["b"])
    .query("new_col > 0.5")
    .groupby("category")
    .agg(total=("amount", "sum"), count=("id", "nunique"))
    .reset_index()
    .sort_values("total", ascending=False)
)
# Faster to write, fewer bugs, less cognitive overhead
```

## Pandas — One-Hot Encoding

```python
# Simple
pd.get_dummies(df, columns=["cat_col"], drop_first=True)

# Or with sklearn
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["encoded"] = le.fit_transform(df["cat_col"])
```

## Matplotlib / Seaborn — Quick Plots

```python
# Histogram
df["col"].hist(bins=30)
plt.title("Distribution of X")
plt.xlabel("X")
plt.ylabel("Count")
plt.savefig("plot.png")
plt.show()

# Bar plot
df.groupby("cat")["target"].mean().plot(kind="bar")

# Boxplot
df.boxplot(column="numeric", by="category")

# Overlapping histograms
df[df["target"] == 0]["col"].hist(alpha=0.5, label="No", color="blue")
df[df["target"] == 1]["col"].hist(alpha=0.5, label="Yes", color="red")
plt.legend()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="RdBu_r")

# Seaborn countplot
sns.countplot(data=df, x="category", hue="target")

# ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
```

## Scikit-Learn — ML Pipeline

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,  # classification
    mean_squared_error, mean_absolute_error, r2_score,         # regression
)
from sklearn.preprocessing import StandardScaler

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale (for Logistic Regression, SVM, KNN — NOT needed for trees)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
X_test_scaled = scaler.transform(X_test)          # transform only on test!

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)           # class labels
y_prob = model.predict_proba(X_test)[:, 1]  # probabilities (for AUC)

# Evaluate — Classification
accuracy_score(y_test, y_pred)                     # overall accuracy
precision_score(y_test, y_pred)                     # TP/(TP+FP)
recall_score(y_test, y_pred)                        # TP/(TP+FN)
f1_score(y_test, y_pred)                            # harmonic mean of P&R
roc_auc_score(y_test, y_prob)                       # AUC-ROC
classification_report(y_test, y_pred)               # full report
confusion_matrix(y_test, y_pred)                    # TP, FP, TN, FN

# Evaluate — Regression
from math import sqrt
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)                                    # or mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

## Scikit-Learn — Preprocessing (Module 4/5 Pattern)

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

# Impute missing values
imputer = SimpleImputer(strategy="mean")       # or "median", "most_frequent", "constant"
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Label Encoding (ordinal categories: low < medium < high)
le = LabelEncoder()
df["encoded"] = le.fit_transform(df["category"])

# One-Hot Encoding (nominal categories: red, blue, green)
# Option 1: pandas (simpler for quick work)
df = pd.get_dummies(df, columns=["color"], drop_first=True)

# Option 2: sklearn (needed for pipeline)
ohe = OneHotEncoder(sparse_output=False, drop="first")
encoded = ohe.fit_transform(df[["color"]])

# Scaling
scaler = StandardScaler()   # mean=0, std=1 (use for most models)
# scaler = MinMaxScaler()   # range [0,1] (use when you need bounded values)
X_scaled = scaler.fit_transform(X)
```

## Scikit-Learn — Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

# Quick cross-validation (returns array of scores)
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")

# Stratified K-Fold (maintains class proportions — use for imbalanced data)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring="precision")

# Available scoring strings: "accuracy", "precision", "recall", "f1",
# "roc_auc", "neg_mean_squared_error", "neg_mean_absolute_error", "r2"
```

## Scikit-Learn — Complete Module 5 Template

```python
# ⚡ THE PATTERN: memorize this end-to-end, write it in <15 min
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score

# 1. Load & separate features/target
df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# 2. Handle categoricals
X = pd.get_dummies(X, drop_first=True)

# 3. Impute missing values
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 4. Split (ALWAYS stratify for classification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Scale (optional for RF, but doesn't hurt)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Train (RF is forgiving — good default for hitting precision >0.95)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 7. Predict & evaluate
y_pred = model.predict(X_test)
prec = precision_score(y_test, y_pred)
print(f"Precision: {prec:.4f}")  # Must be > 0.95

# 8. Save results if required
import os
# Delete input files to free disk space first!
os.remove("data.csv")
results = pd.DataFrame({"actual": y_test, "predicted": y_pred})
results.to_csv("predictions.csv", index=False)
```

## Common Models — Quick Init

```python
# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=42, max_iter=1000)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# XGBoost
import xgboost as xgb
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1,
    random_state=42, eval_metric="logloss",
)

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind="barh")
```

## Class Imbalance Handling

```python
# Method 1: scale_pos_weight in XGBoost
ratio = (y_train == 0).sum() / (y_train == 1).sum()
xgb.XGBClassifier(scale_pos_weight=ratio)

# Method 2: class_weight in sklearn
LogisticRegression(class_weight="balanced")
RandomForestClassifier(class_weight="balanced")

# Method 3: SMOTE (needs imblearn)
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)
```

## Common Gotchas

- `fit_transform` on train, `transform` only on test (prevent data leakage)
- `stratify=y` in train_test_split for imbalanced targets
- `errors="coerce"` for pd.to_datetime and pd.to_numeric
- `drop_first=True` in get_dummies to avoid multicollinearity
- XGBoost needs numeric features only (encode categoricals first)
- `predict_proba()[:, 1]` for the positive class probability
- **ALWAYS `drop_duplicates()` before `merge()`** — Cartesian products kill everything
- **Delete input CSVs with `os.remove()` before `to_csv()`** — disk space limit
- **`index=False`** in `to_csv()` — hidden unit tests expect no index column
- **Round to exact decimals requested** — `df["col"].round(3)` or `round(value, 3)`
- **`ignore_index=True`** in `pd.concat()` — avoids duplicate index issues
- **Check `df.shape` after every merge** — if row count explodes, you have dupes
- **`random_state=42`** everywhere — reproducibility for unit tests
- For classification with precision threshold: try `RandomForestClassifier(n_estimators=200)` before anything fancier — it's more forgiving than LogisticRegression for hitting >0.95
