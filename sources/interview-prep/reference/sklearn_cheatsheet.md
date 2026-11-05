# sklearn Cheatsheet — CodeSignal DS Assessment
*Use this after you've done the probability problems.*

---

## The Universal Pattern (memorize this first)

```python
# Every sklearn model follows this exact structure:
from sklearn.X import ModelName

model = ModelName(hyperparameter=value)
model.fit(X_train, y_train)          # train
predictions = model.predict(X_test)  # predict
score = model.score(X_test, y_test)  # evaluate (returns accuracy for classifiers, R² for regressors)
```

That's it. Everything else is just filling in X, model name, and metrics.

---

## Step 0: Data Prep

### Train/Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% for testing
    random_state=42       # reproducibility
)
```

### Scaling (when asked to normalize/standardize)
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler: mean=0, std=1 (use for most models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit AND transform
X_test_scaled = scaler.transform(X_test)          # ONLY transform (never fit on test!)

# MinMaxScaler: scales to [0, 1]
scaler = MinMaxScaler()
# same pattern as above
```

**⚠️ EXAM TRAP:** `fit_transform` on train, `transform` only on test. Fitting on test = data leakage.

---

## Step 1: Models

### Classification

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Logistic Regression
model = LogisticRegression(max_iter=1000)  # increase max_iter if convergence warning
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # probability of positive class

# Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Feature importance (Random Forest / tree models)
importances = model.feature_importances_  # array, one value per feature
```

### Regression

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Coefficients
print(model.coef_)        # slope for each feature
print(model.intercept_)   # y-intercept
```

---

## Step 2: Metrics

### Classification Metrics
```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)   # add average='macro' for multiclass
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
auc       = roc_auc_score(y_test, y_prob)     # needs probabilities, not predictions

cm = confusion_matrix(y_test, y_pred)
# cm[0][0] = TN, cm[0][1] = FP
# cm[1][0] = FN, cm[1][1] = TP

print(classification_report(y_test, y_pred))  # prints precision/recall/f1 for all classes
```

**Confusion matrix layout:**
```
                 Predicted Negative  Predicted Positive
Actual Negative        TN                  FP
Actual Positive        FN                  TP
```

**The formulas (in case they ask):**
- Accuracy  = (TP + TN) / total
- Precision = TP / (TP + FP)  ← "of what I called positive, how many were right?"
- Recall    = TP / (TP + FN)  ← "of all actual positives, how many did I catch?"
- F1        = 2 × (precision × recall) / (precision + recall)

### Regression Metrics
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)                          # or mean_squared_error(..., squared=False)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)             # 1.0 = perfect, 0 = predicts mean, negative = worse than mean
```

---

## Step 3: Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

# Simple k-fold CV
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(scores.mean(), scores.std())

# Manual KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(X):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
```

**Common scoring strings:** `'accuracy'`, `'f1'`, `'roc_auc'`, `'r2'`, `'neg_mean_squared_error'`

---

## Step 4: Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
best_model = grid_search.best_estimator_
```

---

## Step 5: Pipelines (clean, no leakage)

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Use pipeline with cross-validation — no leakage, scaler refits each fold
scores = cross_val_score(pipe, X, y, cv=5)
```

---

## Pandas Quick Reference (for data wrangling modules)

```python
import pandas as pd
import numpy as np

df = pd.read_csv('file.csv')

# Exploration
df.head()
df.shape          # (rows, cols)
df.dtypes
df.describe()
df.info()
df['col'].value_counts()
df.isnull().sum()

# Selection
df['col']                          # single column (Series)
df[['col1', 'col2']]              # multiple columns (DataFrame)
df[df['col'] > 5]                  # filter rows
df.loc[0:5, 'col1':'col3']        # label-based slicing
df.iloc[0:5, 0:3]                  # position-based slicing

# Cleaning
df.dropna()                        # drop rows with any NaN
df.dropna(subset=['col'])          # drop rows where col is NaN
df['col'].fillna(df['col'].mean()) # fill NaN with mean
df.drop_duplicates()

# Engineering
df['new_col'] = df['col1'] + df['col2']
df['encoded'] = pd.get_dummies(df['cat_col'], drop_first=True)
df['col'].apply(lambda x: x * 2)

# Aggregation
df.groupby('category')['value'].mean()
df.groupby('category').agg({'value': ['mean', 'sum', 'count']})
df.pivot_table(values='val', index='row_cat', columns='col_cat', aggfunc='mean')

# Merging
merged = pd.merge(df1, df2, on='id', how='inner')  # inner/left/right/outer
```

---

## Full Pipeline — Copy-Paste Template

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# 1. Load
df = pd.read_csv('data.csv')

# 2. Prep
X = df.drop('target', axis=1)
y = df['target']

# Handle missing values (choose one)
X = X.fillna(X.mean())          # numerical: fill with mean
# X = X.dropna()                # or drop rows

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Pipeline (scaler + model)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Train
pipe.fit(X_train, y_train)

# 6. Evaluate
y_pred = pipe.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Cross-validate
cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
print(f"CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

---

## The 5 Things You Actually Need to Memorize

1. `fit` on train, `transform` on test (never `fit_transform` on test)
2. `predict_proba(X)[:, 1]` for probability of positive class (needed for AUC)
3. Confusion matrix: `cm[1][1]` = TP, `cm[0][0]` = TN, `cm[0][1]` = FP, `cm[1][0]` = FN
4. `cross_val_score(model, X, y, cv=5)` — takes the full dataset, not train split
5. Pipeline: scaler goes first, model goes last, fit the pipeline not the pieces

---

*You've seen this code before. You've written it. This is just reminding your fingers where the commas go.*
