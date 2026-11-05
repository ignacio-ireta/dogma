# ML Concepts Flashcards

*Cover the answer, try to recall, then check. Repeat daily.*

---

## ⚠️ Module 2 Strategy (Read This First)

**Format:** 5-6 questions, each with 5-6 options. You must select **ALL correct answers.** Partial = penalized.

**The trap:** Options are designed to be "50/50 confusing." Real-world nuance will tempt you to overthink.

**Rules:**
1. **Go with textbook definitions, not production edge cases.** If an option *could* be true in a weird scenario but isn't the standard academic answer, skip it.
2. **Max 2-3 min per question.** If you're deliberating, pick your best answer and MOVE ON. These minutes are worth more in M5.
3. **Common wrong-answer patterns to watch for:**
   - "Random Forest reduces **bias**" → WRONG. RF reduces **variance** (bagging = variance reduction).
   - "L1 regularization produces smooth weight decay" → WRONG. That's **L2**. L1 produces **sparsity** (weights go to exactly zero).
   - "Accuracy is the best metric for imbalanced datasets" → WRONG. Use F1, precision/recall, or AUC.
   - "More features always improve model performance" → WRONG. Curse of dimensionality, overfitting.
   - "High training accuracy means the model is good" → WRONG. Could be overfitting.

---

**Q: Precision?**
A: Of everything we flagged positive, how many were actually positive? TP/(TP+FP). "How trustworthy are our positive predictions?"

**Q: Recall (Sensitivity)?**
A: Of all actual positives, how many did we catch? TP/(TP+FN). "How many positives did we miss?"

**Q: F1 Score?**
A: Harmonic mean of precision and recall. 2×(P×R)/(P+R). The "weakest link" average — punishes imbalance between P and R.

**Q: Accuracy — when is it misleading?**
A: When classes are imbalanced. If 94% of customers pay, a model predicting "everyone pays" gets 94% accuracy while catching zero deadbeats.

**Q: AUC-ROC?**
A: Probability that the model ranks a random positive higher than a random negative. Plots TPR vs FPR at all thresholds. 1.0 = perfect, 0.5 = random coin flip. Threshold-independent.

**Q: Loss function vs evaluation metric?**
A: Loss = what the model minimizes DURING training (steering wheel). Metric = what YOU measure AFTER training (speedometer). They're often different.

**Q: XGBoost default loss for binary classification?**
A: Log loss (binary cross-entropy). Measures how far predicted probability is from true label.

**Q: What is gradient descent?**
A: Iteratively updates parameters by following the gradient (derivative) of the loss function downhill. Learning rate controls step size. Finds local minima.

**Q: Bagging vs Boosting?**
A: Bagging = parallel models on random subsets, averaged. Reduces variance. (Random Forest). Boosting = sequential models, each fixes previous errors. Reduces bias. (XGBoost).

**Q: What is XGBoost?**
A: eXtreme Gradient Boosting. Sequential ensemble where each new tree predicts the gradient of the loss function (corrects residual errors of previous trees).

**Q: Bias vs Variance?**
A: Bias = model too simple, misses patterns (underfitting). Variance = model too complex, memorizes noise (overfitting). Goal: minimize total error.

**Q: L1 (Lasso) regularization?**
A: Adds |weights| to loss. Drives some weights to EXACTLY zero. Feature selection. Sparse model. "Kill the useless features."

**Q: L2 (Ridge) regularization?**
A: Adds weights² to loss. Shrinks ALL weights toward zero but never exactly zero. Dense model. "Turn down the volume on everything."

**Q: Regularization vs Normalization?**
A: Regularization = penalty on model complexity during TRAINING (prevents overfitting). Normalization = scaling features to same range during PREPROCESSING. Completely different things.

**Q: Parameter vs Hyperparameter?**
A: Parameter = model learns it (weights, tree splits). Hyperparameter = YOU set it before training (learning_rate, max_depth, n_estimators).

**Q: Cross-validation?**
A: K-fold: split data into k parts, train on k-1, test on 1, rotate. Average scores. More reliable than single train/test split. Use stratified k-fold for imbalanced data.

**Q: Overfitting — how to detect?**
A: High training accuracy, low test/validation accuracy. The gap between them is the tell.

**Q: Overfitting — how to fix?**
A: Regularization (L1/L2), reduce model complexity, more training data, feature selection, dropout (NNs), early stopping.

**Q: Class imbalance — how to handle?**
A: scale_pos_weight (XGBoost), class_weight="balanced" (sklearn), SMOTE (oversample minority), undersampling majority, adjust decision threshold.

**Q: Sigmoid activation?**
A: Squashes output to (0,1). Used for binary classification output layer. Gives probability.

**Q: Softmax activation?**
A: Multi-class sigmoid. Outputs probability distribution across all classes (sums to 1). Used for multi-class output layer.

**Q: ReLU activation?**
A: max(0, x). Default for hidden layers. Fast, avoids vanishing gradient. Kills negative values.

**Q: MSE vs MAE?**
A: MSE (Mean Squared Error) = average of squared differences. Penalizes large errors more (outlier sensitive). MAE (Mean Absolute Error) = average of absolute differences. More robust to outliers.

**Q: What's a confusion matrix?**
A: 2×2 table of TP, FP, TN, FN. Shows exactly where your model succeeds and fails.

**Q: Feature importance — what does it mean?**
A: How much each feature contributes to model predictions. In tree models: based on how much each feature reduces impurity across all splits.

**Q: GBM vs Random Forest?**
A: Both are tree ensembles. RF = bagging (parallel, reduces variance). GBM = boosting (sequential, reduces bias). GBM usually more accurate but slower and easier to overfit.

**Q: When does adding more features hurt?**
A: Curse of dimensionality, data leakage risk, overfitting, increased training time, decreased interpretability, spurious correlations.

---

## 🆕 Additional Concepts (Confirmed on DSF)

**Q: R² (Coefficient of Determination)?**
A: Proportion of variance in the target explained by the model. R²=1.0 = perfect fit, R²=0 = no better than predicting the mean, R²<0 = worse than the mean. **Adjusted R²** penalizes adding useless features (accounts for number of predictors).

**Q: RMSE vs MAE vs MSE?**
A: MSE = mean of squared errors (punishes large errors). RMSE = √MSE (same units as target, easier to interpret). MAE = mean of absolute errors (robust to outliers). **Use RMSE when large errors are costly. Use MAE when outliers shouldn't dominate.**

**Q: Gini Coefficient (as evaluation metric)?**
A: Related to AUC: Gini = 2×AUC - 1. Ranges 0 to 1 (1 = perfect separation). Common in credit risk / insurance. Not the same as Gini impurity used in tree splits.

**Q: Log Loss (Binary Cross-Entropy)?**
A: Measures how far predicted probabilities are from true labels. Unlike accuracy, penalizes confident wrong predictions heavily. Lower = better. Formula: -[y×log(p) + (1-y)×log(1-p)]. Used by XGBoost, logistic regression.

**Q: How to choose k in KNN?**
A: Small k = complex boundary (overfitting, high variance). Large k = smoother boundary (underfitting, high bias). Use odd k for binary classification (avoids ties). **Cross-validation to find optimal k.** Typical: √n as starting point.

**Q: Naive Bayes — what's the "naive" assumption?**
A: All features are **conditionally independent** given the class label. This is almost never true in practice, but NB still works surprisingly well for text classification, spam filtering. Fast to train, good baseline.

**Q: What is data leakage?**
A: When information from outside the training set (e.g., future data, test set statistics) bleeds into training. Examples: fitting scaler on full data before splitting, using a feature that encodes the target, time-series trained on future data. **Always split FIRST, then preprocess.**

**Q: Regression coefficient interpretation?**
A: In linear regression, coefficient β₁ means: "a 1-unit increase in x₁ is associated with a β₁ change in y, **holding all other features constant.**" Sign = direction, magnitude = strength. Only meaningful if features are scaled similarly.

**Q: Logistic regression — what does it actually output?**
A: Log-odds (logit) of the positive class. Apply sigmoid to get probability. Coefficient β₁ means: "a 1-unit increase in x₁ changes the **log-odds** by β₁." To get odds ratio: e^β₁.

**Q: What is a decision boundary?**
A: The surface/line where the model switches from predicting one class to another. Linear models = straight line/hyperplane. Trees = axis-aligned rectangles. KNN = complex irregular boundaries. SVMs = margin-maximizing hyperplane.

**Q: Ensemble methods — why do they work?**
A: Combining multiple weak learners reduces overall error. Bagging (RF): reduces variance by averaging independent models. Boosting (GBM/XGB): reduces bias by sequentially fixing errors. Stacking: uses a meta-model to combine different algorithms.

**Q: What is feature scaling and when is it needed?**
A: Transforming features to similar ranges. **Needed for:** distance-based models (KNN, SVM), gradient-based (logistic regression, neural nets). **NOT needed for:** tree-based models (RF, XGBoost) — they split on thresholds, scale doesn't matter. StandardScaler: mean=0, std=1. MinMaxScaler: range [0,1].

**Q: Stratified sampling — when and why?**
A: When target classes are imbalanced. Ensures each fold/split maintains the same class proportions as the full dataset. Use `stratify=y` in train_test_split, `StratifiedKFold` for cross-validation. Prevents folds with no minority class samples.

**Q: What is cross-entropy loss?**
A: Generalization of log loss to multi-class. Measures divergence between predicted probability distribution and true labels. Used by neural networks and logistic regression. Binary cross-entropy = log loss for 2 classes.

---

## 🆕 Post-CodeSignal Interview Concepts

*These won't be on the OA, but they'll come up in the case interviews.*

**Q: A/B Testing — key components?**
A: Control group (A) vs treatment group (B). Random assignment. Define metric and minimum detectable effect BEFORE experiment. Statistical significance (p < 0.05). Watch for: sample size requirements, multiple testing correction, novelty effects, Simpson's paradox.

**Q: Causal inference vs correlation?**
A: Correlation ≠ causation. To establish causality: randomized experiments (A/B tests), instrumental variables, difference-in-differences, regression discontinuity. In consulting: always ask "is this causal or just correlated?" before recommending action.

**Q: How do you explain a model to a non-technical stakeholder?**
A: Focus on business impact, not math. "The model identifies customers 3× more likely to churn" > "The model achieves 0.85 AUC-ROC." Use analogies. Show feature importance as "what drives the prediction." Always connect to dollars or KPIs.
