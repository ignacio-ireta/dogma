# Fisher — The Use of Multiple Measurements in Taxonomic Problems (1936)

#paper #milestone #read

**Author:** Ronald A. Fisher  
**Published:** *Annals of Eugenics*, 7(2):179–188, September 1936  
**DOI:** 10.1111/j.1469-1809.1936.tb02137.x  
**Why this paper is famous:** it introduced what became **Fisher's linear discriminant** and made the **Iris dataset** immortal.

---

## Title
Classification becomes geometry.

Before this paper, statistics had tools for fitting curves, estimating means, and reasoning about variance. Fisher asked a sharper question:

> if you already know the groups, can you find the linear combination of measurements that separates them best?

That is the birth of **linear discriminant analysis** in spirit: take labeled data, project it onto a line, and choose the projection that makes classes far apart while keeping each class internally tight.

This is one of the cleanest bridges in the whole Library because it fuses:
- **probability** (class distributions),
- **linear algebra** (projections, directions, eigen-style thinking), and
- **supervised learning** (labels matter).

## Paper Reference
**Fisher, R. A. (1936).** *The Use of Multiple Measurements in Taxonomic Problems.* *Annals of Eugenics*, 7(2), 179–188.

Historical note: Fisher used measurements of **three Iris species**—50 samples each, with sepal length, sepal width, petal length, and petal width—to show that labeled biological groups can be separated by a carefully chosen linear projection.

The dataset later became one of the most famous toy datasets in all of machine learning, which is both charming and a little absurd. One flower dataset has taught half the field how classification works. 😏

## Key Ideas

### 1. A good classifier can be a projection
Fisher's move is conceptually simple and deep:
- start with several measurements per example,
- combine them linearly,
- choose the combination that makes the classes easiest to tell apart.

Instead of classifying directly in the original feature space, he asks for the **best axis for discrimination**.

That matters because it turns classification into a geometric optimization problem.

### 2. Separate the class means, compress the class spread
The heart of Fisher's criterion is:
- make the **distance between class means** large,
- make the **variance within each class** small.

In modern shorthand, Fisher's linear discriminant maximizes a ratio like:

`between-class scatter / within-class scatter`

This is the classification cousin of PCA.
- **PCA:** find directions of maximum overall variance, ignoring labels
- **LDA / Fisher discriminant:** find directions that best separate labeled groups

That distinction is foundational. One is unsupervised structure discovery. The other is supervised discrimination.

### 3. Labels change the mathematics
Without labels, you can only ask: *where is the data spread out?*
With labels, you can ask: *where do the groups differ?*

That is the conceptual hinge from descriptive multivariate statistics to actual supervised machine learning.

Fisher shows that the target direction is not just about the cloud of points, but about the **relationship between class identity and geometry**.

### 4. The model is linear, but the idea is bigger than linearity
Fisher's method gives a **linear boundary** under the usual shared-covariance assumptions.
That sounds modest now, but the underlying idea echoes through the whole field:
- choose a representation,
- make same-class points cluster,
- make different-class points separate.

That instinct later reappears in:
- logistic regression,
- SVM margins,
- metric learning,
- contrastive learning,
- embedding spaces for modern foundation models.

The tools got more powerful. The geometric instinct stayed the same.

### 5. The Iris dataset became a permanent teaching object
Fisher's paper is also historically important because it gave ML one of its canonical examples.
The Iris dataset survives because it is small enough to understand and rich enough to show something real:
- **setosa** is cleanly separable,
- **versicolor** and **virginica** are harder,
- labels reveal structure that naive clustering does not fully recover.

So this paper also became an early demonstration of the difference between:
- **supervised classification** and
- **unsupervised clustering**.

That distinction still trips people up now.

## Lineage

```text
Bayes 1763 (reasoning under uncertainty)
  └── Gauss 1809 (normal errors, covariance thinking)
      └── Pearson 1901 (best-fit directions / PCA)
          └── Fisher 1936 (best discriminating direction with labels)
              ├── classical discriminant analysis
              ├── multivariate classification
              ├── logistic regression comparisons
              ├── support vector machines
              └── modern representation learning
```

More specifically:
- **Built on:** Gauss-style distribution/error thinking, [[../../Mathematics/06-Probability/Gauss - Least Squares and Normal Errors 1809|Gauss]] for covariance intuition, and [[../../Mathematics/07-Multivariate-Statistics/Pearson - Principal Components 1901|Pearson 1901]] for the geometry of directions in data
- **Contrasts with:** [[../02-Linear-Algebra/Eckart-Young Best Low-Rank Approximation 1936|Eckart–Young 1936]] and PCA-style dimensionality reduction, because Fisher uses labels instead of only reconstruction structure
- **Leads to:** quadratic discriminant analysis, logistic regression as a competing probabilistic classifier, kernel discriminants, SVMs, and the broader tradition of supervised feature extraction
- **Connected to:** the Iris dataset, multivariate Gaussian assumptions, covariance matrices, Bayes classifiers, and modern embedding-space separation

## Why It Matters

### For AI/ML
This note closes a very real hole in the Library: the **Classification** section existed, but its founding paper was missing.

Fisher matters because classification is one of the oldest core tasks in ML:
- spam or not spam,
- fraud or not fraud,
- churn or not churn,
- disease class A vs B,
- eligible lead vs low-probability lead.

And Fisher gives the field one of its first elegant answers: **project the data into a direction where the labels become easy to separate**.

That is still a live idea, even when hidden inside much larger models.

### For mathematics
This is where multivariate statistics stops merely summarizing data and starts **using geometry to make decisions**.

Pearson asked for the best low-dimensional summary.
Fisher asks for the best low-dimensional separation.

That is a beautiful fork in the lineage:
- one branch becomes compression and representation,
- the other becomes classification and decision.

## The Subtle Historical Truth
Fisher's original setup is not identical to every modern textbook version of LDA.
The modern matrix form—with explicit scatter matrices, generalized eigenvalue formulations, and multiclass extensions—was developed and standardized later.

But the original spark is already there:
**use labels to find the projection that best separates groups.**

That is the enduring move.

## Leads To
- [[../_MOC|AI/ML Lineage]]
- [[../../Mathematics/07-Multivariate-Statistics/Pearson - Principal Components 1901|Pearson 1901]]
- [[../../Mathematics/06-Probability/Gauss - Least Squares and Normal Errors 1809|Gauss 1809]]
- Future note: Naive Bayes as a different probability-first classification tradition
- Future note: Logistic regression as the modern statistical classifier that often displaces LDA in practice
- Future note: Support Vector Machines as margin-based linear separation

## References
- Fisher, R. A. (1936). *The Use of Multiple Measurements in Taxonomic Problems.* *Annals of Eugenics*, 7(2), 179–188.
- Wikipedia summary pages used for quick factual orientation on the historical role of Fisher's discriminant and the Iris dataset.
- Secondary statistical summaries of LDA's modern formulation used only as conceptual cross-checks.

## Status
- [x] Note created (May 2, 2026)
- [ ] Add a proper AI/ML probability note so the Fisher note's probability links stop pointing into the future
- [ ] Add Logistic Regression note as the next natural classifier in the chain
- [ ] Add SVM note to complete the linear-classification arc
- [ ] Consider a Naive Bayes note for the probability-first branch of classification

---

*Fisher's move was simple and enormous: once labels exist, geometry can become judgment.*
