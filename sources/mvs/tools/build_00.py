"""Generate 00_orientation_and_method_map.ipynb."""

from nbbuild import (SETUP_CELL, build, close, code, header, learn, md,
                     practice, predict, retrieve, answer_key_banner)

cells = []

cells.append(header(
    title="00 · Orientation and Method Map",
    est_time="45–60 min",
    objectives=[
        "Name the five exam methods and say, for each, whether it is supervised or unsupervised.",
        "Classify a research question by its **goal verb**: predict, reduce, find-latent, group, or discriminate.",
        "Walk a short decision tree from *question + data* to a defensible method.",
        "Recite the shared analysis workflow used in every later notebook.",
    ],
    prereq="Estadística Descriptiva e Inferencial (mean, variance, correlation, a hypothesis test).",
    mvp="Do **🔁 Retrieve**, read the method-map table, and complete the **scenario → method** drill in 🛠️ Practice. That alone fixes the single most-tested skill: choosing a method.",
    extra="\n\nThis notebook has no math to grind. Its job is to give you a **stable map** so that "
          "nothing later feels like it appears from nowhere. Every method you meet will hang on this map.",
))

cells.append(code(SETUP_CELL))

# ---------------------------------------------------------------- Retrieve
cells += retrieve([
    "In one sentence, what does the **variance** of a variable measure?",
    "If two variables have correlation 0, are they necessarily independent? Why or why not?",
    "A hypothesis test gives p = 0.02. What does that number mean, in words?",
    "What is the difference between a **population parameter** and a **sample statistic**?",
], note="*(These come from your previous course. They are the soil everything else grows in.)*")

# ---------------------------------------------------------------- Predict
cells += predict([
    "The course has five methods: multiple regression, PCA, factor analysis, cluster analysis, discriminant analysis. "
    "Which of these need a **labelled outcome / known groups**, and which work on **unlabelled** data? Sort them into two piles.",
    "PCA and factor analysis both produce a smaller set of new variables. Guess one way they **differ in purpose**.",
    "Clustering and discriminant analysis both deal with groups. Guess the key difference between them.",
])

# ---------------------------------------------------------------- Learn
cells.append(learn("Learn the system — five methods, five verbs"))

cells.append(md(r"""
Every multivariate method in this course is answering one of **five questions**. If you can name the
verb, you are most of the way to choosing the method.

| Goal **verb** | Question it answers | Method (this course) | Supervised? | Output |
|---|---|---|---|---|
| **Predict** | "How does a numeric outcome depend on several predictors?" | **Multiple regression** | Supervised | coefficients, fitted values, R² |
| **Reduce** | "Can I describe many correlated variables with a few new axes?" | **PCA** | Unsupervised | components, scores, loadings, % variance |
| **Find latent** | "Is there a small number of *hidden factors* generating these correlations?" | **Factor analysis** | Unsupervised | loadings, communalities, uniquenesses |
| **Group** | "Do the observations fall into natural clusters I did *not* know in advance?" | **Cluster analysis** | Unsupervised | cluster labels, dendrogram |
| **Discriminate** | "Given *known* groups, what separates them and how do I classify a new case?" | **Discriminant analysis / logistic** | Supervised | discriminant axes, boundaries, confusion matrix |

**Supervised** = you have a target/label you are trying to explain or predict (regression, discriminant).
**Unsupervised** = no target; you are describing structure in the X's alone (PCA, factor analysis, clustering).
"""))

cells.append(md(r"""
#### The map, as a picture

We will place the five methods on two axes:

- **horizontal:** is there a target/label? (unsupervised ⟶ supervised)
- **vertical:** what kind of structure? (relationships among *variables* ⟶ structure among *observations*)

Run the cell and keep the picture in your head — every later notebook lives somewhere on it.
"""))

cells.append(code(r"""
fig, ax = plt.subplots(figsize=(7.5, 5.2))
# (x = supervised-ness 0..1, y = variable-structure 0 .. observation-structure 1)
methods = {
    "Multiple\nregression":      (0.92, 0.15, P.FIT),
    "Discriminant /\nlogistic":  (0.92, 0.85, P.HIGHLIGHT),
    "PCA":                       (0.10, 0.20, P.COMPONENT),
    "Factor\nanalysis":          (0.10, 0.42, P.COMPONENT),
    "Cluster\nanalysis":         (0.10, 0.85, P.ORIGINAL),
}
for name, (x, y, c) in methods.items():
    ax.scatter(x, y, s=900, color=c, alpha=0.30, edgecolor=c, linewidth=2)
    ax.text(x, y, name, ha="center", va="center", fontsize=9, weight="bold")

ax.axvline(0.5, color=P.NEUTRAL, ls="--", lw=1)
ax.text(0.5, 1.04, "no target  ◀  unsupervised | supervised  ▶  has a target",
        ha="center", fontsize=9, color=P.NEUTRAL)
ax.set_yticks([0.15, 0.85], ["structure among\nVARIABLES", "structure among\nOBSERVATIONS"])
ax.set_xticks([])
ax.set_xlim(-0.05, 1.05); ax.set_ylim(0.0, 1.12)
ax.set_title("Q: where does each method live? (memorise this map)")
ax.grid(False)
plt.show()
"""))

cells.append(md(r"""
#### A decision tree you can actually walk

```
START: What is my goal?
│
├─ I have a NUMERIC outcome to explain/predict ............ → MULTIPLE REGRESSION   (Unit 1)
│
├─ I have KNOWN groups and want to separate / classify .... → DISCRIMINANT or LOGISTIC (Unit 5)
│
└─ I have NO target, only a table of variables
   │
   ├─ Goal is to COMPRESS / re-express the variables
   │     ├─ I want new axes that capture maximum variance .. → PCA               (Unit 2)
   │     └─ I believe a few HIDDEN factors cause the
   │        correlations, and want to model/name them ..... → FACTOR ANALYSIS    (Unit 3)
   │
   └─ Goal is to GROUP the observations into clusters ...... → CLUSTER ANALYSIS  (Unit 4)
```

Two traps this tree protects you from:
- **PCA vs factor analysis** — same shape of output (fewer new variables), *different question*. PCA re-describes total variance; factor analysis models only *shared* variance as caused by latent factors. (Whole notebook on this later.)
- **Clustering vs discriminant** — both about groups, but clustering *discovers* groups (unsupervised) while discriminant analysis *uses known* groups (supervised).
"""))

cells.append(md(r"""
#### The shared workflow (every later notebook follows it)

> **question → data → assumptions → fit → diagnostics → interpretation → report**

| Step | What you actually do | Why it is on the exam |
|---|---|---|
| **question** | State the goal as one of the five verbs. | Picking the method is graded. |
| **data** | Check shape: rows = observations, cols = variables; types; missing; scale. | Wrong shape/scale silently breaks methods. |
| **assumptions** | Name them *before* fitting (e.g. linearity, equal covariance, scaling). | "Choose and defend" needs named assumptions. |
| **fit** | Run the method (Python). | Practical part. |
| **diagnostics** | Check residuals / variance explained / silhouette / confusion. | Output interpretation is graded. |
| **interpretation** | Translate numbers into sentences about the question. | The point of the whole course. |
| **report** | 3–5 sentences: what you did, what you found, limitations. | Explicit course objective #5. |
"""))

# ---------------------------------------------------------------- Practice
cells.append(practice())

cells.append(md(r"""
**Drill 1 (Recognize · choose the method).** For each scenario, name the method and the goal-verb.
Write answers in the cell below; check against the key.

1. A bank has 12 financial ratios for 400 firms and wants to see whether the firms fall into a few
   natural types it had not defined in advance.
2. A biologist measured 4 body dimensions on penguins of 3 *known* species and wants a rule to
   classify a new penguin to its species.
3. An economist wants to explain a country's life expectancy from 8 numeric development indicators.
4. A psychologist has 25 questionnaire items and suspects they reflect ~4 underlying traits.
5. An engineer has 30 correlated sensor readings and wants 3 new variables that retain most of the
   information for a dashboard.
"""))
cells.append(md("**YOUR ANSWERS (Drill 1):**\n\n1. \n2. \n3. \n4. \n5. "))

cells.append(md(r"""
**Drill 2 (Compute · no formula, just sorting).** Complete the code: assign each method to
`"supervised"` or `"unsupervised"`. The assert tells you if you have them all right.
"""))
cells.append(code(r"""
# TODO: fill each value with "supervised" or "unsupervised".
# HINT 1 (concept): supervised methods need a target column / known labels.
# HINT 2: regression predicts a number; discriminant uses known groups.
# HINT 3: the other three describe the X's with no target at all.
kind = {
    "multiple_regression": "",   # <- fill
    "pca":                 "",   # <- fill
    "factor_analysis":     "",   # <- fill
    "cluster_analysis":    "",   # <- fill
    "discriminant":        "",   # <- fill
}

expected = {
    "multiple_regression": "supervised",
    "pca": "unsupervised",
    "factor_analysis": "unsupervised",
    "cluster_analysis": "unsupervised",
    "discriminant": "supervised",
}
def _verify():
    assert kind == expected, "Not yet — compare each entry to the map table above."
    print("Correct: you have the supervised/unsupervised split.")
selfcheck(all(kind.values()), _verify)
"""))

cells.append(md(r"""
**Drill 3 (Choose-and-defend · the one that matters most).** A nutrition researcher has, for 200
people, 9 numeric nutrient-intake variables **and** a label `diet_group ∈ {vegan, vegetarian, omnivore}`.

She asks: *"Which nutrients best separate the three diet groups, and can I classify a new person?"*

In 2–3 sentences: name the method, name **one** assumption you would check first, and name the
**output** you would read to answer her question. Then say which method you would use *instead* if she
had **no** diet labels.
"""))
cells.append(md("**YOUR ANSWER (Drill 3):**\n\n"))

cells.append(md(r"""
**Drill 4 (Diagnose the flawed plan).** A student writes:

> "I have 50 firms and 8 financial ratios. I'll run multiple regression to find the natural groups
> of firms."

What is wrong with this plan, in one sentence — and what should they do instead?
"""))
cells.append(md("**YOUR ANSWER (Drill 4):**\n\n"))

# ---------------------------------------------------------------- Close
cells.append(close())

cells.append(md(r"""
**Confidence map.** Rate your current confidence 1 (lost) – 5 (could teach it) for each unit.
Be honest — this just tells you where to spend time. Edit the numbers and run.
"""))
cells.append(code(r"""
confidence = {
    "Linear algebra / covariance": 3,
    "Multiple regression":         3,
    "PCA":                         2,
    "Factor analysis":             2,
    "Cluster analysis":            2,
    "Discriminant / logistic":     2,
}
labels = list(confidence)
vals = [confidence[k] for k in labels]
colors = [P.RESIDUAL if v <= 2 else (P.COMPONENT if v == 3 else P.FIT) for v in vals]

fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.barh(labels, vals, color=colors)
ax.set_xlim(0, 5); ax.set_xlabel("confidence (1 lost … 5 could teach it)")
ax.set_title("Q: where should I spend my study time? (red = start here)")
ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
plt.tight_layout(); plt.show()
print("Red bars are your highest-leverage study targets.")
"""))

cells.append(md(r"""
**Exam-style closing (5 sentences).** Without scrolling up, write 5 sentences that a) list the five
methods, b) split them into supervised vs unsupervised, and c) give the goal-verb of each. This is a
classic opening exam question.
"""))
cells.append(md("**YOUR 5 SENTENCES:**\n\n"))

cells.append(md(r"""
#### 🃏 Course-map card — fill from memory

| Goal verb | Method | Supervised? |
|---|---|---|
| Predict a number |  |  |
| Reduce / compress variables |  |  |
| Model hidden factors |  |  |
| Discover groups |  |  |
| Separate known groups |  |  |
"""))

cells.append(md(r"""
**Transfer question.** Canonical correlation analysis (not in your exam) relates *two whole sets* of
variables to each other. Using only the map's logic, would you expect it to be supervised or
unsupervised, and is its goal closer to *reduce* or *predict*? (One sentence — reasoning matters more
than being right.)
"""))
cells.append(md("**YOUR ANSWER:**\n\n"))

cells.append(md(r"""
**Suggested Anki prompts**
- Front: *Five goal-verbs of multivariate analysis?* → Back: predict, reduce, find-latent, group, discriminate.
- Front: *Supervised methods in this course?* → Back: multiple regression; discriminant/logistic.
- Front: *PCA vs factor analysis, one-line difference?* → Back: PCA re-describes total variance; FA models shared variance as caused by latent factors.
- Front: *Clustering vs discriminant?* → Back: clustering discovers groups (unsupervised); discriminant uses known groups (supervised).
- Front: *The shared workflow?* → Back: question → data → assumptions → fit → diagnostics → interpretation → report.
"""))

# ---------------------------------------------------------------- Answer key
cells.append(answer_key_banner())

cells.append(md(r"""
**🔁 Retrieve**
1. Variance measures the average squared distance of values from their mean — the spread.
2. No. Zero correlation rules out a *linear* association only; variables can be strongly but
   non-linearly dependent (e.g. Y = X²) and still have correlation 0.
3. If the null hypothesis were true, there is a 2% probability of seeing data at least this extreme;
   p is *not* the probability the null is true.
4. A parameter describes the whole population (fixed, usually unknown); a statistic is computed from
   a sample and estimates a parameter (varies sample to sample).

**🔮 Predict** (model answers)
- Need labels/target: multiple regression, discriminant analysis. Unlabelled: PCA, factor analysis, clustering.
- PCA: pragmatic compression capturing maximum variance. FA: explanatory model positing hidden factors that *cause* the correlations.
- Clustering discovers unknown groups; discriminant analysis uses known groups to build a separation/classification rule.

**🛠️ Practice**
- *Drill 1:* (1) cluster analysis — group; (2) discriminant analysis — discriminate; (3) multiple
  regression — predict; (4) factor analysis — find latent; (5) PCA — reduce.
- *Drill 2:* see the `expected` dict — regression & discriminant supervised; PCA, FA, clustering unsupervised.
- *Drill 3:* **Discriminant analysis** (groups are known). Check e.g. approximately equal
  group covariances (LDA assumption) and roughly multivariate-normal predictors. Read the
  **discriminant function coefficients / loadings** to see which nutrients separate groups, and the
  **confusion matrix / cross-validated accuracy** to judge classification. With *no* labels you would
  instead use **cluster analysis** (you'd be discovering diet groups, not separating known ones).
- *Drill 4:* Multiple regression needs a numeric outcome; "natural groups" with no target is an
  unsupervised grouping task → use **cluster analysis** (or PCA first to compress, then cluster).

**✅ Close — course-map card:** predict→multiple regression (supervised); reduce→PCA (unsupervised);
hidden factors→factor analysis (unsupervised); discover groups→cluster analysis (unsupervised);
separate known groups→discriminant/logistic (supervised).

**Transfer:** Canonical correlation is *unsupervised* in the sense of having no single target, but its
goal is closer to *predict/relate* — it finds linear combinations of one variable set most correlated
with linear combinations of another. (Any coherent reasoning earns the point.)
"""))

build("00_orientation_and_method_map.ipynb", cells)
