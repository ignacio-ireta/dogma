"""Generate 05_cluster_analysis.ipynb."""

from nbbuild import (METHOD_CARD_BLANK, SETUP_CELL, answer_key_banner, build,
                     close, code, header, learn, md, practice, predict, retrieve)

cells = []

cells.append(header(
    title="05 · Cluster Analysis",
    est_time="100–130 min",
    objectives=[
        "Explain why scaling and distance choice change clusters, and contrast Euclidean / standardized / Mahalanobis distance.",
        "Run and read hierarchical clustering (linkages, dendrograms) and k-means (objective, initialization, local optima).",
        "Choose k with the elbow, silhouette, and domain usefulness — and accept that 'clusters' need not be 'truth'.",
        "Profile clusters and avoid the two classic traps: clustering unscaled variables and reading cluster labels as ordinal.",
    ],
    prereq="01 (standardization, Mahalanobis distance). 03 (scaling traps).",
    mvp="Do 🔁 Retrieve, the **scaling trap** cell (raw vs standardized penguins), and the **choose-k** cell (elbow + silhouette). "
        "Those are the two graded ideas: scale first, and justify k.",
    extra="\n\nClustering is **unsupervised**: there is no answer key in the data. The algorithm will *always* return "
          "clusters — your job is to decide whether they mean anything.",
))

cells.append(code(SETUP_CELL))

# ---------------------------------------------------------------- Retrieve
cells += retrieve([
    "From 01: write the formula for Mahalanobis distance and say what it corrects for vs plain Euclidean.",
    "From 03/01: why does standardizing variables matter before a variance- or distance-based method?",
    "From 00: clustering and discriminant analysis both involve groups — what is the key difference?",
    "From 03: name one method for choosing 'how many' (components) that also transfers to choosing k clusters.",
])

# ---------------------------------------------------------------- Predict
cells += predict([
    "k-means minimizes a specific quantity over cluster assignments and centres. Predict what that quantity is.",
    "Penguin body mass is in grams (~4000) and bill depth in mm (~18). Predict which variable dominates an *unscaled* Euclidean clustering, and whether that is meaningful.",
    "You run k-means twice with different random seeds and get different clusters. Predict why, and one fix.",
    "A dendrogram merges everything eventually into one cluster. Predict how you decide how many clusters to 'cut' it into.",
])

# ---------------------------------------------------------------- Learn
cells.append(learn("Learn the system — clustering as a system"))

cells.append(md(r"""
**Research question.** "Do my observations fall into natural groups I did **not** define in advance?"

**Input shape.** $X$: $n\times p$ numeric (no labels). **Unsupervised**. Output = a cluster label per observation.

**Everything rides on distance.** Most methods cluster points that are "close". So *scale* and *distance*
are modelling choices, not details:
- **Euclidean** on raw data → large-scale variables dominate (the trap).
- **Standardized Euclidean** → every variable contributes comparably (usual default).
- **Mahalanobis** → accounts for correlations/covariance (distance after whitening).

**Two families.**

| | Hierarchical | k-means (partitioning) |
|---|---|---|
| Idea | iteratively merge nearest clusters → a tree | choose $k$; alternate *assign to nearest centre* / *recompute centres* |
| Choose k | **after**, by cutting the dendrogram | **before**, as input |
| Objective | depends on **linkage** | minimize within-cluster sum of squares $\sum_k\sum_{i\in C_k}\|x_i-\mu_k\|^2$ |
| Output | dendrogram + any cut | labels + centroids |
| Notes | deterministic; $O(n^2)$ memory | fast; **random init → local optima** (use several inits/seeds) |

**Linkage (hierarchical) changes everything.** *single* = nearest pair (chains, finds straggly shapes),
*complete* = farthest pair (compact balls), *average*, *ward* = merge that least increases variance (compact,
the common default). Read a dendrogram **bottom-up**: merge height = how dissimilar two clusters were when joined.

**Choosing k.** **Elbow** (inertia vs k — look for the bend), **silhouette** (−1…1; higher = better separated),
**stability** across seeds/subsamples, and **domain usefulness**. No rule overrides "does this grouping help
answer the question?".

**Diagnostics / profiling.** Cluster sizes; per-cluster feature means (profile each cluster in words);
silhouette per point; sensitivity to scaling/linkage/seed.

**Common traps.** (1) **Not scaling** → biggest-unit variable defines the clusters. (2) Treating labels
`0,1,2` as **ordinal** — they are arbitrary names; cluster 2 is not "more" than cluster 1. (3) Reading
clusters as ground truth. (4) One outlier swallowing a whole cluster.

**Neighbours.** **Discriminant analysis** (06): same "groups" topic but *supervised* (groups known in
advance). **PCA** (03): often run *before* clustering to denoise/visualize.
"""))

# ---------------------------------------------------------------- Practice
cells.append(practice())

cells.append(md(r"""
**P1 (Compute · k-means from scratch).** Implement Lloyd's algorithm (assign → recompute) and confirm it
matches `sklearn.KMeans` started from the **same** initial centres. This shows the objective is just
within-cluster squared distance.
"""))
cells.append(code(r"""
from sklearn.cluster import KMeans
blob = np.vstack([RNG.normal([0, 0], 0.5, (60, 2)),
                  RNG.normal([3, 3], 0.5, (60, 2)),
                  RNG.normal([0, 3], 0.5, (60, 2))])
k = 3
init_idx = RNG.choice(len(blob), k, replace=False)
C0 = blob[init_idx].copy()

# TODO: complete one assignment step and one update step inside the loop.
def kmeans_lloyd(X, C0, n_iter=50):
    C = C0.copy()
    for _ in range(n_iter):
        # HINT 1 (concept): assign each point to its nearest centre (squared Euclidean).
        # HINT 2 (formula): D[i,j] = ||x_i - c_j||^2 ; labels = argmin over j.
        # HINT 3 (code): D = ((X[:,None,:]-C[None,:,:])**2).sum(2); labels = D.argmin(1)
        D = None          # <- replace: n x k squared-distance matrix
        labels = None     # <- replace: nearest-centre index per point
        newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j]
                         for j in range(len(C))])
        if np.allclose(newC, C):
            C = newC; break
        C = newC
    inertia = float(((X - C[labels])**2).sum())
    return labels, C, inertia

def _v():
    _, _, inertia_hand = kmeans_lloyd(blob, C0)
    sk = KMeans(n_clusters=k, init=C0, n_init=1, max_iter=50, random_state=0).fit(blob)
    assert_close(inertia_hand, sk.inertia_, tol=1e-6, name="Lloyd inertia vs sklearn (same init)")
selfcheck(False, _v, blank_msg="Fill the two lines in kmeans_lloyd, set the selfcheck flag True, and re-run.")
"""))

cells.append(md(r"""
**P2 (Choose-and-defend · the scaling trap — failure case #1).** Cluster penguins on raw features, then on
standardized features, and measure agreement with the (held-out) species labels. Watch raw clustering
collapse onto body mass.
"""))
cells.append(code(r"""
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

pen = data.load_penguins_df()
feat = data.PENGUIN_FEATURES
Xp = pen[feat].values
species = pen["species"].values

km_raw = KMeans(n_clusters=3, n_init=10, random_state=7).fit(Xp)
Xs = StandardScaler().fit_transform(Xp)
km_std = KMeans(n_clusters=3, n_init=10, random_state=7).fit(Xs)

print(f"agreement with species (adjusted Rand)  raw: {adjusted_rand_score(species, km_raw.labels_):.2f}")
print(f"agreement with species (adjusted Rand)  std: {adjusted_rand_score(species, km_std.labels_):.2f}")

# which feature dominates raw clustering? compare centroid spread per feature (in raw units)
spread_raw = pd.Series(km_raw.cluster_centers_.std(0), index=feat)
print("\nspread of RAW centroids across features (bigger = drives the split):")
print(spread_raw.round(1).to_string())
"""))
cells.append(md(r"""
**YOUR ANSWER (P2):** (a) Which clustering matches species better, raw or standardized? (b) Which single
feature dominates the raw clustering and why? (c) State the one-line rule this demonstrates.
"""))

cells.append(md(r"""
**P3 (Interpret · linkage changes the tree — rerunnable).** Change `linkage_method` among
`'single','complete','average','ward'` and rerun. Single linkage "chains"; ward gives compact clusters.
"""))
cells.append(code(r"""
from scipy.cluster.hierarchy import linkage, fcluster

linkage_method = "ward"   # <- try 'single', 'complete', 'average', 'ward'

sub = pen.sample(40, random_state=1)
Xsub = StandardScaler().fit_transform(sub[feat].values)
Z = linkage(Xsub, method=linkage_method)
plots.dendrogram_plot(Z, labels=sub["species"].values, color_threshold=0.7 * Z[:, 2].max(),
                      question=f"Q: how do penguins merge under '{linkage_method}' linkage?")
plt.show()
print("Tip: with 'single' you often see one big chain; 'ward' usually shows 3 tidy blocks ~ the species.")
"""))

cells.append(md(r"""
**P4 (Choose-and-defend · how many clusters? elbow + silhouette).** Two standard k-selection diagnostics on
standardized penguins.
"""))
cells.append(code(r"""
from sklearn.metrics import silhouette_score
ks = range(2, 8)
inertia = [KMeans(n_clusters=k, n_init=10, random_state=7).fit(Xs).inertia_ for k in ks]
sil = [silhouette_score(Xs, KMeans(n_clusters=k, n_init=10, random_state=7).fit_predict(Xs)) for k in ks]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
a1.plot(list(ks), inertia, "-o", color=P.ORIGINAL); a1.set_xlabel("k"); a1.set_ylabel("inertia")
a1.set_title("Q: where is the elbow?")
a2.plot(list(ks), sil, "-o", color=P.FIT); a2.set_xlabel("k"); a2.set_ylabel("silhouette")
a2.set_title("Q: which k is best separated?")
plt.tight_layout(); plt.show()
print(f"best silhouette at k = {list(ks)[int(np.argmax(sil))]}  (penguins have 3 species — does k agree?)")
"""))
cells.append(md("**YOUR ANSWER (P4):** Which k do the elbow and silhouette suggest, and does that match the "
                "biology (3 species)? If they disagreed, which would you trust and why?"))

cells.append(md(r"""
**P5 (Interpret · cluster profiling).** A cluster is only useful if you can describe it. Profile the k=3
standardized solution by per-cluster feature means (in original units).
"""))
cells.append(code(r"""
pen2 = pen.copy(); pen2["cluster"] = km_std.labels_
profile = pen2.groupby("cluster")[feat].mean().round(1)
print("Per-cluster average measurements (original units):")
print(profile.to_string())
print("\ncross-tab cluster vs species (clusters are NOT labelled with species — we compare after):")
print(pd.crosstab(pen2["cluster"], pen2["species"]).to_string())
"""))
cells.append(md("**YOUR ANSWER (P5):** Describe each cluster in one phrase from its profile (e.g. 'large, "
                "deep-billed birds'). Then: the cross-tab shows cluster 0/1/2 align with species — but is "
                "cluster **2** 'more' than cluster **1**? Explain the label trap."))

cells.append(md(r"""
**P6 (Diagnose · clusters ≠ truth — failure case #2).** We force k=3 on data with **no** real cluster
structure (one uniform blob). The algorithm still returns three confident, meaningless clusters.
"""))
cells.append(code(r"""
nostruct = RNG.uniform(-1, 1, size=(300, 2))
km = KMeans(n_clusters=3, n_init=10, random_state=7).fit(nostruct)
sil_bad = silhouette_score(nostruct, km.labels_)

fig, ax = plt.subplots(figsize=(4.6, 4.4))
for j in range(3):
    m = km.labels_ == j
    ax.scatter(nostruct[m, 0], nostruct[m, 1], s=12, color=P.group_color(j), label=f"cluster {j}")
ax.set_title(f"Q: are these 'clusters' real?  (silhouette={sil_bad:.2f})"); ax.legend(fontsize=8)
plt.show()
print(f"Silhouette {sil_bad:.2f} is near 0 -> no real separation. The clusters are an artefact of forcing k=3.")
"""))

# ---------------------------------------------------------------- Close
cells.append(close())

cells.append(md(r"""
**Exam-style interpretation (5 sentences).** Report a penguin clustering: state the question, say you
standardized and why, give the method (k-means or hierarchical) and how you chose k, describe what the
clusters represent (profile), and state one limitation (e.g. clusters ≠ species ground truth).
"""))
cells.append(md("**YOUR 5 SENTENCES:**\n\n"))

cells.append(md("#### 🃏 Method card — Cluster analysis (fill from memory)\n\n" + METHOD_CARD_BLANK))

cells.append(md(r"""
**Transfer question.** Suppose you actually *did* know the penguin species in advance and wanted a rule to
classify new birds. Which method (next notebook) would you switch to, and what changes about the problem
(one sentence)?
"""))
cells.append(md("**YOUR ANSWER:**\n\n"))

cells.append(md(r"""
**Suggested Anki prompts**
- Front: *k-means objective?* → Back: minimize within-cluster sum of squared distances to centroids (inertia).
- Front: *Why scale before clustering?* → Back: distance is unit-sensitive; unscaled, the largest-range variable dominates.
- Front: *Reading a dendrogram?* → Back: bottom-up; merge height = dissimilarity at which clusters joined; cut to choose #clusters.
- Front: *Single vs ward linkage?* → Back: single chains/elongated clusters; ward gives compact, variance-minimizing clusters.
- Front: *Two cluster-label traps?* → Back: labels are nominal (not ordinal); clusters are not guaranteed to be real/ground truth.
"""))

# ---------------------------------------------------------------- Answer key
cells.append(answer_key_banner())
cells.append(md(r"""
**🔁 Retrieve**
1. $d=\sqrt{(x-\mu)^\top S^{-1}(x-\mu)}$; it corrects for different variances and correlations among variables (Euclidean assumes equal, independent scales).
2. Distance/variance methods are unit-sensitive; without standardizing, a large-range variable dominates the result.
3. Clustering discovers unknown groups (unsupervised); discriminant analysis uses known groups (supervised).
4. Parallel analysis / scree-style elbow transfers; for clusters we use the inertia elbow and silhouette.

**🔮 Predict** (model answers)
1. Within-cluster sum of squared distances to the cluster centres (inertia).
2. **body_mass_g** dominates (range in thousands) — not meaningful; it just reflects units.
3. Random initialization → different local optima; fix with multiple inits (`n_init`) / fixed seed.
4. Cut the tree at a chosen height (or for a chosen number of branches), guided by big gaps in merge heights + domain sense.

**🛠️ Practice**
- *P1:* `D = ((X[:,None,:]-C[None,:,:])**2).sum(2); labels = D.argmin(1)`; then set the selfcheck flag True — inertia matches sklearn from the same init.
- *P2:* (a) **standardized** matches species far better (higher adjusted Rand). (b) **body_mass_g** dominates raw clustering (centroid spread in the hundreds/thousands vs single digits for mm features). (c) "Standardize features before any distance-based clustering."
- *P3:* 'ward'/'complete' give ~3 compact blocks aligning with species; 'single' tends to chain.
- *P4:* Both point to **k=3**, matching the 3 species; if they disagreed, prefer the one that yields a domain-useful, stable grouping (often silhouette + interpretability over the elbow's eyeballing).
- *P5:* e.g. cluster of "large, heavy, long-flippered" (Gentoo-like) etc. The label trap: clusters `0,1,2` are **nominal names**; cluster 2 is not greater than cluster 1 — never average or order them.
- *P6:* Silhouette ≈ 0 → the three "clusters" just partition a structureless blob; forcing k always returns clusters, real or not.

**Transfer:** Switch to **discriminant analysis / logistic** (Unit 5). The problem becomes **supervised**:
you now have known labels to *separate and predict*, so you can train and validate a classification rule.
"""))

build("05_cluster_analysis.ipynb", cells)
