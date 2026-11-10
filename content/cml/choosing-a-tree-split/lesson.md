---
title: Choosing a tree split
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A node holds $n$ samples with labels $y$. A split on one numeric feature $x$ at threshold $t$ sends $x\le t$ to the **left** child (size $n_L$) and $x>t$ to the **right** child (size $n_R$), with $n_L+n_R=n$.

- **Impurity** $I$ scores a set of labels with class proportions $p_k$:
  - Gini: $G=1-\sum_k p_k^2$ (0 when pure, largest when the classes are balanced).
  - Entropy in bits: $H=-\sum_k p_k\log_2 p_k$ (with $0\log 0=0$, from `cml/entropy-and-mutual-information`).
- **Impurity decrease** is the weighted drop in impurity:
$$\Delta(t)=I(\text{parent})-\frac{n_L}{n}\,I(\text{left})-\frac{n_R}{n}\,I(\text{right}).$$
The weights matter. A pure child that holds one sample should not count as much as a pure child that holds half the node.
- **Candidate thresholds** are the midpoints $(a+b)/2$ between consecutive distinct sorted values $a<b$. Equal values never produce a candidate.
- **The search** is greedy: at each node, take the feature and threshold with the largest $\Delta$. Nothing looks ahead.

## Worked example

Take $x=(0,1,2,3)$ and $y=(0,1,1,0)$. The parent has two classes in equal shares, so $G=0.5$ and $H=1$ bit.

| threshold $t$ | left labels | right labels | $\Delta$ (Gini) | $\Delta$ (bits) |
|---|---|---|---|---|
| 0.5 | $\{0\}$ | $\{1,1,0\}$ | $0.5-\tfrac34\cdot\tfrac49=0.1667$ | $0.3113$ |
| 1.5 | $\{0,1\}$ | $\{1,0\}$ | $0.5-\tfrac24\cdot\tfrac12-\tfrac24\cdot\tfrac12=0$ | $0$ |
| 2.5 | $\{0,1,1\}$ | $\{0\}$ | $0.1667$ | $0.3113$ |

Thresholds 0.5 and 2.5 tie exactly, so the rule returns **0.5**. Both criteria pick the same split here, which is the usual case.

## Why it works

**Sort once, scan once.** Sort the samples by $x$. Moving the threshold past one sample moves that sample from right to left, so the class counts on each side are running sums (prefix counts). Each candidate then costs $O(\text{number of classes})$. With $C$ classes, this prefix-table implementation costs $O(n\log n+nC)$ time and $O(nC)$ space. For fixed $C$, the sort dominates. Trying every pair of rows is $O(n^2)$.

**Why the decrease is never negative.** Gini and entropy are concave in the class proportions. Concavity gives $\sum_j \frac{n_j}{n}I(\text{child}_j)\le I(\text{parent})$, so a split cannot increase the weighted impurity.

**Regression version.** Use the squared error as impurity, with leaf value the mean. The decrease in sum of squared errors is
$$\mathrm{SSE}(\text{parent})-\mathrm{SSE}(L)-\mathrm{SSE}(R)=\frac{n_L n_R}{n}\,(\bar y_L-\bar y_R)^2,$$
so a regression split maximises the gap between the child means, weighted by child sizes. I checked this identity numerically on 12 random values.

**Categorical feature, binary target.** Order the categories by the proportion of the positive class, then scan them like a numeric feature. For two classes and Gini, Breiman et al. (1984) showed that the best subset split is among these orderings. With more classes, the subset search grows exponentially.

## Signal vs. noise

- **Signal:** the weighted decrease and the midpoint rule. These decide every split.
- **Signal:** Gini and entropy often give similar splits, but their rankings can differ. Choose Gini for speed, and entropy when information-theoretic language helps.
- **Noise:** the exact midpoint. Any threshold between the two values gives the same training partition, so only predictions between the two training values depend on it.
- **Noise for now:** pruning and stopping rules. They belong in `cml/building-and-pruning-a-cart-tree`.

## Where it breaks

- **Greedy blindness (XOR).** Take $x_1=(0,0,1,1)$, $x_2=(0,1,0,1)$, $y=(0,1,1,0)$. Every single split on either feature has $\Delta=0$ at the root, so a greedy search sees no gain anywhere. The label depends on the pair jointly, and a depth-two tree is needed to capture it. Greedy splitting is a local choice, and it can miss interactions.
- **Ties and duplicates.** Equal values have no midpoint, so a split between them is invalid. Floating-point noise can make a true tie look like a small gain, so an explicit tie rule is needed.
- **Rounded real measurements.** In `breast_cancer`, `worst perimeter` has 569 rows but only 514 distinct values. A valid Gini stump splits at 105.95 with decrease 0.3220. A scan that cuts between every adjacent sorted row can split two observations both equal to 106.0 and claim an impossible 0.3247 decrease. Evaluate a candidate only after the feature value changes.
- **Many-valued features.** A continuous feature offers $n-1$ thresholds. Some look good by chance, so deep trees fit noise. The same bias appears in information gain for many-category features.
- **Unweighted decrease.** Forgetting the child weights rewards splits that isolate one sample. This is a common bug.

## Interview angle

*60-second answer:* A tree picks, at each node, the feature and threshold that most reduce impurity. The decrease is the parent impurity minus the child impurities, weighted by child size. For one numeric feature, I sort the values once and scan prefix class counts, so the search costs $O(n\log n)$. Gini and entropy usually agree. The split is greedy, which is cheap and has a known blind spot: XOR-type interactions give zero gain at every single split, so a greedy search cannot see them.

*Likely follow-ups:* Why use Gini? (It is cheaper than entropy, and the two often give similar splits.) Why does a tree split on a feature that is not useful? (Under many candidate thresholds, some gains are spurious; the fix is pruning or a minimum leaf size.)

## Connections

- **Prerequisites:** `cml/entropy-and-mutual-information` (entropy and information gain), `cml/expectation-and-variance` (the variance behind the regression split).
- **Unlocks:** `cml/building-and-pruning-a-cart-tree` (recursion and stopping), `cml/random-forests-and-feature-importance` (many trees on resampled data), `cml/adaboost-and-gradient-boosting` (stumps as weak learners).
- **Lineage:** CART (Breiman et al., 1984) defined the binary, greedy, impurity-based tree. ESL chapter 9 gives the standard treatment.
- **References:** [The scikit-learn mathematical formulation](https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation) states the weighted child-impurity criterion.
