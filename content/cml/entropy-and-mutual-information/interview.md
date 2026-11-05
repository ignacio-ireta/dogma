# Interview notes: Entropy and mutual information

## Q1. What is entropy, and how does it connect to the cross-entropy loss?

**60-second answer.** Entropy is the average surprise of a distribution, $H(\mathbf p)=-\sum_i p_i\log p_i$. The base sets the unit: bits for base 2, nats for base $e$. An outcome with probability zero adds nothing, because $0\log 0=0$. Entropy is largest for a uniform distribution and zero for a certain outcome. Cross-entropy $H(\mathbf p,\mathbf q)$ is the expected code length when the code is built for the model $\mathbf q$ and the data follow $\mathbf p$. It splits into $H(\mathbf p)+D_{\mathrm{KL}}(\mathbf p\|\mathbf q)$. Training with log-loss minimises that cross-entropy, so the entropy term is a constant and the KL term is the part the model can reduce.

**Follow-ups.**
- *Why is the KL term non-negative?* Jensen's inequality applied to the concave logarithm; equality only when the two distributions match.
- *What changes if I use nats instead of bits?* Only a constant factor $\ln 2$; the optimum does not move.

## Q2. A feature has high mutual information with the label on the training set, but adds nothing on validation. What do you suspect?

**60-second answer.** Plug-in mutual information is biased upward on small samples, and the bias grows with the number of categories. With 10 categories on each side and 50 rows, independent variables still show about $(k-1)^2/(2n)\approx 0.8$ nats. A high-cardinality feature, such as an ID-like column, can look informative purely by chance. I would check the number of categories against the sample size, estimate the mutual information on a held-out split, and compare with a permutation baseline: shuffle the labels and recompute. If the shuffled value is about as large, the signal is noise.

**Follow-ups.**
- *Is the same problem present in information gain for a tree?* Yes. A many-valued feature gives many chances for a spurious gain, so trees and their importance scores need the same care.
- *What is a practical fix?* Group rare categories, use a held-out estimate, or compare against the label-shuffled baseline.

## Common mistakes

- Computing $p\log p$ over every entry, which gives `nan` for zero probabilities.
- Reporting mutual information in bits when the library returns nats, or the reverse.
- Treating zero correlation as independence. Zero correlation rules out only a linear relationship.
- Trusting a plug-in mutual information estimate on a small table without a baseline.
