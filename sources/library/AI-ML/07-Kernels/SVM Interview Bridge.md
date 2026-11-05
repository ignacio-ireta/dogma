# SVM Interview Bridge

*Short synthesis from the SVM lineage into practical interview judgment.*

**Source notes:**
- `AI-ML/07-Kernels/Boser Guyon Vapnik Optimal Margin Classifiers 1992.md`
- `AI-ML/07-Kernels/Cortes Vapnik Support-Vector Networks 1995.md`
- `LINEAGE.md` -> AI/ML kernel branch

---

## The one-sentence bridge

**Boser, Guyon, and Vapnik explain why margin matters; Cortes and Vapnik explain why margin needs forgiveness. In interviews, that becomes a model-choice story: prefer the simplest classifier whose geometry matches the data, then regularize the parts where reality is noisy.**

## Interview answer pattern

If asked **"When would you use an SVM?"**, answer in this order:

1. **Data shape:** medium-sized supervised classification, often with well-engineered numeric features or embeddings.
2. **Geometry:** the question is whether classes can be separated by a useful margin in the original or kernel-induced feature space.
3. **Regularization:** soft margins tolerate overlap and label noise through the `C` tradeoff.
4. **Scaling caveat:** kernel SVMs can become expensive on very large datasets because they depend on pairwise similarities and support vectors.
5. **Modern caveat:** if representation learning is the hard part, a neural model or pretrained embedding pipeline may be better; if the data are messy enterprise tables, boosted trees are often the stronger first baseline.

The adult answer is not "SVMs are old." It is: **SVMs are strong when the feature geometry is already meaningful; boosted trees or deep models are stronger when the representation or tabular interactions are the real problem.**

## Useful comparison table

| Question | SVM answer | XGBoost answer |
|---|---|---|
| Core bias | Wide separating margin | Additive error-correcting trees |
| Nonlinearity | Kernel-defined feature space | Tree splits and interactions |
| Noise handling | Soft margin via `C` | Regularization, depth, shrinkage, sampling |
| Strength | Medium data, clean features, strong geometry | Messy tabular data, missing values, thresholds |
| Interview trap | Ignoring scaling and feature scaling | Treating feature importance as causal truth |

## Polished 45-second answer

> "An SVM is a margin-based classifier: it chooses the separating boundary with the largest safety margin, and the soft-margin version allows controlled violations when classes overlap or labels are noisy. I would consider it when I have a medium-sized supervised classification problem with meaningful numeric features or embeddings. I would be more careful on very large or messy tabular enterprise data, because kernel methods can be expensive and boosted trees often capture thresholds, missingness, and interactions more naturally. For the project, that is why XGBoost was the better default, but SVM is still a useful classical baseline and a clean way to discuss regularization."

## Source path citations

- `AI-ML/07-Kernels/Boser Guyon Vapnik Optimal Margin Classifiers 1992.md` - maximum-margin classifier, support vectors, and kernel trick.
- `AI-ML/07-Kernels/Cortes Vapnik Support-Vector Networks 1995.md` - soft margins, slack, `C`, and robustness to non-separable data.
- `corpus/raw/library-papers/cortes-vapnik-support-vector-networks-1995.pdf` - primary source PDF for soft-margin SVM.
