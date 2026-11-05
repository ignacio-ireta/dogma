# Interview notes: Choosing a tree split

## Q1. How does a decision tree choose where to split a numeric feature?

**60-second answer.** At a node, for each feature I consider the midpoints between consecutive distinct sorted values. Each candidate has a weighted impurity decrease: the parent's impurity minus the child impurities, with each child weighted by its share of the rows. Gini or entropy scores impurity. I keep the candidate with the largest decrease. Sorting once and scanning running class counts makes the search $O(n\log n)$ per feature. Ties go to the smallest threshold, so the result is deterministic. The search is greedy: it never looks ahead to what the next split could achieve.

**Follow-ups.**
- *Why weight the children?* Without weights, a split that isolates one sample into a pure child looks as good as a split that cleanly separates half the node.
- *Why are equal values skipped?* They have no midpoint, so no threshold can separate them.

## Q2. When is greedy splitting a problem?

**60-second answer.** Greedy splitting chooses the best split for the current node only. Some structures give zero gain at every single split. The XOR pattern is the standard example: the label is 1 exactly when the two binary features differ. Each feature alone is 50/50 on both sides of any cut, so the root decrease is zero for both. A tree that splits on one feature and then the other can represent the pattern, but a greedy search that stops at zero gain never reaches the second level. Remedies include allowing splits with zero gain at shallow depth, looking ahead, or using trees deep enough to represent the interaction. Boosting stumps alone cannot represent XOR, because their sum is additive.

**Follow-ups.**
- *Is Gini or entropy better?* They often select similar splits, though their rankings can differ. Gini is cheaper, because it avoids a logarithm.
- *Does a regression tree use the same search?* Yes, with the squared error as impurity. The decrease is the reduction in sum of squared errors, which equals $\frac{n_L n_R}{n}(\bar y_L-\bar y_R)^2$.

## Common mistakes

- Forgetting the child weights in the impurity decrease, which rewards splits that isolate single samples.
- Placing thresholds at data values instead of midpoints, which changes predictions between training points.
- Treating a tie as a reason to return no split, instead of applying the tie rule.
- Calling the greedy search globally optimal. It is locally optimal at each node only.
