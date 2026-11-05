## Hint 1 — concept

The rule gives a rare class a large weight and a common class a small weight, so that every class has the same total sample weight. Each class's weighted count is its count times its weight, and the rule makes that equal to n divided by the number of classes. Actual losses also depend on prediction errors.

## Hint 2 — formula / approach

Count the samples in each class, and let k be the number of distinct labels. For a class with count c, the weight is n / (k c). Use true division: integer division gives 0 for most classes. The keys are the labels themselves, not their positions.

## Hint 3 — code shape

Convert the input to an array and check that it is 1-D and not empty. Use `np.unique(arr, return_counts=True)` to get the labels and their counts, then build the dictionary with a comprehension over the two lists, pairing each label with its own count. Return the dictionary.
