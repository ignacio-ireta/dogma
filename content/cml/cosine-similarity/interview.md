# Interview notes: Cosine similarity

## Q1. Cosine similarity or Euclidean distance for embeddings?

**60-second answer.** Cosine compares direction and ignores length. Euclidean compares position, so
length matters. For text embeddings the norm often tracks document length or token frequency rather
than meaning, so cosine is the default. If the embeddings are normalised to unit length, the two
rank neighbours identically, because $\lVert\mathbf{a}-\mathbf{b}\rVert^2 = 2 - 2\cos\theta$. I
would check that empirically on the actual model, because the norm is not always uninformative.

**Follow-ups.**
- *What if one vector is zero?* Cosine is undefined. Decide and document a convention (here, 0.0),
  and check how often zero vectors occur, since they often signal an empty input.
- *When would Euclidean win?* When the norm carries information, such as a confidence or a count.

## Q2. Why normalise once and then use dot products in a vector index?

**60-second answer.** Normalising every stored vector once costs one pass over the data. After
that, cosine for any query is one dot product, which is what hardware and approximate-nearest-
neighbour indexes are built to compute fast. The query gets normalised once too. The ranking matches
cosine exactly, so nothing is lost in quality.

**Follow-ups.**
- *Does normalising change the ranking when vectors have different lengths?* Yes, relative to the
  raw dot product, because the length is removed. Relative to cosine, nothing changes.
- *Cost of the index?* One matrix multiply for a batch, memory-bound for large dimensions.

## Q3. When does cosine similarity mislead?

**60-second answer.** Three cases. First, when magnitude is signal: a count of 100 is not the same
evidence as a count of 1 with the same mix. Second, when features are non-negative, so cosines are
compressed into $[0,1]$ and most pairs look similar. Third, when the space has a shared mean
direction, which inflates every cosine. Centring the vectors, or subtracting the mean embedding,
addresses the last two. I would look at the distribution of cosines on random pairs before setting
a threshold.

**Follow-ups.**
- *How would you check a threshold?* Compare against known positive and negative pairs, and check
  the score distribution.
- *Pearson correlation?* It is the cosine of the centred vectors, so the same tool with a different
  centre.

## Common mistakes

- Dividing by the norm without checking for zero, which produces NaN that then spreads silently.
- Using `*` (element-wise) where `@` (matrix product) was meant, so the result has the wrong shape.
- Forgetting to normalise the second matrix, so only one side of the comparison is scaled.
- Trusting values such as 1.0000000000000002 without clipping, then failing a bounds check.
- Treating cosine as a distance. It is a similarity; $1 - \cos$ is not a metric in general.
