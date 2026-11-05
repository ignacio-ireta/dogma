## Hint 1 — concept

Score each calibration row by how little probability the model gave the true class, 1 minus that probability. The threshold is one calibration score, chosen by a rank that uses the number of calibration rows and alpha. A class joins a test row's set when its own score is at or under the threshold.

## Hint 2 — formula / approach

The rank is k = ceil((n + 1)(1 - alpha)). The threshold is the k-th smallest calibration score, or infinity when k > n. For a test row, a class is included when 1 minus its probability is at most the threshold. Read alpha as the decimal it was written as (repr gives that text) and compute the rank with exact rational arithmetic, because a float product can round an integer up.

## Hint 3 — code shape

Validate the probability matrices, the labels (integer dtype only) and alpha, raising ValueError. Pick each calibration row's true-class probability with paired indexing, subtract from 1, and sort. Compute k, then the threshold. Build the (test rows, K) mask by comparing 1 minus the test matrix with the threshold, and return it as a boolean array.
