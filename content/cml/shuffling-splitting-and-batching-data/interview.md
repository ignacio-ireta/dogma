# Interview notes — Shuffling, splitting and batching data

## Q1. How would you split a dataset into training and test sets?

**60-second answer.** I start from the structure of the rows. If they are i.i.d., a random split is
a fair stand-in for the population, so I shuffle with a fixed seed and take the test count as the
ceiling of the test fraction times the row count. If a class is rare, I stratify, so each side keeps
its share of that class within one row. I fit nothing before the split. The seed and the split
protocol go into the experiment record, so the split can be rebuilt exactly.
**Follow-ups.**
- *When is random splitting wrong?* When rows are ordered in time or share an entity. Then a
  chronological or group split keeps future rows or related rows out of training.
- *How many splits do you need?* Enough to see the variance of the metric. Repeat the whole protocol
  with several seeds and report the spread, not one lucky split.

## Q2. Why does shuffling matter for mini-batch training, and what can go wrong?

**60-second answer.** A batch is only a fair sample of the data if the rows are mixed. Sorted data
puts one class or one time range in each batch, which biases the steps. Shuffling once per epoch,
with a generator built from a seed, gives every epoch different batches and keeps runs reproducible.
The short final batch is either kept, so every row is used, or dropped, so every batch has the same
size. I say which one I chose.
**Follow-ups.**
- *Why build the generator once?* Rebuilding it per epoch repeats the same permutation.
- *Does the order inside a batch matter?* In exact arithmetic the mean gradient of a batch does not depend
  on the order of its rows, because the sum is order-free. Float sums can differ in the last bits.
  The index lists do record the order, which is why the protocol fixes it.

## Common mistakes

- Rounding the test count to nearest instead of taking the ceiling.
- Splitting first and then stratifying with a new random generator, so the split is not reproducible.
- Splitting time-ordered rows at random.
- Letting rows from one entity appear on both sides of the split.
- Sorting a batch, or shuffling the caller's array in place.
