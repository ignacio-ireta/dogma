# Interview notes: Hash maps and sets

## Q1. Find the k most frequent words in a document.

**60-second answer.** One pass with a dict counts every word: O(n). Then rank the distinct words.
Sorting the d distinct words costs O(d log d). If k is small, a heap of size k costs O(d log k).
I state the tie-break before I code, because it changes the answer: equal counts are ordered
alphabetically here. I also say which k cases I guard: k zero or negative returns nothing, and k
larger than the vocabulary returns everything.

**Follow-ups.**
- *The corpus is too big for memory?* Stream it. Use approximate counts (Count-Min sketch) or
  Misra-Gries for heavy hitters.
- *Does the answer depend on case or punctuation?* Yes. Normalise the tokens first, and say so.

## Q2. What are the average and worst-case costs of a hash lookup, and why?

**60-second answer.** Average O(1): the hash function spreads keys across buckets, and the table
resizes to keep the load factor bounded, so each bucket holds O(1) keys on average. Worst case
O(n): if many keys land in one bucket, the lookup scans them. A good hash function and resizing make
the worst case rare for ordinary data.

**Follow-ups.**
- *When is a balanced tree better?* When you need keys in sorted order or range queries. The cost is
  O(log n) per operation, even in the average case.
- *Can a list be a key?* No. Use a tuple.

## Q3. When would you not use a hash map?

**60-second answer.** When the output must be ordered, or when you need range queries such as
"all keys between a and b". Then a sorted array or a balanced tree fits better. A hash map is also a
poor fit when the keys are not hashable, or when memory is tight and an approximate structure would
do.

**Follow-ups.**
- *Is a set iteration order useful?* No. Sort explicitly when the order matters.
- *What about a Bloom filter?* It answers membership with a small false-positive rate in far less
  memory. It cannot list the members.

## Common mistakes
- Testing membership with `x in some_list` inside a loop, which makes the loop O(n squared).
- `counts[t] += 1` on a new key, which raises KeyError. Use `get` with a default.
- Sorting by count alone and forgetting the tie-break, so ties come out in first-appearance order.
- Treating a dict's or a set's iteration order as a ranking.
- Sorting the input list in place to count it, which changes the caller's data.
