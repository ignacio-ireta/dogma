## Hint 1 — concept

Byte-pair training is a loop with one idea repeated. Count the adjacent pairs of symbols, pick the most
frequent pair, and merge it everywhere. Then count again, because the merge changed the symbols. The
word counts are the weights: a word that occurs 6 times contributes 6 to each of its pairs. Ask why the
word table is the right thing to learn from, and what would go wrong if you counted pairs in the raw text.

## Hint 2 — formula / approach

Give each word its characters followed by one end-of-word symbol, `</w>`. The count of a pair is the sum,
over the words that contain it, of the word's count times the number of times the pair occurs in that
word. Two occurrences that overlap both count, so `aaa` has two `(a, a)` pairs. Choose the largest count.
On a tie, choose the smallest pair under tuple order, which makes the result independent of dict order.
To merge, scan each word from the left and replace each match with the joined symbol. Skip past both
symbols after a match, so an overlap is never merged twice. Stop early when no pair is left.

## Hint 3 — code shape

Validate the arguments first, then run the loop. The skeleton has the steps in order; fill in the blanks:

```python
entries = [(count, [*word, END]) for word, count in word_counts.items()]
merges = []
for _ in range(num_merges):
    counts = {}
    for weight, symbols in entries:
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            counts[pair] = ...  # add the word weight, not 1
    if not counts:
        break
    best = min(counts, key=lambda pair: (..., pair))  # largest count, then smallest pair
    merges.append(best)
    entries = [(weight, ...) for weight, symbols in entries]  # apply the merge
return merges
```

Run your trainer by hand on `{"ab": 1, "ba": 1}` with two merges. You should get `("a", "</w>")` and then `("a", "b")`.
