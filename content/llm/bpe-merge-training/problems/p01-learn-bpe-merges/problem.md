# Learn BPE merges from word counts

Implement `train_bpe_merges(word_counts, num_merges)`. It learns byte-pair-encoding merges. The input
is a table of words and how often each occurs, not the raw text. Each word starts as a sequence of
symbols: its characters, then one extra symbol `</w>` that marks the end of the word. Repeat
`num_merges` times: count every adjacent pair of symbols, weighted by the word counts, and merge the
most frequent pair in every word. Return the merges in the order they were learned.

## Contract

- `word_counts`: a `dict` from `str` to `int`. Each key is a non-empty `str` with no whitespace, and it
  must not contain the text `</w>`. Each value is a Python `int` of at least 1. Must not be mutated.
  - `TypeError` for a non-dict, a non-str key, or a count that is not a Python `int` (`bool` is rejected).
  - `ValueError` for an empty word, a word with whitespace or the text `</w>`, or a count below 1.
- `num_merges`: a Python `int` of at least 0. `TypeError` for other types (`bool` included), `ValueError`
  when negative.
- Symbols: a word `w` starts as `[*w, "</w>"]`. The marker is one symbol, so `"low"` starts as
  `["l", "o", "w", "</w>"]`. Case and punctuation are ordinary characters: `"Low."` starts as
  `["L", "o", "w", ".", "</w>"]`, and `"Low"` and `"low"` share no symbols.
- Pair counts: for a pair `(x, y)`, the count is the sum over words of `count(word) × k`, where `k` is the
  number of positions `i` with `symbols[i] == x` and `symbols[i + 1] == y`. Overlapping positions all count,
  so `"aaa"` has two `("a", "a")` pairs.
- Choosing a merge: take the pair with the largest count. On a tie, take the smallest pair under Python
  tuple order: compare the left symbols as strings, then the right symbols. So `("a", "</w>")` beats
  `("a", "b")`, and `("a", "b")` beats `("b", "a")`.
- Applying a merge `(x, y)`: in each word, scan from left to right. Where `x` is followed by `y`, replace
  both with the one symbol `x + y` and continue after them. Overlaps are never merged twice.
- Stop when `num_merges` merges are learned, or when no adjacent pair is left. Then return fewer merges.
- Returns: a `list` of `tuple[str, str]` of length at most `num_merges`. An empty table returns `[]`.

## Examples

```python
>>> train_bpe_merges({"low": 5, "lower": 2, "newest": 6, "widest": 3}, 3)
[('e', 's'), ('es', 't'), ('est', '</w>')]
# (e,s), (s,t) and (t,</w>) all count 9. The tie goes to ('e', 's').
>>> train_bpe_merges({"aaa": 1, "bb": 1}, 1)
[('a', 'a')]       # (a, a) counts 2 because the two occurrences overlap in "aaa"
>>> train_bpe_merges({"ab": 1, "ba": 1}, 2)
[('a', '</w>'), ('a', 'b')]
>>> train_bpe_merges({"a": 3}, 2)
[('a', '</w>')]    # after one merge every word is a single symbol, so learning stops
>>> train_bpe_merges({"ab": 1, "cd": 2}, 1)
[('c', 'd')]       # the count of "cd" is 2, so its pairs outweigh those of "ab"
>>> train_bpe_merges({}, 5)
[]
>>> train_bpe_merges({"a": 1}, "2")
TypeError
```

On real text, such as the first 3,000 words of `gutenberg_frankenstein` counted with `collections.Counter`,
keys keep their capital letters and punctuation exactly as they appear, and the learned merges must match
the naive trainer.

## Constraints

- Allowed: built-in types and the standard library.
- Not allowed: any tokenizer or BPE implementation from another library (for example `tokenizers`,
  `sentencepiece` or `tiktoken`), and any pretrained merge list. Learn the merges from `word_counts` alone.
- Must not mutate `word_counts`.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare your merge list with an
independent naive trainer on seeded random word tables, and check the families: the tie-break in both
directions and across the end-of-word marker; overlapping pairs such as `aaa`; word counts that weight
pairs; early stop when every word is one symbol; a count above 2^53 that must stay exact; an empty table
and zero merges; the list returned and the input unchanged; rejected types and values; and real
English word counts with capital letters and non-ASCII punctuation, kept in the words exactly as given.
