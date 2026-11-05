## Hint 1 — concept

A tokenizer is two tables. One table maps each word to an integer id, and the other maps each id back to
its string. The vocabulary is the set of words that get an id of their own. Every other word needs one
shared fallback id, the one for `<unk>`. Before you write anything, decide three things from the contract:
what counts as a word, which words are kept, and in what order the ids are assigned.

## Hint 2 — formula / approach

Count with a dictionary: `count[word] += 1` for every word of every document, so repeats add up. Keep the
words where `count >= min_count`. Sort them with one key that has two parts, a descending count and an
ascending word, so the tie-break is part of the key and not a second pass. Put the four specials in front,
so their ids are 0 to 3. A word that is one of the specials never enters the count. Encoding is then one
lookup per word, with `<unk>` as the default. Decoding is the reverse lookup, plus the range checks.

## Hint 3 — code shape

Validate the arguments first, with `type(x) is int` for `min_count` and the list and str checks for the
corpus. Then the steps in order:

```python
counts: dict[str, int] = {}
for document in corpus:
    for word in document.split():
        ...  # skip the reserved words, then count
kept = [...]  # the words that reach min_count
kept.sort(key=lambda word: (..., word))  # descending count, then the word
self.tokens = list(SPECIALS) + kept
self._ids = {...}  # word -> id, starting at len(SPECIALS)
```

Test your sort on the examples in `problem.md` by hand before you run the tests.
