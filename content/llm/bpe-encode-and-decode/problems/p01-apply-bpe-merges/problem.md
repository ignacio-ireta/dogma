# Apply BPE merges by rank

Implement `BPETokenizer(merges)`. It applies a learned list of byte-pair merges to a word and decodes the
result back. The merges are given in the order they were learned. Each merge has a **rank**, which is its
index in the list. A word is first split into characters followed by the end-of-word marker `</w>`, which
is a separate symbol. Then the tokenizer repeats one step until nothing changes: among the adjacent pairs
that are merges, pick the one with the **lowest rank**, and merge every occurrence of it, scanning left to
right.

The rank order is the whole point. Scanning the word left to right and merging whatever pair you see first
gives a different answer, as the examples show.

## Contract

- `BPETokenizer(merges)`: `merges` is a `list` of `tuple` pairs `(left, right)`, each element a non-empty
  `str`. Raises `TypeError` for other shapes or types, and `ValueError` for an empty symbol or a pair that
  appears twice (each pair has one rank).
- `encode(word)`: `word` is a `str`. Returns a `list` of `str`. The word starts as `[*word, "</w>"]`, then
  the rule above runs until no adjacent pair is a merge. The symbols of the result, concatenated in order,
  always equal `word + "</w>"`. An empty word encodes to `["</w>"]`.
  - Raises `TypeError` when `word` is not a `str`.
  - Raises `ValueError` when `word` contains whitespace or the text `</w>`.
  - A character that no merge touches stays a single symbol. Nothing is `<unk>`.
- `decode(tokens)`: `tokens` is a `list` of `str`. Concatenate them into one string, which must end with
  `</w>`. Remove that trailing marker and return the rest. `decode(["</w>"])` is `""`.
  - Raises `TypeError` when `tokens` is not a list of `str`.
  - Raises `ValueError` when the joined text does not end with `</w>`, or when `</w>` appears anywhere else.
    The check is on the joined string, so a marker split across tokens counts. `decode([])` joins to `""`,
    which has no marker, so it raises `ValueError`.
- Must not mutate `merges`.

## Examples

```python
>>> BPETokenizer([("b", "c"), ("a", "b")]).encode("abc")
['a', 'bc', '</w>']
# (b, c) has rank 0 and is applied first. Then (a, bc) is not a merge, so the word stops here.
# A left-to-right scan would merge (a, b) first and give ['ab', 'c', '</w>'].

>>> BPETokenizer([("ab", "x"), ("a", "b")]).encode("abx")
['abx', '</w>']
# Only (a, b) is a merge at first. It gives 'ab', and then ('ab', 'x') has rank 0 and is applied.
# A rule that does one pass per merge, in rank order, would stop at ['ab', 'x', '</w>'].

>>> BPETokenizer([("a", "a")]).encode("aaa")
['aa', 'a', '</w>']   # the left two a's merge; the right a is left alone

>>> BPETokenizer([("a", "b")]).encode("é")
['é', '</w>']         # characters no merge touches stay as they are

>>> tok = BPETokenizer([("b", "c"), ("a", "b")])
>>> tok.decode(['a', 'bc', '</w>'])
'abc'
>>> tok.decode(['a', 'bc'])
ValueError            # the end-of-word marker is missing
```

On real text, such as the pinned `gutenberg_frankenstein` slice, some words start with a capital letter,
and some carry a curly quote such as `’` or `“`, or a dash. Encoding each word and joining its symbols must
give back the word and the marker, character for character.

## Constraints

- Allowed: built-in types and the standard library.
- Not allowed: any BPE or tokenizer library (for example `tokenizers`, `sentencepiece` or `tiktoken`), and any
  pretrained merge list. The merges come in through the constructor only.
- Must not mutate `merges`.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare your encoding with an independent
naive encoder on seeded random merge lists, check that every word round-trips through `encode` and `decode`
under Hypothesis, and cover these families: the rank-versus-position case in both directions; a merge created by
an earlier merge that outranks a later rule; overlapping pairs such as `aaa`; every occurrence of the pair
merged; unrelated characters left alone; the empty word; duplicate and malformed merges; whitespace and the
marker text in the word; missing or misplaced markers in the tokens; inputs that must not change; and real
English prose with capital letters, curly quotes and non-ASCII punctuation, where merges must keep every
original character.
