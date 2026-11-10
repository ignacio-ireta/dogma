# Word-level vocabulary with UNK

Implement a class `WordTokenizer`. It learns a vocabulary from a corpus, maps text to integer ids,
and maps ids back to text. A word that did not survive the frequency cutoff encodes to the `<unk>`
id, so the tokenizer never fails on unseen input. Decoding an `<unk>` id gives back the string
`<unk>`, so the original word is lost. That loss is the price of a fixed-size vocabulary.

## Contract

- Word rule (corpus and text alike): a word is a maximal run of non-whitespace characters, as
  `str.split()` with no argument gives. Tabs, newlines and repeated spaces all separate words, and
  leading or trailing whitespace makes no empty word. Case is kept: `Cat` and `cat` are different
  words. Punctuation is kept: `cat.` and `cat` are different words.
- `WordTokenizer(corpus, min_count=1)`:
  - `corpus`: a `list` of `str`. A single `str` is rejected. Must not be mutated.
  - `min_count`: a Python `int` that is at least 1. Raises `TypeError` for other types (`bool` is
    rejected), and `ValueError` when it is below 1.
  - Raises `TypeError` when an entry of `corpus` is not a `str`.
- Counting: a word's count is the number of times it appears in the whole corpus. Every appearance
  counts, so a word that appears three times in one document has count 3.
- Special tokens are reserved: `<pad>`, `<unk>`, `<bos>`, `<eos>`. A corpus word equal to one of them
  is not counted. In `encode`, such a string is out of vocabulary and gives the `<unk>` id.
- `tokens`: a `list` of `str`. `tokens[i]` is the string with id `i`.
  - ids 0 to 3 are `<pad>`, `<unk>`, `<bos>`, `<eos>`, in that order, always, even for an empty corpus.
  - Then come the words with count at least `min_count`, ordered by **descending count**. Ties are
    broken by the **word itself, ascending in Python string order** (code-point order, so `B` sorts
    before `a`). Two words with the same count never follow first-appearance order.
- `encode(text)`: `text` must be a `str`, otherwise `TypeError`. Returns a `list` of Python `int`:
  one id per word of `text`, in order, with the `<unk>` id (1) for out-of-vocabulary words. No
  `<bos>` or `<eos>` is added. Empty or whitespace-only text returns `[]`.
- `decode(ids)`: `ids` must be a `list` of Python `int` (`TypeError` otherwise, `bool` included). An
  id outside `0 .. len(tokens) - 1` raises `ValueError`. Returns the token strings joined by single
  spaces. `decode([])` returns `""`.
- Round trip: when every word of a text is in the vocabulary, `decode(encode(text))` equals
  `" ".join(text.split())`.

## Examples

```python
>>> tok = WordTokenizer(["the cat sat", "the cat ran", "a dog sat"], min_count=2)
>>> tok.tokens
['<pad>', '<unk>', '<bos>', '<eos>', 'cat', 'sat', 'the']
          # the, cat, sat appear 2 times; ran, a, dog appear once and are cut
          # cat, sat, the all have count 2, so they are ordered alphabetically
>>> tok.encode("the dog sat")
[6, 1, 5]  # dog is out of vocabulary and gets <unk> = 1
>>> tok.decode([6, 1, 5])
'the <unk> sat'
>>> tok.encode("The\tcat  ")
[1, 4]     # "The" is not "the"; the tab and the two spaces separate words
>>> WordTokenizer(["b b a", "c"]).tokens
['<pad>', '<unk>', '<bos>', '<eos>', 'b', 'a', 'c']
          # b has count 2, so it comes before a even though "a" sorts first;
          # a and c tie at 1, so a comes before c
>>> WordTokenizer([], min_count=1).tokens
['<pad>', '<unk>', '<bos>', '<eos>']
>>> WordTokenizer(["a"]).encode("")
[]
>>> WordTokenizer(["a"]).encode(3)
TypeError  # text must be a str
```

On real prose, such as the pinned `gutenberg_frankenstein` slice, `“What` and `What` are different words,
and so are `father’s` and `father`: punctuation and case are kept, so the vocabulary and the ids of held-out text match
the naive count-and-sort.

## Constraints

- Allowed: built-in types and the standard library.
- Not allowed: any tokenizer from another library, and any pretrained vocabulary file. The
  vocabulary must come from `corpus` alone. Nothing may be read from disk or the network.
- The corpus must not be changed: no sorting, popping or editing of the list or its strings.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare your vocabulary and ids
with an independent naive count-and-sort on seeded random corpora, and check the families: equal
counts ordered by word rather than first appearance; descending count beating alphabetical order;
the `min_count` threshold itself (kept at exactly `min_count`, dropped one below); counting
occurrences and not documents; case and punctuation kept; tabs, newlines and repeated spaces;
reserved special strings in the corpus and in the text; empty and whitespace-only text; a
`min_count` above every count; a 5,000-word vocabulary with dense ids; round trips on generated
text; bool and float arguments rejected; out-of-range and bool ids rejected; and a corpus that must
not change. Realism families: real English prose with capital letters, curly quotes and
dashes, counted with `min_count` and checked on held-out text.
