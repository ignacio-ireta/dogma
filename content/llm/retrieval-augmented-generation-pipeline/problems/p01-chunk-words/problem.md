# Chunk a document into overlapping word windows

Retrieval starts by splitting a document into chunks. Implement `chunk_words(text, size, overlap)`,
which cuts `text` into windows of `size` words, where each window starts `size - overlap` words after
the previous one. The overlap keeps a sentence that straddles a boundary visible in two chunks.

## Contract

- `text`: `str`. A **word** is a maximal run of non-whitespace characters, so `text.split()` defines
  the words. Each returned chunk joins its words with a single space.
- `size`: `int`, at least 1.
- `overlap`: `int`, with `0 <= overlap < size`.
- Returns: `list[str]`, one string per chunk, in document order.
- Window rule: the first window starts at word 0. Window `j` starts at word `j * (size - overlap)`
  and covers up to `size` words, cut at the end of the text. Emit windows while their start is inside
  the text. **Stop after the first window that reaches the last word.** This rule prevents a trailing
  chunk that is wholly contained in the previous one.
- Text with no words (empty or whitespace only) returns `[]`.
- Raises `TypeError` when `text` is not a `str`, or when `size` or `overlap` is not an `int`
  (`bool` is not accepted as an `int` here).
- Raises `ValueError` when `size < 1`, or when `overlap` is outside `[0, size)`.
- Must not mutate inputs.

## Examples

```python
chunk_words("a b c d e f g h i j", 4, 2)
# -> ["a b c d", "c d e f", "e f g h", "g h i j"]

chunk_words("a b c d e", 4, 2)
# -> ["a b c d", "c d e"]        # the second window reaches the last word and stops the loop
```

```python
chunk_words("  one   two\nthree ", 5, 1)
# -> ["one two three"]           # whitespace is collapsed; one window covers everything
```

```python
chunk_words("", 3, 1)
# -> []
```

## Constraints

Allowed: Python standard library (`str.split` and `str.join` are enough). Not allowed: `textwrap`,
`nltk`, `tiktoken`, `langchain` or `llama_index`. Build each chunk from the word list, not by slicing
the raw string by characters.

## Done looks like

The public examples pass, plus an empty-text case. Hidden tests compare against an independent
enumeration of window starts on seeded random documents, and check the structural properties of
any overlapping chunking: every word is covered, consecutive full windows share exactly `overlap`
words, no chunk is contained in its predecessor, and no chunk exceeds `size` words. They also cover
`size` larger than the document, `overlap = 0`, and every invalid-argument family.
