# Longest common subsequence with a witness

Two DNA-like reads from a sequencing run are two strings over a small alphabet. The lab wants the
longest sequence of symbols that appears in both reads, in the same order, though not necessarily
next to each other. Implement `longest_common_subsequence(a, b)`. It returns the length of the
longest common subsequence, and one such subsequence as a string, so that the length can be checked
against the string.

A subsequence keeps the order of the original characters and may skip any of them. `"ace"` is a
subsequence of `"abcde"`, but `"ace"` is not a contiguous substring of it.

## Contract

- `a`, `b`: Python `str`, possibly empty. Characters are compared by equality. Neither argument is
  changed.
- Returns: a `tuple[int, str]`, `(length, witness)`.
  - `length` is the number of characters in a longest common subsequence of `a` and `b`.
  - `witness` is one such subsequence, a `str` of exactly `length` characters. Any valid
    longest subsequence is accepted, because several can exist.
- Empty input gives `(0, "")`: when either string is empty, the only subsequence is the empty one.
- Raises `TypeError` when `a` or `b` is not a Python `str` (`bytes`, `None` and lists are rejected).
- Complexity: O(len(a) x len(b)) time. The hidden tests include strings of 1,200 characters, which
  must finish within the time limit. Python's default recursion limit is 1,000 frames, so a
  recursion that goes one level per character can fail on those inputs.

## Examples

```python
>>> longest_common_subsequence("ABCBDAB", "BDCABA")[0]
4        # for example "BCBA"; "BDAB" and "BCAB" are also valid witnesses
>>> longest_common_subsequence("abc", "def")
(0, "")
>>> longest_common_subsequence("aaaa", "aa")
(2, "aa")
>>> longest_common_subsequence("abcde", "aXcXe")
(3, "ace")  # the longest common contiguous block is only one character
>>> longest_common_subsequence("", "xyz")
(0, "")
```

## Constraints

- Allowed: built-in types and the standard library.
- Not allowed: `difflib` (its matching algorithm is not the longest common subsequence) and
  `functools`. Build the table yourself.
- A longest common substring is a different problem. A window that only looks at contiguous
  characters returns the wrong length on the example `"abcde"`, `"aXcXe"`.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check the length against a
brute-force search over subsets for short strings, and against an independent suffix-based table
for longer ones. They check that the witness is a subsequence of both strings and has exactly the
returned length. The edge families they cover: empty strings, identical strings, disjoint alphabets,
repeated characters, the substring trap, the return type, rejected types, and 1,200-character
strings within the time limit.

The hidden tests draw their bulk inputs from the real datasets `gutenberg_frankenstein`,
`gutenberg_pride_prejudice` and `gutenberg_walden` (see `docs/DATASETS.md`).
