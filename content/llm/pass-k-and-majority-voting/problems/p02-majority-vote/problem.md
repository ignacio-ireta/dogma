# Majority vote over normalised answers

Self-consistency samples several reasoning paths for one question, extracts a final answer from each,
and returns the answer that most paths agree on. The vote only works if equal answers are recognised
as equal, and if ties are settled by a rule that does not depend on luck. Implement
`majority_vote(answers)`.

## Contract

- `answers`: a list or tuple of `str`, one final answer per sample, in sampling order. It must not be
  a bare `str`, which is a sequence of characters, not a list of answers.
- **Normalise** each answer: `" ".join(a.split()).lower()`. This collapses runs of whitespace,
  trims both ends and lowercases. Punctuation is kept, so `"3."` and `"3"` are different answers.
- **Drop** every answer whose normalised form is the empty string. Dropped samples cast no vote.
- **Vote.** Count the surviving normalised answers. The winner has the largest count.
- **Tie rule.** If several answers share the largest count, the winner is the one that appeared
  **first** in the original `answers` list, counting dropped positions as absent. Ties never depend
  on alphabetical order or on the last occurrence.
- Returns: the normalised winning answer, a `str`.
- Raises `TypeError` when `answers` is a bare `str` or any element is not a `str`.
- Raises `ValueError` when `answers` is empty, or when every answer normalises to the empty string.
- Must not mutate the input list.

## Examples

```python
majority_vote(["42", "42", "41"])  # -> "42"
majority_vote(["  Paris ", "London", "paris"])  # -> "paris"   (2 votes after normalising)
```

```python
majority_vote(["Paris", "London", "paris", "london"])  # -> "paris"   (2 vs 2: Paris appeared first)
majority_vote(["  ", "7", "7 ", ""])  # -> "7"       (blank answers are dropped)
```

## Constraints

Allowed: Python standard library. Not allowed: `collections.Counter` (it is the reference for the
hidden tests), `statistics.mode` and `statistics.multimode`. Count with a dictionary or a loop, and
record the first position at which each normalised answer appears.

## Done looks like

The public examples pass, plus a blank-answer edge case. Hidden tests compare against an
independent counter on seeded random answer lists, check the tie rule in several shapes (a tie at the
start, a tie with a later first occurrence, and a three-way tie), check normalisation of case and
internal whitespace, and check that a strict majority does not depend on the order of the list. They
also cover blank-only input, empty input, a bare string, non-string elements and input that is not
mutated.
