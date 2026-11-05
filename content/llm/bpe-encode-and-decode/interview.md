# Interview notes — BPE encode and decode

## Q1. How do you apply a learned BPE vocabulary to a new word?
**60-second answer.** Start from the characters and the end-of-word marker. Repeatedly find the adjacent pair with
the lowest merge rank, merge all of its occurrences from left to right, and stop when no adjacent pair is a merge.
The order is essential: merges were learned on the output of earlier merges. I would not use greedy longest-match,
because it ignores the order and can produce segmentations that training never produced.
**Follow-ups.** What about a word of unseen characters? Each stays a single symbol, so nothing is unknown. How do
you test it? A round trip on random words and a hand-checked case where the rank order differs from left-to-right.

## Q2. How do you decode byte-level BPE?
**60-second answer.** Concatenate the token bytes, then decode UTF-8 once. A token can end in the middle of a
multi-byte character, so decoding each token separately is wrong. The byte base also means no input is out of
vocabulary, so decode never needs an unknown token.
**Follow-ups.** What happens if a model generates a truncated byte sequence? The decoder must handle the invalid
trailing bytes, usually with a replacement character, instead of failing.

## Q3. What invariants would you check on a tokenizer?
**60-second answer.** Round trip: decoding the encoding gives the input back. Coverage: the symbols concatenate to the
word plus the marker. Determinism: the same word gives the same symbols. Validation: duplicate merges and missing
markers are rejected instead of guessed around.
**Follow-ups.** Which one catches the most bugs? The concatenation check catches dropped or duplicated characters,
and the rank-order example catches the leftmost-first mistake.

## Common mistakes
- Merging the leftmost pair instead of the lowest-rank pair.
- Merging only the first occurrence of the chosen pair.
- Confusing BPE encoding with greedy longest-match over the vocabulary.
- Decoding byte-level tokens one at a time.
