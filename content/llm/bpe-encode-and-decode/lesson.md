---
title: BPE encode and decode
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01 with the hint ladder."
---

## The system

Training gives an **ordered** list of merges $m_1, \dots, m_K$. Each merge $m_r = (x, y)$ has **rank** $r$, its position in the list. To encode a word:

1. Start from its symbols: its characters, then the end-of-word marker `</w>`.
2. Among the adjacent pairs that are merges, pick the pair with the **lowest rank**.
3. Merge every occurrence of that pair, scanning left to right. A match consumes both symbols.
4. Repeat from step 2 until no adjacent pair is a merge.

The result is a list of symbols. Decoding is the inverse: concatenate the symbols and remove the trailing marker. Concatenation is exact, because a merged symbol is the join of its two parts.

Two rules make this deterministic. The **lowest rank wins**, not the leftmost pair. And the loop **repeats**, because a merge can create a symbol that pairs with a neighbour, and that new pair may have a lower rank than anything left.

## Worked example

Merges $[(\text{b},\text{c}), (\text{a},\text{b})]$, word `abc`. Symbols: `a b c </w>`.

| step | adjacent merges present (rank) | chosen | symbols after |
|---|---|---|---|
| 0 | (b, c) rank 0, (a, b) rank 1 | (b, c) | `a bc </w>` |
| 1 | none: (a, bc) is not a merge | stop | `a bc </w>` |

Result: `['a', 'bc', '</w>']`. Decoding joins them to `"abc</w>"` and removes the marker, so `"abc"`.

Applied **left to right**, merging whatever pair is found first, `(a, b)` is leftmost and gives `['ab', 'c', '</w>']`. Both are valid segmentations of `abc`, but only the first matches training, which learned `(b, c)` first. This is the case the problem checks.

## Why it works

Training and encoding must agree on the order, because each merge was chosen from the symbols that earlier merges produced. A rank is a priority: a merge learned later was learned on text where the earlier merges had already applied. Applying the merges in priority order reproduces the training segmentation.

**Greedy longest-match** is a different algorithm. It takes the longest vocabulary string that starts at the current position. For `abc` with the vocabulary `{a, b, c, ab, bc}`, it gives `['ab', 'c']`, which matches neither the rank result nor the left-to-right result. It ignores the merge order, and nothing forces its vocabulary to be reachable by the merges. Use it only when you have a vocabulary and no merge list.

**Byte-level BPE** starts from the 256 byte values instead of characters, so every string is encodable. The GPT-2 paper motivates this base with the size argument [corpus: library-papers/radford-gpt2-2019.pdf.md p.4]. Decoding must then work on bytes. A non-ASCII character can be several bytes, and a merge can end in the middle of one, so decoding each token alone fails. Concatenate the bytes, then decode UTF-8 once.

## Signal vs. noise

- **Signal:** lowest rank first, merge every occurrence, repeat until nothing changes. Those three steps decide every answer.
- **Signal:** the round trip. `"".join(encode(w)) == w + "</w>"` is a check you can run on every word.
- **Noise for now:** the speed of an encoder on huge vocabularies. The naive loop is correct first.

## Where it breaks

- **Leftmost-first encoding.** A common mistake. It gives a valid split that training never produced, so the model reads tokens it has not learned.
- **One pass per merge.** Applying each merge once in rank order differs from the repeated loop. With `(ab, x)` at rank 0 and `(a, b)` at rank 1, the loop gives `['abx', '</w>']`, and one pass gives `['ab', 'x', '</w>']`.
- **Overlaps.** In `aaa` with `(a, a)`, the left two symbols merge first, giving `['aa', 'a', '</w>']`. A right-to-left scan gives `['a', 'aa', '</w>']`.
- **Decoding without the marker.** A token list that does not end in `</w>` is not a word. Reject it, do not guess.
- **Non-unique merges.** A pair listed twice has two ranks, so the order is ambiguous. Reject the list.

## Interview angle

*60-second answer:* "Encoding applies the learned merges in rank order. At each step I take the adjacent pair with the lowest rank, merge all of its occurrences left to right, and repeat until nothing changes. That is not greedy longest-match, which ignores the order. Decoding concatenates the tokens and strips the end-of-word marker. For byte-level BPE I would decode the bytes once, not each token separately, because a token can end inside a multi-byte character."

*Likely follow-ups:* Why does order matter? (Each merge was learned on the output of the earlier ones.) What breaks if the decoder works per token? (Multi-byte characters that a merge split.)

## Connections

- **Requires:** `llm/bpe-merge-training`, which produces the ordered merge list this topic applies.
- **Leads to:** the byte-level tokenizers that production models use, and the padding and masking of batches of encoded words.
- **Lineage:** Sennrich, Haddow and Birch (2016) split a word into characters at test time, then apply the learned operations to merge them into known symbols [corpus: library-papers/1508.07909.pdf.md p.4].
