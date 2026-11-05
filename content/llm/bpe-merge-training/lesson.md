---
title: BPE merge training
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01 with the hint ladder."
---

## The system

Byte-pair encoding (BPE) learns a vocabulary of subword symbols by compressing a word list. Start with the
characters. Repeat one rule: **count every adjacent pair of symbols, then merge the most frequent pair into one
new symbol**. Each merge adds one symbol, so after $k$ merges the vocabulary has the initial symbols plus $k$
new ones.

Set-up for one word $w$ with count $c_w$ (how often it occurs in the corpus):

- Its symbols are its characters followed by the end-of-word marker `</w>`, a separate symbol. So
  `low` $\to$ `l o w </w>`.
- A pair $(x, y)$ has count $\text{count}(x,y) = \sum_w c_w \cdot k_w(x,y)$, where $k_w(x,y)$ is the number of
  positions in $w$'s current symbol list where $x$ is directly followed by $y$. Overlapping positions all count.
- The merge is the pair with the largest count. Ties go to the **smallest pair in tuple order**, so
  `("e", "s")` beats `("s", "t")`. This is the course's rule. The paper's sample code takes a `max` over the
  counts and does not say how it breaks ties [corpus: library-papers/1508.07909.pdf.md p.4].

The output is the ordered list of merges, not the vocabulary. Applying them is the job of
`llm/bpe-encode-and-decode`.

## Worked example

Words `low` (count 5), `lower` (2), `newest` (6), `widest` (3). Count the pairs in the first step, weighted
by the word counts:

| pair | count | where it comes from |
|---|---|---|
| (e, s) | 9 | newest $6$ + widest $3$ |
| (s, t) | 9 | newest $6$ + widest $3$ |
| (t, </w>) | 9 | newest $6$ + widest $3$ |
| (w, e) | 8 | lower $2$ + newest $6$ |
| (l, o) | 7 | low $5$ + lower $2$ |
| (o, w) | 7 | low $5$ + lower $2$ |

Three pairs tie at 9. The smallest is `("e", "s")`, so merge 1 is `e s` $\to$ `es`. Recount: `(es, t)` and
`(t, </w>)` now tie at 9, and `("es", "t")` is smaller, so merge 2 is `es t` $\to$ `est`. Then `(est, </w>)`
has count 9, so merge 3 is `est </w>` $\to$ `est</w>`.

The result is `[("e","s"), ("es","t"), ("est","</w>")]`. Each step is the same rule, applied to the symbols
that the previous merge produced.

## Why it works

Learning on the **word-count table** is the same as learning on the raw text, with two differences. The table
is smaller, because each distinct word is stored once with its count. And no pair crosses a word boundary.
The paper says it does not consider pairs across word boundaries for efficiency, and that the algorithm can run
on the dictionary extracted from a text, with each word weighted by its frequency [corpus: library-papers/1508.07909.pdf.md p.3].
Counting raw text would let `e` followed by a space become a merge, which spends vocabulary on word-boundary artefacts.

The size bookkeeping is simple. The final vocabulary size equals the initial size plus the number of merges, and
the number of merges is the only hyperparameter [corpus: library-papers/1508.07909.pdf.md p.3].

## Signal vs. noise

- **Signal:** the marker is a symbol, so it can merge into the last character. Each word's count multiplies its
  pairs. A tie always goes to the same pair. Those three rules decide almost every wrong answer.
- **Signal:** recount after every merge. The symbols changed, so the counts change.
- **Noise for now:** incremental pair indexes for speed. The naive recount is correct, and write it first.

## Where it breaks

- **Overlap.** In `aaa` the pair `(a, a)` occurs at two positions, so its count is 2. Counting non-overlapping
  pairs gives 1 and changes the tie-break.
- **Merge direction.** Merging `aaa` from the right gives `a` and `aa` instead of `aa` and `a`. The next
  merge then differs.
- **Missing marker.** Without `</w>`, word endings are not distinct from word middles: `est` at the end of `newest`
  and `est` inside `estimate` share one symbol.
- **Float counts.** A count above $2^{53}$ cannot be stored exactly as a float, so two different counts can compare
  equal and a real difference becomes a tie.
- **Early stop.** When every word is one symbol, no pair is left. Return fewer merges and do not raise.

## Interview angle

*60-second answer:* "BPE training starts from characters plus an end-of-word symbol, and it works on a table of
word counts. Each step counts adjacent pairs, weighted by word frequency, and merges the most frequent pair
everywhere. I break ties deterministically, count overlapping occurrences, and stop early when nothing is left.
A naive recount costs time linear in the total symbol count per merge."

*Likely follow-ups:* Why learn on word counts rather than raw text? (Smaller, and no pair crosses a word boundary.)
Why is the tie-break needed? (Without it, the vocabulary depends on dict order, so two runs disagree.)

## Connections

- **Requires:** `llm/word-and-character-tokenizers` (the word-level vocabulary and the reserved ids).
- **Leads to:** `llm/bpe-encode-and-decode`, which applies these merges to new text.
- **Lineage:** Sennrich, Haddow and Birch (2016) adapt BPE, a data compression method, to word segmentation. They start
  from characters with an end-of-word symbol and merge the most frequent pair [corpus: library-papers/1508.07909.pdf.md p.3].
