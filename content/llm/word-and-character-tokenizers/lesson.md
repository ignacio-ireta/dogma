---
title: Word and character tokenizers
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A model never sees strings. It sees integer ids, and each id picks one row of an embedding table. A **tokenizer** is therefore a pair of maps:

- `encode`: text $\to$ a list of ids, one id per token.
- `decode`: ids $\to$ text.

A **vocabulary** $V$ is the finite list of token strings the tokenizer knows. Its size $|V|$ fixes the number of embedding rows. The two granularities sit at opposite ends of one trade-off:

| Tokenizer | What a token is | Vocabulary size $\lvert V\rvert$ | Tokens for a text | Unseen input |
|---|---|---|---|---|
| Word | one whitespace-separated word | large (every word worth keeping) | few | needs an `<unk>` id |
| Character | one character | small (the alphabet) | many (about one per character) | none, if the alphabet covers the text |

Larger $\lvert V\rvert$ gives shorter sequences. Attention cost grows with sequence length, so this is a budget decision.

**Special tokens** are reserved ids that carry control meaning, not text:

| Token | Id (in this course) | Role |
|---|---|---|
| `<pad>` | 0 | fills a batch to equal length; masked out |
| `<unk>` | 1 | stands for any word outside the vocabulary |
| `<bos>` | 2 | marks the start of a sequence |
| `<eos>` | 3 | marks the end of a sequence |

Reserve the ids before counting. If a corpus word can be spelled `<unk>`, it can take the id the fallback needs.

## Worked example

Corpus `["the cat sat", "the cat ran", "a dog sat"]`, with `min_count = 2`. Count every occurrence:

| word | count | at least 2? |
|---|---|---|
| the | 2 | keep |
| cat | 2 | keep |
| sat | 2 | keep |
| ran | 1 | cut |
| a | 1 | cut |
| dog | 1 | cut |

Order the kept words by **descending count**, then by the word. `cat`, `sat` and `the` tie at 2, so they sort alphabetically: `cat` $\to$ 4, `sat` $\to$ 5, `the` $\to$ 6. Then:

- `encode("the dog sat")` $= [6, 1, 5]$. `dog` is cut, so it becomes the `<unk>` id 1.
- `decode([6, 1, 5])` $=$ `"the <unk> sat"`.

The round trip lost `dog`. That loss is the price of a fixed vocabulary, and it is the thing to check in any tokenizer you meet.

## Why it works

The id is an index into a table, so **the mapping must be reproducible**. If the same corpus gives a different id order on another run, a trained model silently reads the wrong row for every word. Three rules make the order deterministic: the specials come first in a fixed order, words are ordered by count, and ties are broken by the word itself. The tie-break exists because a dictionary's iteration order is not part of the contract.

Why a cutoff? Each kept word costs one embedding row, and a rare word has too few examples to train its row well.

Byte-level fallback removes `<unk>` altogether. The base alphabet is the 256 byte values, which cover any string. A code-point base vocabulary would start above 130,000 symbols [corpus: library-papers/radford-gpt2-2019.pdf.md p.4]. The price is that one non-ASCII character can take several bytes, so decoding must work on bytes (see the next topics).

## Signal vs. noise

- **Signal:** the reserved ids, the cutoff rule (`>=`, not `>`), the id order, and what happens to an unseen word. Those decide almost every wrong answer.
- **Signal:** the trade-off. Ask "how many tokens does this text need, and what is $|V|$?" before anything else.
- **Noise for now:** the exact byte counts of different scripts, and the vocabularies of published models.

## Where it breaks

- **Case and punctuation.** Under this contract `Cat`, `cat` and `cat.` are three words. Each takes its own row, and each rare variant becomes `<unk>`.
- **Whitespace is lost.** `decode(encode(text))` equals `" ".join(text.split())`, not the original text.
- **Names, numbers and code.** These are the common `<unk>` cases. Measure the `<unk>` rate on held-out text.
- **Reserved strings in the input.** A user can type `<unk>` or `<eos>`. The tokenizer must map that text to `<unk>`, never to a control id.

## Interview angle

*60-second answer:* "A word tokenizer is a learned dictionary with a shared `<unk>` fallback. It is short, but it loses every word outside the vocabulary. A character tokenizer never loses input, but it makes sequences much longer. Subword and byte-level tokenizers sit between them. I'd fix the reserved ids first, make the id order deterministic, and measure the `<unk>` rate on held-out data."

*Likely follow-ups:* Why not a character tokenizer? (Longer sequences cost attention and context.) Why is an `<unk>` id a risk? (It hides the loss, so you measure its rate.)

## Connections

- **Requires:** `algo/hash-maps-and-sets` (the count dictionary and the id lookup).
- **Leads to:** `llm/bpe-merge-training`, which learns subword units instead of fixed words, and `llm/bpe-encode-and-decode`, which applies them.
- **Lineage:** Sennrich, Haddow and Birch (2016) describe the trade-off between vocabulary size and text length [corpus: library-papers/1508.07909.pdf.md p.3]. They offer a fixed-size vocabulary of variable-length sequences as an open vocabulary [corpus: library-papers/1508.07909.pdf.md p.2].
