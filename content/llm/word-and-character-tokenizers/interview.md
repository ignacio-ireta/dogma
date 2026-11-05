# Interview notes — Word and character tokenizers

## Q1. How would you tokenize text for a model with a fixed vocabulary?
**60-second answer.** Choose the granularity first. A word tokenizer gives short sequences but needs an
unknown-token id, so every rare word is lost. A character tokenizer never loses input but makes sequences
long. I would reserve the special ids, count the corpus, keep words at or above `min_count` in a deterministic
order, and map everything else to the unknown id. Then I would measure the unknown rate on held-out text
before I trust the tokenizer.
**Follow-ups.** Why not characters? Longer sequences cost attention and context length. How do you avoid
losing words? Use subword or byte-level units, which compose unseen words from known pieces.

## Q2. What does the unknown token cost you?
**60-second answer.** It keeps the interface total, so encoding never fails, but it silently erases
information. The model cannot tell two unknown words apart. The unknown rate is therefore a metric, not a
detail, and I track it on data the vocabulary was not built from.
**Follow-ups.** How would you reduce it? Lower the cutoff, or switch to subwords. Does a bigger corpus
fix it? It usually lowers the rate, but the rate depends on the domain, not only on the size.

## Q3. Why must special tokens be reserved before counting?
**60-second answer.** A special token is a control symbol. If a corpus word can take its id, the model
reads ordinary text as padding or as a boundary. I reserve the ids in a fixed order, skip those strings
while counting, and map any text that spells them to the unknown id.
**Follow-ups.** What if a user types `<eos>` in a prompt? It maps to the unknown id, so it cannot end the
sequence. Why fixed positions? Other code, such as the embedding and the loss mask, looks them up by id.

## Common mistakes
- Using `>` where the cutoff is `>=`, which drops words that occur exactly `min_count` times.
- Counting documents instead of occurrences.
- Letting the id order depend on the order of the corpus.
- Lowercasing silently, so `Cat` and `cat` merge without anyone deciding that they should.
- Treating `decode(encode(text))` as an identity without checking whitespace and unknown words.
