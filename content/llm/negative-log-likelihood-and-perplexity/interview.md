# Interview notes — Negative log-likelihood and perplexity

## Q1. Define the loss for a causal language model and how perplexity follows from it.

**60-second answer.** For each non-ignored position, the NLL is the log-sum-exp of the logits minus the logit of the true next token. I average it over the non-ignored tokens and exponentiate the average. That is perplexity, in nats. The log-sum-exp is computed with the row maximum subtracted, so large logits do not overflow. Perplexity reads as the number of effective choices per token: a uniform guess over four tokens has perplexity four.

**Follow-ups.** Why nats? Natural logs make the exponential the inverse of the loss, so exp of the mean loss is perplexity. Bits give the same perplexity with base two. What does a perplexity of 1 mean? The model puts all its probability on the true token.

## Q2. Why must the average use only the non-ignored tokens?

**60-second answer.** The denominator is the number of tokens the model is actually scored on. Padding positions and prompt positions that are masked out are not predictions of the text, so they must not count. Dividing by the batch size or by B times T lets padding lower the loss, so the reported perplexity improves without any change in the model.

**Follow-ups.** How does this match cross-entropy with `ignore_index`? PyTorch's `reduction="mean"` divides by the number of non-ignored targets, so the two agree. What if every target is ignored? The mean is undefined, and the function should raise rather than return NaN.

## Q3. Can you compare the perplexity of two models with different tokenizers?

**60-second answer.** Not directly. Perplexity is per token, and a tokenizer that splits text into more tokens gives each token a smaller share of the information. The fair comparison uses a common unit: total NLL in bits divided by the number of bytes or characters of the same held-out text. That is the unit the GPT-2 paper relies on when it reports per canonical unit.

**Follow-ups.** Does a lower perplexity on training text show a better model? Not on its own. Use held-out text. Do de-tokenizers matter? They change the reported perplexity (the GPT-2 paper reports gains of 2.5 to 5 perplexity from them), so the evaluation text must be processed the same way for both models.

## Common mistakes

- Averaging over B times T, which counts padding.
- Using a naive softmax, which overflows for large logits.
- Comparing perplexity across tokenizers without normalising to bytes or characters.
- Reporting `inf` or `NaN` instead of raising when no token is scored.
