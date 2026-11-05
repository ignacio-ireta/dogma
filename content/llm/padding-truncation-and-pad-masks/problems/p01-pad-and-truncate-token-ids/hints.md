## Hint 1 — concept

Think in three steps: keep part of each row, fill the rest with the pad id, then mark which columns hold kept tokens. The mask is a fact about lengths. It is not a fact about what the token ids look like, so a real token that happens to equal the pad id still counts as real.

## Hint 2 — formula / approach

For a row with n tokens and a width W, the number of kept tokens is k = min(n, W). Truncation keeps the first W tokens for the right side and the last W for the left side. Padding puts the k kept tokens in columns 0 to k-1 for right padding, or in columns W-k to W-1 for left padding. The mask is 1 in exactly those k columns.

## Hint 3 — code shape

Allocate `input_ids` filled with `pad_id` and `attention_mask` filled with zero, both int64 with shape (B, W). For each row, slice to get the kept tokens, compute `start` from the padding side, and assign the kept tokens and a 1 into the slice `start:start + k`. Validate every argument before you allocate anything.
