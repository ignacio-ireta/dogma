## Hint 1 — concept

A chunk is a window over the list of words, not over the characters. Decide first what a word is,
then decide how far the window moves each time. The overlap is what remains when you subtract the
step from the window size.

## Hint 2 — formula / approach

Let `step = size - overlap`. Window `j` covers words `[j * step, min(j * step + size, n))`, where
`n` is the word count. Emit a window for each start that is inside the text. The loop must stop
after the window whose end is `n`, because any later window is contained in that one.

## Hint 3 — code shape

Validate the four arguments first, then split the text once. Use a single loop over the start
index: compute the end as the smaller of start plus size and the word count, join that slice,
append it, and break when the end equals the word count. Otherwise advance the start by the step.
