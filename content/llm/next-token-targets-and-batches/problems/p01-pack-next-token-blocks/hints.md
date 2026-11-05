## Hint 1 — concept

Build one long token stream first, with an end-of-sequence id after every document, and only then cut it into blocks. Each input token is paired with the token that follows it in that stream, so the targets are the same stream one position later. Ask yourself how many positions have a next token before you decide how many blocks fit.

## Hint 2 — formula / approach

The stream S has N tokens. A block of size T needs T + 1 consecutive tokens: the first T are its inputs and the last T are its targets. Consecutive blocks start T positions apart, so they share one token at the boundary. The number of full blocks is (N - 1) // T, and any leftover tokens are dropped, not padded.

## Hint 3 — code shape

Build a Python list by extending it with each document and then the EOS id, and take its length. Convert it to one int64 tensor. Make a column of block start positions, `b * T`, and a row of offsets `0..T`. Adding them gives a (B, T + 1) index matrix. Index the stream tensor with it, then take columns `:T` for inputs and `1:` for targets. Handle the zero-block case before indexing.
