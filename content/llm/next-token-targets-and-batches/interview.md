# Interview notes — Next-token targets and batches

## Q1. How do you build (inputs, targets) for a causal language model from raw documents?

**60-second answer.** Tokenize each document, append an EOS id, and concatenate. Call the stream S with N tokens. Block size T gives a window of T+1 tokens with stride T. Inputs are the first T tokens of the window, and targets are the last T, so targets are inputs shifted left by one. The number of full blocks is floor((N-1)/T). The leftover tokens are dropped, and that rule is stated in the contract.

**Follow-ups.** Why stride T and not T+1? Stride T+1 would skip one target per block. Why an EOS? It marks where a document ends, so the model can learn to stop.

## Q2. Packing puts several documents in one block. What goes wrong?

**60-second answer.** Plain causal attention lets a token attend to every earlier token in the block. The first tokens of the second document can then read the first document, which is unrelated context. The model learns a dependency that does not exist at inference time, when each prompt is alone. Packing is still used because it wastes no compute on pads, and the fix is a document mask that blocks attention across the EOS boundary.

**Follow-ups.** Is the EOS enough on its own? It teaches the model where a document ends, but it does not block attention. Does it change the loss? The loss formula is unchanged, but the first tokens of each later document are predicted from context that belongs to another document.

## Q3. What do you do with a tail that does not fill a block?

**60-second answer.** Either drop it or pad it. Dropping loses at most T-1 tokens per epoch and keeps every block full. Padding keeps every token but needs targets at pad positions set to -100, so the loss ignores them. Either choice is fine. What matters is that the contract says which one, because the number of training tokens depends on it.

## Common mistakes

- Targets equal to inputs (no shift).
- Concatenating documents without an EOS, so they run together.
- Using stride T+1 and silently losing one target per block.
- Padding a tail with zeros and leaving those zeros as targets.
