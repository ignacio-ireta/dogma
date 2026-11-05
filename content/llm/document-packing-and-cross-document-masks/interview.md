# Interview notes: Document packing and cross-document masks

## Q1. Why pack documents instead of padding each one to the row length?

**60-second answer.** Short documents padded to the row length spend compute on pad positions.
Packing fills each row with whole documents, separated by an end-of-sequence token, so almost
every position is a real token. The price is that attention and the loss must become
document-aware: a token must not read another document, and a target that starts a new document
must not be scored as if it followed the previous one.

**Follow-ups.**
- *Is the waste gone entirely?* No. The last document may not fit, so the row is padded, split
  across rows, or the tail is dropped. Each choice is stated in the data pipeline.
- *Does the EOS token solve the attention problem?* No. It marks where a document ends, but it
  does not stop attention from reading across that point.

## Q2. What mask does a packed batch need, and what goes wrong with a plain causal mask?

**60-second answer.** A block-diagonal causal mask: query $i$ may attend key $j$ only when both
tokens are in the same document and $j \le i$. With a plain causal mask, a token in the second
document can attend the first. In training the model then uses context that is absent when the
document is used alone, so training and inference disagree. The mask is a boolean $T \times T$
matrix, and the two rules are separate and combined with AND.

**Follow-ups.**
- *How do you test it?* Check the one-document row against the plain causal mask, then check a
  multi-document row against per-document causal blocks placed on the diagonal.
- *Cost?* A dense mask is still $T \times T$. Variable-length attention kernels take cumulative
  document lengths instead, and their compute scales with the sum of squared document lengths.

## Q3. What else changes at a document boundary?

**60-second answer.** Two things. Position ids restart at zero at the first token of each
document, so a document sees the same position ids it sees alone. The loss target that crosses
a boundary is set to $-100$, so the model is not asked to predict the first token of an
unrelated document from the last token of the previous one.

**Follow-ups.**
- *Does the position restart matter with RoPE?* Within a document the attention score depends
  only on relative position, so the restart changes little there. It matters for absolute
  position embeddings, where the offset changes the input.
- *How many targets are lost?* One per boundary inside the training window. A window with
  $b$ boundaries has $b$ targets set to $-100$, and the rest keep their normal label.

## Common mistakes

- Calling a causal mask packed. The one-document case passes, and the multi-document case leaks.
- Restarting positions only for the first document in a row.
- Padding the tail with the id of the last real document, so the pads share its run.
- Dropping the boundary loss mask, so the model learns to predict the start of unrelated documents.
- Reusing an id for two documents in one row. The mask then treats them as one document.
