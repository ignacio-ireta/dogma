# Interview notes — InfoNCE contrastive loss

## Q1. Write the InfoNCE loss for a batch of paired embeddings.

**60-second answer.** Normalise both sides so dot products are cosines. Build the N by N matrix of similarities divided by the temperature. The diagonal holds the true pairs. Apply cross-entropy along each row with label i, which asks each anchor to find its partner among the N candidates, and along each column for the other direction. Average the two. Each anchor has N minus 1 negatives from the same batch, so the loss needs no extra sampling.

**Follow-ups.** Why normalise? Without it, the loss depends on embedding lengths rather than direction. What is the chance level? All similarities equal gives a loss of log N, so a loss above that is worse than guessing.

## Q2. What does the temperature do, and what goes wrong at the extremes?

**60-second answer.** The temperature divides the cosine similarities, so it sets how far apart the logits are. A small temperature sharpens the softmax, and hard negatives dominate. A very small one makes the logits large, so the log-sum-exp must be computed stably. A large temperature flattens the softmax, the loss stays near log N, and the gradients are weak. CLIP learns the temperature and clips the logit scale at 100, which means a temperature no smaller than 0.01.

**Follow-ups.** Why is it a fixed value in the exercise? It keeps the loss a pure function of the inputs and makes the oracle comparison exact. What is the logit scale? The inverse of the temperature.

## Q3. Why does a larger batch help, and why does it complicate comparisons?

**60-second answer.** Each anchor sees N minus 1 negatives, so a larger batch gives more negatives and a harder task. CLIP trained with a batch of 32,768. The complication is that the chance level, log N, depends on N. A loss value means little without the batch size. The fair comparison is a retrieval metric such as recall at k, computed on a fixed set of candidates.

**Follow-ups.** Does more negatives always help? Only if they are informative. Duplicate or near-duplicate items in a batch become false negatives. Does memory limit it? Yes: the N by N matrix grows with N squared.

## Common mistakes

- Skipping the normalisation, so the loss depends on vector lengths.
- Using only one direction of the cross-entropy.
- Computing the softmax naively at a small temperature.
- Reporting the loss without the batch size.
