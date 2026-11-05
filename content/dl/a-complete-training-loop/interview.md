# Interview notes: A complete training loop

## Q1. Walk me through a training loop, and how would you make two runs comparable?

**60-second answer.** For each epoch I draw one permutation of the training rows from a seeded generator, cut it into mini-batches (keeping the short last one), and run a training step on each batch. I log the training loss as the batch-size-weighted mean of the losses measured before each step. After the epoch I switch to eval mode and measure validation loss under no-grad. I keep the state with the lowest validation loss. Runs are comparable when they share the seed, the initial parameters, the optimizer, the loss and the shuffle order.

**Follow-ups.** Why must the shuffle draw be the only randomness? Any other draw from the same generator changes every later epoch. How do you compare two batch sizes fairly? Fix the epoch count, the seed and the learning rate schedule, and report on the same validation set. Note that a larger batch means fewer updates per epoch, so the comparison is not only about speed.

## Q2. The validation loss went up after the best epoch. Which weights do you ship?

**60-second answer.** The weights from the best validation epoch, the ones the checkpoint saved, not the last epoch. Rising validation loss after a minimum is the overfitting signal. The checkpoint rule keeps the earliest state at a tie, so the result is deterministic. The validation set has now been used to select a model, so the score I report must come from a separate test set.

**Follow-ups.** Why not just stop at the first increase? One noisy uptick would stop training too early, so early stopping uses a patience window. How do dropout and batch norm affect this? They must be in eval mode during validation, otherwise the validation loss is noisy and the checkpoint choice is random.

## Common mistakes

- Dropping the partial last batch, so a short dataset loses rows.
- Averaging batch losses without weights, so the last short batch counts as much as a full one.
- Measuring validation in train mode, so dropout changes the checkpoint choice.
- Reporting the best validation loss as the final score, which is optimistic.
