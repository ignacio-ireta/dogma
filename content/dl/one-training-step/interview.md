# Interview notes: One training step

## Q1. Walk me through one training step.

**60-second answer.** Clear the old gradients, run the forward pass on the current parameters, reduce to one scalar loss, call backward so each parameter's gradient describes that loss, then update the parameters by minus the learning rate times the gradient. The update runs outside the graph. The order matters because a gradient is valid only for the parameters that produced it. I would return the loss from the forward pass, not from after the update.

**Follow-ups.** Why does the update run under `no_grad`? Recording it would grow the graph every step and make the next backward differentiate through the update. Where does momentum go? In the update line only, since it changes the step, not the gradient.

## Q2. The loss stops falling after a few epochs. What do you check first?

**60-second answer.** I check the cheap failure modes in order. First, that gradients are cleared every step, because an accumulated gradient looks like a learning rate that grew without bound. Second, that the update is applied to the same parameter objects that the optimizer and the forward pass use. Third, that the learning rate and the loss scale are sane. I reproduce a single step against a reference optimizer on the same batch before I tune anything.

**Follow-ups.** How do you test the training step itself? Compare the parameters after several steps with `torch.optim.SGD` on a deep copy, and call the step twice on one batch to expose stale gradients. Why not trust the loss curve? It can go down while the update is wrong, for example with a sign error that a small learning rate hides.

## Common mistakes

- Clearing gradients after `backward()`, so the update uses nothing.
- Updating parameters with autograd on, which raises an error on leaves or grows the graph.
- Creating new `Parameter` objects during the update, which orphans the optimizer's references.
- Returning the post-update loss, so the reported curve lags by one step.
