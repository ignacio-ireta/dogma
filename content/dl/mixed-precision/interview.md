# Interview notes — Mixed precision

## Q1. Walk me through one fp16 training step with dynamic loss scaling.
**60-second answer.** The master weights stay in fp32. The forward pass runs in fp16 under autocast, and the loss is multiplied by the scale $S$ before the backward pass, so small gradients are lifted out of the subnormal range. Before the optimizer step the gradients are divided by $S$. If any gradient is infinite or NaN, the step is skipped and $S$ is halved. After $n$ consecutive good steps, $S$ doubles. The skip is what makes the rule safe, because an overflowing gradient never reaches the weights.
**Follow-ups.**
- *What does a skipped step cost?* One batch of progress. Many skips in a row mean $S$ is too high for the current gradients.
- *Where do you unscale relative to clipping?* Unscale first, then clip, then step.

## Q2. Why does bf16 usually not need loss scaling?
**60-second answer.** bf16 has the same eight-bit exponent as fp32, so its range is about $3.4\times10^{38}$ and its smallest subnormal is around $10^{-40}$. Gradients of typical size do not underflow. The price is precision: bf16 has seven mantissa bits, so each rounding costs about $0.4\%$. That is why the master weights stay in fp32 even when bf16 is used for computation.
**Follow-ups.**
- *Does bf16 remove the need for fp32 weights?* No. An update of $10^{-4}$ to a weight near $1$ is smaller than bf16's spacing there, so the update would be lost.
- *Which operations need care?* Reductions, softmax and normalisations. Check their output dtypes, because autocast does not always promote them.
- *Does a wider format fix cancellation?* It shrinks the stored error of each input, but the subtraction still amplifies whatever error is there. Reformulate the subtraction when the operands are nearly equal.

## Q3. How would you tell whether a loss-scaling run is healthy?
**60-second answer.** Watch the scale and the skip count. A healthy run settles where the scale oscillates slowly around its safe value, with rare skips. A scale that keeps halving points to a bad input or a learning rate that makes gradients explode. A run with no skips should double its scale once per growth interval. If it has not doubled after several intervals with no skips, `update` is probably not being called on every step. Log the loss in fp32 and compare it with a short fp32 run on the same batches.
**Follow-ups.**
- *What is the first thing to try if the loss becomes NaN?* Check whether the scale is still falling. If it is, look at the gradients, not the scaler.
- *Why compare with fp32?* It separates a precision problem from a modelling problem.

## Common mistakes
- Clipping gradients before unscaling.
- Saying bf16 needs no fp32 copy of the weights.
- Assuming autocast keeps softmax and normalisation in fp32 without checking the output dtype.
- Treating a skipped step as an error. It is part of the algorithm.
