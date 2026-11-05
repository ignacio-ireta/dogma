# Interview notes — LoRA adapters

## Q1. How does LoRA fine-tune a model, and what is trained?
**60-second answer.** LoRA freezes the pretrained weight W and learns a low-rank update, scaled by alpha over
r. The update is B A, with A of shape r by d_in and B of shape d_out by r. B starts at zero, so training starts
at the base model. Only A and B get gradients. At inference I can merge the update into W, which costs nothing,
or keep it separate to swap adapters per request. The trade-off is capacity: the update can only have rank r.
**Follow-ups.** Why zero-initialise B and not A? Zero for both gives zero gradient. Zero B makes the first output
the base output, and a random A lets gradients reach B. Why does merging matter? A merged model has the base
architecture and the same latency, but one merged copy serves one adapter. Does LoRA assume the pretrained weights are low rank? No. The update is low rank by construction, and the rank is tuned against task results.

## Q2. What does the rank r buy, and what does it cost?
**60-second answer.** The trainable parameters per adapted matrix are r times (d_in + d_out), against d_in times
d_out for full fine-tuning. A small r cuts memory and storage and limits how far the weights can move. A large r
approaches full fine-tuning. I pick r by validation, not by the formula alone.
**Follow-ups.** What happens to alpha when r changes? The scale alpha over r changes with it, so retune alpha
or hold it fixed deliberately. Which matrices should get adapters? That is a configuration choice. The
attention projections are common starting points.

## Common mistakes
- Initialising both A and B to zero, so nothing trains.
- Scaling the update by alpha alone, without dividing by r.
- Detaching the adapter, or merging in place into the frozen weight.
