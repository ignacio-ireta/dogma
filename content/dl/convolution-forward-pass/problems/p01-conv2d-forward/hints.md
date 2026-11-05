## Hint 1 — concept

A convolution layer slides one small filter per output channel over the input and sums over all
input channels at each position. The filter is applied as it is stored; deep learning does not flip
it. Output size comes from how many full windows fit, so the division rounds down.

## Hint 2 — formula / approach

Pad the input with zeros first, to size $H+2p$ by $W+2p$. The output has $\lfloor (H+2p-k)/s\rfloor+1$
positions per axis. For each kernel offset $(u,v)$, the positions it touches are rows $u, u+s, u+2s, \ldots$
and the same for columns. For each such offset, mix the channels with $W[:,:,u,v]$, and add the bias.

## Hint 3 — code shape

Work in four blocks, in this order.

1. **Pad.** Compute $H_{out}$ and $W_{out}$ first, and raise `ValueError` if either is below 1. Make a
   zero tensor of shape $(N, C_{in}, H+2p, W+2p)$ and copy `x` into its centre.
2. **Output.** Make a zero tensor of shape $(N, C_{out}, H_{out}, W_{out})$ in `x`'s dtype.
3. **Kernel offsets.** Loop over $u \in [0, k_H)$ and $v \in [0, k_W)$. For each pair, slice the padded
   input with step $s$ on both spatial axes, so the slice has shape $(N, C_{in}, H_{out}, W_{out})$.
   Its axes are batch, input channel, output row and output column.
4. **Channel mix.** For the same pair, `weight[:, :, u, v]` has shape $(C_{out}, C_{in})$. Contract the
   input-channel axis of the slice with the input-channel axis of that weight slice. The result has
   shape $(N, C_{out}, H_{out}, W_{out})$. Add it to the output.

After the loops, reshape `bias` to $(1, C_{out}, 1, 1)$ and add it, if a bias is given. Check the final
shape before returning.
