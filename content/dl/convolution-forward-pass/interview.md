# Interview notes — Convolution forward pass

## Q1. What is the output shape of a convolution layer, and how many parameters does it have?

**60-second answer.** The input is $(N, C_\text{in}, H, W)$ and the kernel is $(C_\text{out}, C_\text{in}, k_H, k_W)$. With padding $p$ and stride $s$, each spatial axis becomes $\lfloor(H+2p-k)/s\rfloor+1$. The layer has $C_\text{out}C_\text{in}k_Hk_W+C_\text{out}$ parameters, which does not depend on $H$ or $W$. I check the floor first, since rounding up is the usual shape bug.

**Follow-ups.** What if the kernel does not fit? The output size would be below one, so the input is invalid. What does the stride do to the size? It divides it, by about $s$.

## Q2. Why is a convolution better than a dense layer on an image?

**60-second answer.** Two properties. The kernel is shared across positions, so the layer is equivariant to translation and needs far fewer parameters. A $32\times32\times3$ input to a 16-channel $3\times3$ conv has 448 parameters. A dense layer to the same output shape needs about $5\times10^7$. The locality of the kernel also matches the structure of images, where nearby pixels are related.

**Follow-ups.** Is it invariant to any translation? No. It is equivariant: the output shifts with the input. The shift has to be a multiple of the stride, and content near the padded border changes. Pooling or global averaging adds invariance on top.

## Q3. How would you implement a convolution efficiently?

**60-second answer.** I would unfold each receptive field into a column with im2col. That turns the layer into one matrix product, $W_\text{mat}\,\mathrm{col}+b$, which BLAS handles well. The cost is $N\cdot C_\text{out}\cdot C_\text{in}\cdot k_H k_W\cdot H_\text{out}W_\text{out}$ multiply-adds. For large kernels, FFT-based methods can be cheaper. I would test the implementation against a reference convolution on asymmetric kernels and non-square inputs.

**Follow-ups.** Why not just loop over pixels? That is correct but slow in Python. What is the memory cost of im2col? The unfolded matrix repeats each input value up to $k_Hk_W$ times.

## Common mistakes

- Using $\lceil\cdot\rceil$ instead of $\lfloor\cdot\rfloor$ for the output size.
- Flipping the kernel when the convention is cross-correlation, or failing to flip when a textbook formula is required.
- Padding only one side of the input.
- Counting the parameters as $C_\text{in}HW$ instead of the kernel size.
