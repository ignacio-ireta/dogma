---
title: "Convolution forward pass"
minutes: 12
min_viable: "Read The system and Worked example, then do p01 using Hint 1 if needed."
---

## The system

A convolutional layer maps a batch $X\in\mathbb{R}^{N\times C_\text{in}\times H\times W}$ to $Y\in\mathbb{R}^{N\times C_\text{out}\times H_\text{out}\times W_\text{out}}$. Here $N$ is the batch size, $C_\text{in}$ and $C_\text{out}$ are input and output channel counts, and $H, W$ are the spatial sizes. The layer has one kernel per output channel, $W\in\mathbb{R}^{C_\text{out}\times C_\text{in}\times k_H\times k_W}$, and one bias per output channel, $b\in\mathbb{R}^{C_\text{out}}$.

Four rules define the forward pass:

| Rule | Statement |
|---|---|
| Operation | Deep learning's "convolution" is **cross-correlation**. The kernel slides over the input without flipping |
| Output size | $H_\text{out}=\lfloor (H+2p-k_H)/s\rfloor+1$, and the same for $W$. Floor rounds down, so a partial window at the edge is dropped |
| Channel mixing | Each output channel sums over **all** $C_\text{in}$ input channels, with its own filter |
| Bias | $b_o$ is added once to every position of output channel $o$ |

Each output value is

$$Y[n,o,i,j]=b_o+\sum_{c}\sum_{u=0}^{k_H-1}\sum_{v=0}^{k_W-1} X_\text{pad}[n,c,\,is+u,\,js+v]\;W[o,c,u,v],$$

where $s$ is the stride and $X_\text{pad}$ is $X$ with $p$ zeros on each side of $H$ and $W$.

## Worked example

Take $N=1$, $C_\text{in}=C_\text{out}=1$, $H=W=3$, $k=2$, $s=1$, $p=0$. The input rows are $[0,1,2]$, $[3,4,5]$, $[6,7,8]$. The output size is $\lfloor(3-2)/1\rfloor+1=2$ per axis.

Let the kernel be $W=\begin{pmatrix}1&0\\0&-1\end{pmatrix}$. The top-left window is $\begin{pmatrix}0&1\\3&4\end{pmatrix}$. Its output is $0\cdot1+4\cdot(-1)=-4$. The kernel is nonzero only at its $(0,0)$ and $(1,1)$ entries, so each output is the window's top-left value minus its bottom-right value: $0-4$, $1-5$, $3-7$, $4-8$. All four are $-4$. Now flip the kernel to $\begin{pmatrix}-1&0\\0&1\end{pmatrix}$. Every output becomes $+4$. So the flip changes the sign, and a flipped kernel computes true convolution, not the layer the library implements.

Parameter count: $C_\text{out}\cdot C_\text{in}\cdot k_H\cdot k_W+C_\text{out}$. For this layer that is $1\cdot1\cdot4+1=5$.

## Why it works

**Equivariance.** The same kernel is applied at every position. If the input shifts by one pixel, the output shifts by one pixel. That holds exactly for stride 1 and away from the padded border. For stride $s$ it holds for shifts that are multiples of $s$. That is what makes the layer useful for images.

**Parameter sharing.** Consider a $32\times32\times3$ input, a $3\times3$ kernel with 16 output channels and padding 1. The output has the same spatial size, and the layer has $16\cdot3\cdot3\cdot3+16=448$ parameters. A dense layer from the same input to the same output shape would need $(32\cdot32\cdot3)\cdot(32\cdot32\cdot16)\approx5.0\times10^7$ weights. The convolution reuses one small kernel everywhere, so its count does not depend on $H$ or $W$.

**im2col.** Flatten each receptive field into a column. Let $\mathrm{col}$ have shape $(C_\text{in}k_Hk_W)\times(H_\text{out}W_\text{out})$, with one column per output position. Then $Y = W_\text{mat}\,\mathrm{col}+b$, where $W_\text{mat}$ is $W$ reshaped to $C_\text{out}\times(C_\text{in}k_Hk_W)$. This is a dense layer applied to patches. It explains the cost: $N\cdot C_\text{out}\cdot C_\text{in}\cdot k_H k_W\cdot H_\text{out}W_\text{out}$ multiply-adds.

Flipping does not matter when weights are learned, because the network learns the flipped kernel. It matters when you match a textbook formula or a fixed kernel. Use cross-correlation consistently.

## Signal vs. noise

Signal: the output-size formula, the channel-mixing rule, the bias per channel, and the cross-correlation definition. Noise: FFT and Winograd implementations, dilation and groups, and transposed convolution. Those are ways to compute or extend the same layer. Learn them after the forward pass is exact.

## Where it breaks

Five errors show up in practice:

1. **Rounding up.** Using $\lceil\cdot\rceil$ gives an output that asks for a window past the edge. For a width-6 input with a width-3 kernel and stride 2, the answer is $\lfloor 3/2\rfloor+1=2$, not 3.
2. **Flipped kernel.** Correct for a textbook formula, wrong for the layer's convention.
3. **One-sided padding.** Padding only the bottom and right shifts the windows, so the values change. The size formula also breaks, because it assumes $p$ on both sides.
4. **Channel axes.** The kernel's second axis must equal $C_\text{in}$. A transposed kernel fails only when $C_\text{in}\ne C_\text{out}$, which makes the bug easy to miss on square tests.
5. **Bias on the wrong axis.** The bias has length $C_\text{out}$. Broadcasting it against $H$ or $W$ works when those sizes happen to equal $C_\text{out}$.

## Interview angle

Give the output shape first, then the parameter count, then the reason for parameter sharing. A likely follow-up asks how convolution is implemented. Answer with im2col, which reduces it to a matrix multiplication, and mention that FFT methods exist for large kernels. Another asks why CNNs are better than dense layers on images. Answer with equivariance and parameter sharing.

## Connections

The layer builds on the matrix product in [matrix multiplication (`cml/matrix-multiplication`)](../../cml/matrix-multiplication/lesson.md) and on the affine map of [the dense layer forward pass (`dl/dense-layer-forward-pass`)](../dense-layer-forward-pass/lesson.md). It unlocks pooling, CNN training and image architectures in the track. Sources: [paper: LeCun et al. 1998, Gradient-Based Learning Applied to Document Recognition] and [book: Goodfellow, Bengio and Courville, Deep Learning, chapter 9].
