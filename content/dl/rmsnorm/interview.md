# Interview notes — RMSNorm

## Q1. What does RMSNorm change relative to layer normalization, and why?

**60-second answer.** Layer normalization centers each example and then scales it by its standard deviation. RMSNorm skips the centering and divides each example by its root mean square, then applies a gain. The saving is one reduction and one subtraction per row, and the scale invariance that the normalization is for remains. I would present it as a trade-off: fewer operations, a different invariance, and no shift term in the standard form.

**Follow-ups.** Why does it not need a bias? The shift was what centering removed, so a bias term would add back a degree of freedom without a clear purpose. Is it always better? I would not claim that. It is a design choice that several modern language models adopt. Whether it is better depends on the architecture and the training recipe, and I would check the paper rather than quote a number.

## Q2. Where does eps go, and why does its position matter?

**60-second answer.** eps belongs inside the square root, with the mean square. Placing it outside changes the scale of the output for small inputs: an input of order $10^{-3}$ with eps of order $10^{-6}$ gets normalized to a different magnitude. I check this with a test that uses small magnitudes, because the default behaviour on typical activations hides the difference.

**Follow-ups.** Why not just use a huge eps? It flattens the output for all inputs, not only small ones. What about overflow for very large inputs? Squaring is the risk. The tests here use magnitudes up to $10^{100}$ in float64, where the square stays finite; float32 overflows far sooner, at about $3.4\times10^{38}$ in the square.

## Common mistakes

- Subtracting the mean, which turns RMSNorm into layer normalization without the bias.
- Placing eps outside the square root.
- Adding a bias term and calling the result RMSNorm, which changes the function.
