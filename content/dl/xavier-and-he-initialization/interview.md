# Interview notes — Xavier and He initialization

## Q1. How do you choose Xavier versus He initialization?

**60-second answer.** I state the activation, weight layout and fan convention first. Xavier balances fan-in and fan-out with variance 2/(fan_in+fan_out) under approximate linear assumptions. He-normal uses variance 2/fan_in to compensate for rectification in a forward fan-in convention. These scales support reasonable signal flow initially; they do not guarantee successful training for every architecture.

**Follow-ups.**

- What does NumPy normal scale expect? Standard deviation, so take the square root of the variance target.
- What is the Xavier uniform bound? sqrt(6/(fan_in+fan_out)).

## Q2. Why is the sample variance different from the target?

**60-second answer.** The target belongs to the distribution, while a finite random matrix has sampling variation. I do not rescale each draw to force its sample moments, because that changes the specified initialization. To verify a small implementation I compare seeded base draws with independently computed scale, rather than relying on a fragile empirical moment threshold.

**Follow-ups.**

- How do you isolate randomness? Create a fresh local default_rng(seed), leaving global random state untouched.
- Why avoid identical weight vectors? Symmetric units can receive identical updates and keep learning the same feature.

## Common mistakes

- Passing variance as a normal standard deviation.
- Using output width where the contract specifies fan-in.
- Resetting global random state inside an initialization helper.
