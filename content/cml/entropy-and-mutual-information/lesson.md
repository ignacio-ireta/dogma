---
title: Entropy and mutual information
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Let $X$ and $Y$ be discrete random variables. A distribution is a vector $\mathbf p=(p_1,\dots,p_k)$ with $p_i\ge 0$ and $\sum_i p_i=1$. For the usual information-theory properties below, take $b>1$. The log base $b$ sets the unit: $b=2$ gives **bits**, $b=e$ gives **nats**.

- **Surprise** of an outcome with probability $p$ is $-\log_b p$. Rare outcomes are more surprising.
- **Entropy** is the average surprise:
$$H(\mathbf p)=-\sum_i p_i\log_b p_i,\qquad 0\log 0:=0.$$
An outcome that never happens adds nothing. Range: $0\le H\le\log_b k$, zero for a certain outcome, maximal for a uniform one.
- **Cross-entropy** of a model $\mathbf q$ against the truth $\mathbf p$ is $H(\mathbf p,\mathbf q)=-\sum_i p_i\log q_i$. It splits as
$$H(\mathbf p,\mathbf q)=H(\mathbf p)+D_{\mathrm{KL}}(\mathbf p\,\|\,\mathbf q),$$
where the KL divergence is non-negative and zero only when $\mathbf q=\mathbf p$. See `cml/cross-entropy`.
- **Mutual information** (returned in **nats** by the problem) of two discrete variables, from the joint probabilities $p(a,b)$:
$$I(X;Y)=\sum_{a,b}p(a,b)\log\frac{p(a,b)}{p(a)\,p(b)}=H(X)+H(Y)-H(X,Y).$$
It is symmetric, non-negative, zero exactly when $X$ and $Y$ are independent, and at most $\min(H(X),H(Y))$.
- **Information gain** of a split $S$ on a label $Y$ is $IG=H(Y)-\sum_s p(s)\,H(Y\mid S=s)=I(S;Y)$. A tree split that maximises information gain is the entropy version of the split rule in `cml/choosing-a-tree-split`.

## Worked example

**Surprise and entropy.** For $\mathbf p=(0.5,0.25,0.25)$ the surprises in bits are $1,\,2,\,2$. The average is $0.5\cdot1+0.25\cdot2+0.25\cdot2=1.5$ bits. A fair coin gives $H=1$ bit, and a uniform distribution over four outcomes gives $\log_2 4=2$ bits.

**Cross-entropy.** Take $\mathbf p=(0.5,0.5)$ and $\mathbf q=(0.9,0.1)$. Then $H(\mathbf p)=1$ bit, $H(\mathbf p,\mathbf q)=1.737$ bits, and $D_{\mathrm{KL}}=0.737$ bits. Check: $1+0.737=1.737$.

**Information gain.** Eight samples have labels $y=(1,1,1,0,0,0,0,0)$, so $H(Y)=H(3/8)=0.954$ bits. A split sends the first four samples left, with labels $(1,1,1,0)$, and the last four right, with labels $(0,0,0,0)$. The left child has $H(3/4)=0.811$ bits and the right child has $0$. So

$$IG=0.954-\tfrac12(0.811)-\tfrac12(0)=0.549\text{ bits}.$$

In nats this is $0.380$, which matches `sklearn.metrics.mutual_info_score` on the same two label sequences (setup: exact arithmetic, checked by running both).

**Correlation versus mutual information.** Let $X$ be uniform on $\{-1,0,1\}$ and $Y=X^2$. The Pearson correlation is exactly $0$. But $Y$ is determined by $X$, so $I(X;Y)=H(Y)=H(1/3,2/3)=0.637$ nats $(0.918$ bits$)$. A sample of 900 rows, with 300 rows at each value of $X$, reproduces both numbers (the sample version gives $0.6365$ nats, matching the exact value).

## Why it works

**Average code length.** Shannon (1948) showed that $H$ is a lower bound on average lossless code length in bits (base 2), approached per symbol by coding sufficiently long blocks. Single-symbol prefix codes generally require an integer-length overhead. Cross-entropy is the expected ideal code length when the code is built for $\mathbf q$. The KL term is the extra cost of using the wrong code. This is why a classifier trained with log-loss is fitting a code length.

**Why KL is non-negative.** Jensen's inequality for the concave logarithm gives $-D_{\mathrm{KL}}(\mathbf p\|\mathbf q)=\sum_i p_i\log(q_i/p_i)\le\log\sum_i q_i=0$, with equality only when $\mathbf q=\mathbf p$ on the support of $\mathbf p$.

**Why MI is a log-ratio average.** Each term $\log\frac{p(a,b)}{p(a)p(b)}$ is zero when the pair matches independence, so MI measures how far the joint is from the product of its marginals. See Cover & Thomas, chapter 2.

## Signal vs. noise

- **Signal:** entropy is never negative for a valid distribution, and it is zero only for a certain outcome.
- **Signal:** mutual information is zero exactly under independence, and equals information gain for a split.
- **Signal:** the base only rescales the unit, so rankings of splits do not depend on it.
- **Noise for now:** the coding theorem's proof, typical sets, and continuous (differential) entropy.

## Where it breaks

- **$0\log 0$.** Writing `p * log(p)` on a vector with zeros gives `nan`. Drop the zero entries before taking logs.
- **Invalid distributions.** A vector summing to $1.02$ must be rejected, not silently normalised.
- **Plug-in MI is biased upward on small samples.** The plug-in estimate uses the observed frequencies. With $X$ and $Y$ independent and uniform over $k=10$ categories each, 50 rows give a mean estimate of $0.816$ nats over 2000 draws (`default_rng(0)`). The asymptotic prediction is $(k-1)^2/(2n)=0.81$. Many categories with few rows look informative by chance.
- **Continuous variables need binning**, and the bin count changes the value.
- **Information gain favours many-valued features**, for the same reason.

## Interview angle

*60-second answer:* Entropy is the average surprise, in bits for base 2. Cross-entropy is the classifier loss, equal to entropy plus a KL divergence. Mutual information measures how much one variable reduces uncertainty about the other, and it catches non-linear dependence that correlation misses. For a tree, the information gain of a split is the mutual information between the split and the label. Plug-in estimates are biased upward on small samples, so I check them on held-out data or against a shuffled baseline.

*Likely follow-ups:* Why do trees usually use Gini instead of entropy? (Both measure impurity; Gini needs no logarithm and gives nearly the same splits.) What goes wrong with mutual information on 50 rows and 10 categories? (The plug-in estimate is about $(k-1)^2/(2n)$ nats even under independence.)

## Connections

- **Prerequisites:** `cml/probability-basics` (distributions, joint and marginal probabilities).
- **Unlocks:** `cml/cross-entropy` (the classification loss), `cml/choosing-a-tree-split` (impurity and information gain), and the feature-selection discussion in `cml/random-forests-and-feature-importance`.
- **Lineage:** Shannon (1948) defined entropy for communication; Cover & Thomas, chapter 2, gives the standard development.
- **References:** [Cover & Thomas, chapter 2](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X) develops entropy, relative entropy and mutual information.
