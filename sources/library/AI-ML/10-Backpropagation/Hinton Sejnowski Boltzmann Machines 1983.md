# Boltzmann Machines — Hinton & Sejnowski (1983/1985)

#paper #milestone #read

**Authors:** Geoffrey E. Hinton, Terrence J. Sejnowski (with David H. Ackley for the 1985 journal version)
**Published:** CMU tech report (1983); Cognitive Science 9(1):147–169 (1985)
**PDF:** `papers/ackley-hinton-sejnowski1985.pdf`

---

## The Question

Hopfield nets can store and recall patterns, but they can't *learn* representations. They have no hidden units — what you see is what you get. Can we build a network with hidden layers that learns to discover useful features in data, using the same energy-based framework?

Hinton & Sejnowski's answer: yes — add stochasticity, add hidden units, and derive a learning rule from statistical mechanics.

---

## The Core Idea: Making Hopfield Nets Learn

A Boltzmann machine is a Hopfield network with two modifications:

1. **Stochastic units.** Instead of deterministically flipping to minimize energy, each unit turns on with a probability given by the logistic function:
$$P(s_i = 1) = \frac{1}{1 + e^{-z_i / T}}$$
where $z_i$ is the total input and $T$ is the temperature. At $T = 0$, this reduces to a deterministic Hopfield net. At $T > 0$, the network can escape local minima — it samples from a **Boltzmann distribution** over states.

2. **Hidden units.** The network has *visible* units (data) and *hidden* units (latent features). The hidden units learn representations that capture regularities in the data. This is the first neural network model with a principled approach to learning internal representations.

### The Learning Rule

The beauty: the gradient of the log-likelihood has a stunningly simple form:

$$\Delta w_{ij} \propto \langle s_i s_j \rangle_{\text{data}} - \langle s_i s_j \rangle_{\text{model}}$$

- **Positive phase** ($\langle \cdot \rangle_{\text{data}}$): clamp visible units to training data, let hidden units settle. How often do units $i$ and $j$ co-activate when driven by real data?
- **Negative phase** ($\langle \cdot \rangle_{\text{model}}$): let the whole network run freely. How often do they co-activate in the model's "fantasy" of the data?

Learning = make the model's fantasies match reality. Adjust weights to close the gap between what the data says and what the model dreams.

This is **contrastive learning** avant la lettre — the same wake/sleep, data/model dichotomy that appears in VAEs, GANs, and contrastive divergence.

### Simulated Annealing

To reach the Boltzmann distribution reliably, the network uses **simulated annealing** (Kirkpatrick et al., 1983 — published the same year!): start at high temperature (random exploration), gradually cool down (converge to low-energy states). Same idea from metallurgy: heat metal to let atoms reorganize, then slowly cool to reach the lowest-energy crystal structure.

---

## Why This Matters for the AI/ML Lineage

1. **First hidden-unit learning algorithm.** Before Boltzmann machines, there was no principled way to train networks with hidden layers. Backpropagation (1986) would provide a faster solution, but Boltzmann machines came first and proved it was possible.

2. **Generative model.** A Boltzmann machine doesn't just classify — it models the data distribution. It can generate new samples. This generative tradition runs through:
   - Restricted Boltzmann Machines (2000s)
   - Deep Belief Nets (Hinton, 2006) — stacks of RBMs that reignited deep learning
   - VAEs (Kingma & Welling, 2014)
   - Diffusion models (2020s)

3. **The physics → ML pipeline made concrete.** Hopfield (1982) showed physics could describe neural networks. Boltzmann machines showed physics could *train* neural networks. The Boltzmann distribution, simulated annealing, partition functions, free energy — this is statistical mechanics applied wholesale to learning.

4. **The fatal flaw: slowness.** Computing $\langle s_i s_j \rangle_{\text{model}}$ requires running the network to thermal equilibrium, which is extremely slow. This made Boltzmann machines impractical for decades. Every subsequent innovation (RBMs, contrastive divergence, deep belief nets) is about fixing this problem.

---

## Cross-Thread Connections 🔗

- **[[Hopfield Neural Networks 1982]]** ← Direct predecessor. Same energy function, but deterministic and no hidden units. Boltzmann machines = Hopfield + stochasticity + hidden layers + learning.
- **[[Rumelhart Hinton Williams Backpropagation 1986]]** → Hinton's career arc: Hopfield (1982) → Boltzmann (1983) → Backprop (1986). He moved from energy-based to gradient-based learning because Boltzmann machines were too slow. But the generative intuition stayed.
- **[[../12-Frontiers/Goodfellow GANs 2014]]** → GANs also learn by contrasting real data vs. model samples. The positive/negative phase distinction in Boltzmann machines is structurally identical to the discriminator/generator game.
- **Hinton (2006) Deep Belief Nets** → Stacks of Restricted Boltzmann Machines. The "deep learning revival" was literally Boltzmann machines made practical 23 years later.
- **[[../../Mathematics/04-Analysis/Cauchy Limits 1821]]** → The Boltzmann distribution itself is exponential (connects to analysis); the partition function Z connects to statistical mechanics.

### Physics cross-thread
- **Ising model (1920s)** — Boltzmann machines ARE stochastic Ising models with external fields. The energy function is identical.
- **Sherrington-Kirkpatrick model (1975)** — spin glass model; Boltzmann machines are S-K models with learned couplings.
- **Kirkpatrick, Gelatt & Vecchi (1983)** — Published simulated annealing the same year. Not coincidental — same intellectual community (physics ↔ computation) at the same moment.

---

## The Name

Named after Ludwig Boltzmann (1844–1906), the Austrian physicist who derived the statistical mechanics distribution $P \propto e^{-E/kT}$ that governs the network's equilibrium. The temperature parameter $T$ directly controls exploration vs. exploitation — high temperature = random search, low temperature = greedy optimization.

---

## Limitations

- **Impossibly slow.** The negative phase requires running the network to equilibrium for *every* weight update. Even small networks took hours.
- **No efficient inference.** You can't quickly compute what the hidden units "think" about a new input.
- **Superseded by backpropagation.** By 1986, backprop offered a much faster way to train hidden layers — but only for feedforward networks, not for the generative models Boltzmann machines could express.

---

## Status

- [x] PDF downloaded (March 27, 2026)
- [x] Note created (March 27, 2026)
- [x] [[Hinton Contrastive Divergence 2002]] — Restricted Boltzmann Machines + the algorithm that finally made them practical
- [x] [[Hinton Deep Belief Nets 2006]] — the culmination: stacks of RBMs → deep learning revival
