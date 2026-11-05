# Hebb — The Organization of Behavior (1949)

*The moment the artificial neuron stopped being only a logic element and became something that could be shaped by experience.*

**← [[_MOC|Foundations]] ← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**Donald O. Hebb, _The Organization of Behavior: A Neuropsychological Theory_ (1949).**

Hebb was not writing an ML paper. He was trying to bridge neurophysiology and psychology: how could perception, memory, and learning arise from physical nervous tissue rather than from an abstract association table?

**Paper:** corpus/raw/library-papers/hebb-organization-behavior-1949.pdf

The famous rule appears in Chapter 4, in Hebb's careful physiological language:

> “When an axon of cell A is near enough to excite a cell B and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells such that A's efficiency, as one of the cells firing B, is increased.”

That later becomes the compressed slogan: **cells that fire together wire together.**

## Key Ideas

1. **Learning is a change in connection strength.**
   [[McCulloch-Pitts 1943]] gave the field a binary threshold neuron: inputs come in, a logical unit fires or does not fire. Hebb supplied the missing dynamic: repeated co-activation should make future activation easier. The network has memory because its connections are changed by use.

2. **Association becomes local and mechanistic.**
   Earlier associationist psychology could say that two ideas become linked. Hebb asked what that link could physically mean. His answer was local synaptic change: if A reliably helps fire B, A becomes a better cause of B. No central teacher has to write the association into the system.

3. **Cell assemblies make memory distributed.**
   A remembered object or percept is not stored in one special neuron. It is a pattern: a **cell assembly** whose members tend to reactivate one another. This is one of the deep ancestors of distributed representations, attractor networks, and embedding-like intuitions.

4. **Phase sequences give thought a temporal structure.**
   Hebb did not stop at static assemblies. Assemblies could activate in ordered chains, producing a **phase sequence**. That gives a rough biological story for thought as structured unfolding rather than a single symbol lookup.

5. **The rule is powerful but incomplete.**
   Pure Hebbian strengthening can run away: if correlated activity only increases weights, what prevents saturation? Later systems need normalization, competition, error correction, inhibition, or objective-driven updates. That limitation matters because it points directly toward perceptrons, LMS, backpropagation, and modern optimization.

## What Came Before → What It Enabled

### What came before

- **McCulloch & Pitts 1943:** neurons as logical threshold units. Enormously important, but static. The network computes a fixed logical function once its wiring is given.
- **Behaviorism and associationism:** learning as stimulus-response association, but without a precise enough neural mechanism.
- **Neurophysiology:** evidence that neurons fire, reverberate, and connect, but no clean computational story for how experience modifies future behavior.

### What broke

The McCulloch-Pitts model could represent computation, but it could not explain **learning**. If the wiring is fixed, intelligence has to be hand-designed into the circuit. That is fatal for any serious theory of adaptive behavior.

So the next question was unavoidable: **how do the weights change?**

Hebb answered: make connection strength depend on co-activity.

### What it enabled

- **Rosenblatt's Perceptron (1958):** learning weights from examples. Rosenblatt's machinery is not merely Hebbian, but the conceptual leap — adjustable connections as the substrate of learning — is Hebb's bridge.
- **Competitive and self-organizing learning:** many unsupervised learning rules descend from the Hebbian instinct: strengthen correlations, then add constraints so the system does not collapse.
- **Hopfield networks (1982):** memories as stable distributed patterns. Hebb's cell assemblies are one ancestor of attractor memory.
- **Representation learning:** modern networks learn internal features by changing weights through experience. The objective functions became more sophisticated, but the deep idea remains Hebbian in spirit: the network's structure is written by activity.

## Why It Matters

Hebb is the missing hinge between **logical neurons** and **learned classifiers**.

Without Hebb, the early neural-net story jumps too quickly from “neurons can compute logic” to “machines can learn.” But those are different claims. Logic tells you what a circuit can do after it is wired. Learning asks how the wiring comes to fit the world.

That is why they could not just keep doing McCulloch-Pitts. Fixed circuits made neural computation possible, but fixed circuits do not adapt. Hebb's rule made adaptation imaginable: not as magic, not as a homunculus, but as repeated activity leaving a physical trace.

It is also why Hebb is not the end of the story. Correlation is not error. Hebbian learning says, “make togetherness stronger.” Perceptrons and ADALINE add a sharper demand: “change weights to reduce mistakes.” Backpropagation eventually generalizes that demand through many layers.

So Hebb's historical role is beautifully specific:

> McCulloch-Pitts gave the neuron a logic. Hebb gave the connection a memory.

That memory was not enough to build frontier AI. But without it, the rest of the chain has nowhere to start.

## Lineage

- **Built on:** [[McCulloch-Pitts 1943]], associationist psychology, and mid-century neurophysiology
- **Leads to:** [[../01-Perceptrons/Rosenblatt Perceptron 1958|Rosenblatt Perceptron 1958]], [[../03-Optimization/Widrow ADALINE 1960|Widrow-Hoff ADALINE 1960]], [[../10-Backpropagation/Hopfield Neural Networks 1982|Hopfield Networks 1982]]
- **Later descendants:** self-organizing maps, contrastive learning intuitions, attractor memory, representation learning
