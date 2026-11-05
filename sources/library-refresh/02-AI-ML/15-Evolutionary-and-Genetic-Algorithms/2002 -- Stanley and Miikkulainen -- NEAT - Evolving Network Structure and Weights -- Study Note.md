# NEAT — Evolving Network Structure and Weights

**Source:** Kenneth O. Stanley and Risto Miikkulainen, *Evolving Neural Networks through Augmenting Topologies*, Evolutionary Computation 10(2), 99–127 (2002).
**Full text (not imported):** Complete journal article. [Author-university source](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf).
**Reading status:** Method, examples, and experimental design inspected; this records a focused reading, not a reproduction of the experiments.

## The problem left by fixed architectures

Backpropagation supplies derivatives for a chosen computational graph. It does not by itself decide which nodes and connections should exist. Evolution can search network parameters using a performance score, but allowing topology to change introduces another problem: two genomes need not encode corresponding connections at corresponding positions.

NEAT combines three mechanisms: historical identifiers align genes during crossover, speciation protects new structures while their weights adapt, and structural mutation grows networks from small initial graphs. It is a particular neuroevolution algorithm, rather than a synonym for all genetic algorithms.

## An encoding example

Suppose parent A has connection identifiers {1, 2, 4}, while parent B has {1, 3, 4, 5}. Identifiers 1 and 4 match despite their different array positions. Genes 2 and 3 are disjoint; gene 5 is excess relative to A's identifier range. Crossover aligns shared ancestry instead of cutting arbitrary equal-length arrays. The paper specifies separate inheritance rules for matching and unmatched genes, including the role of parental fitness.

An add-node mutation disables an existing connection and inserts two connections through a new node. The incoming weight is initialized to 1 and the outgoing weight to the old weight. With an identity activation this would preserve that local map exactly. NEAT introduces a nonlinear node, so exact functional preservation should not be inferred from the initialization.

## Protecting a structural change

The paper uses a compatibility score

\[
\delta=\frac{c_1E}{N}+\frac{c_2D}{N}+c_3\overline W,
\]

where E and D count excess and disjoint genes, and the final term averages weight differences among matching connections. N ordinarily normalizes by the larger genome; the paper allows N=1 for small genomes. Thresholding this score groups similar genomes for competition. It is a designed compatibility rule, not a theorem that biological species or functional similarity have been recovered.

Why protect a weaker newcomer? A new structure may initially score worse because its parameters are unadapted. Immediate competition against mature networks can eliminate that structure before it becomes useful. Speciation and fitness sharing modify selection pressure to make this exploration possible.

## Evidence and limits

The paper evaluates XOR and pole-balancing tasks and ablates components of the algorithm. These experiments support claims about that design and those benchmarks; they do not establish universal superiority over gradient training or later reinforcement-learning methods. Evaluation count, simulation cost, random seeds, and generalization beyond the training task remain material.

This branch connects population search to representation design: the search variable can include both a continuous weight vector and a discrete graph. A model learned by evolution can still contain ordinary neural computations. Choosing an evolutionary search procedure does not remove the need to define a reliable objective or respect constraints.

**Verified locators:** §§2.2–2.4 identify encoding and innovation problems; §§3.1–3.4 describe NEAT; Figure 3 explains structural mutations; §3.3 equation (1) defines compatibility; §§4–5 report evaluations and ablations. The copy used has 30 pages; journal pagination is 99–127.
