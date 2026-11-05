# Alan Turing — Computability and the Universal Machine (1936)

#person #paper #milestone #computability #universal-machine #mathematical-logic #ai-lineage

**Person:** Alan Mathison Turing (1912–1954)  
**Work:** “On Computable Numbers, with an Application to the Entscheidungsproblem” (received and read in 1936; published in *Proceedings of the London Mathematical Society*, series 2, volume 42, 1937, pp. 230–265)  
**Primary source:** Turing's original journal article  
**Paper:** `corpus/raw/library-papers/turing-computable-numbers-1936.pdf`  
**Source note:** The local file is a 36-page scan of the original article. It was downloaded from ENS Lyon and verified with `file` as a PDF document, version 1.5.

## Title
Mechanical procedure becomes a mathematical object—and acquires limits.

Boole made reasoning symbolic. Later formal logic made proofs into strings transformed by exact rules. But one phrase remained dangerously informal: **“there is a mechanical method.”** Mathematicians could propose an algorithm, yet they had no general mathematical object capturing what any finite, rule-following calculation could be.

That gap mattered because Hilbert's *Entscheidungsproblem* asked for a general procedure deciding whether an arbitrary formula of first-order logic is provable. Before that question could be answered, “general procedure” itself had to be defined.

Turing's necessary move was to analyze a human calculator stripped to the actions relevant to calculation: inspect one symbol, keep one of finitely many internal conditions, write or erase, move one place, and repeat according to a finite table. The resulting machine turns an informal process into an object that can itself be encoded, supplied as data, simulated by one universal machine, and subjected to diagonal reasoning.

The result has two sides:

```text
positive: one fixed machine can execute any encoded machine
negative: no machine can decide every question about encoded machines
```

That program/data duality and its boundary are part of the conceptual floor beneath digital computers, neural computation, interpreters, learned program execution, and modern AI systems.

## Paper / Source Reference

**Turing, A. M.** “On Computable Numbers, with an Application to the Entscheidungsproblem.” *Proceedings of the London Mathematical Society*, series 2, 42, 230–265. `https://doi.org/10.1112/plms/s2-42.1.230`

Local source:
- `corpus/raw/library-papers/turing-computable-numbers-1936.pdf`

Download source used:
- `https://perso.ens-lyon.fr/aurelien.alvarez/PDF/Turing_on-computable-numbers_1936.pdf`

Primary passages: §1 (computing machines), §2 (automatic machines and complete configurations), §5 (standard descriptions and description numbers), §6 (the universal computing machine), §8 (the diagonal argument and the impossibility of a general circle-free test), §9 (why the model captures human calculation), and §11 (the *Entscheidungsproblem*).

## What Problem Made This Necessary?

### Formal rules still lacked a formal executor

By the early twentieth century, logic had become symbolic enough to state axioms and mechanically check individual proof steps. Hilbert's program asked whether mathematical reasoning could be systematized so completely that a finite procedure would settle every properly posed formal question.

The central decision problem was:

> Is there a general mechanical procedure which, for any formula in the functional calculus, eventually answers whether it is provable?

The words “mechanical procedure” appeared to explain the requirement, but they merely renamed it. A person could recognize familiar algorithms; mathematics still needed a definition broad enough to include every ordinary paper-and-pencil calculation yet precise enough to prove theorems about all such calculations.

Turing reverses the usual direction. Instead of beginning with a particular algorithm and building a machine for it, he asks what irreducible actions a rule-following human computer performs. Finite attention, finite internal memory, discrete symbols, and local steps become the constraints of the model.

That makes a new kind of proof possible:

```text
define all mechanical procedures as finite machine descriptions
  -> encode those descriptions as symbols or numbers
  -> let machines operate on descriptions of machines
  -> ask whether one procedure can decide a property of every procedure
```

Once procedures become data, self-reference is no longer philosophical fog. It becomes a construction.

## Key Ideas

### 1. Computation is local rule-following over external memory

Turing gives the machine:

- a tape divided into squares, each holding one symbol;
- a head that scans exactly one square at a time;
- one state from a finite collection of what he calls *m-configurations*;
- a finite table determining what to write, whether to move left or right, and which state comes next.

The next action depends only on the current state and scanned symbol. The tape supplies potentially unbounded working memory, but every individual step uses finite local information.

The abstraction removes irrelevant details. The calculator's handwriting, speed, paper size, and physical mechanism disappear. What remains is the causal skeleton of an algorithm:

$$
(q_t, s_t) \longmapsto (s_{t+1}, d_t, q_{t+1}),
$$

where $q_t$ is the current state, $s_t$ the scanned symbol, and $d_t$ a move. A computation is the sequence of complete configurations generated by repeatedly applying this transition rule.

### 2. A finite machine can have an unbounded computation

The rule table is finite, but the tape and number of steps need not have a fixed bound. This distinction separates the description of a procedure from the size of the problem instance or duration of its execution.

It is one ancestor of the modern separation between:

```text
model / program parameters: finite stored rule
runtime state:              changing intermediate configuration
working memory:             external or addressable storage
```

The machine does not need infinitely many instructions to process arbitrarily long inputs. Repetition and mutable state provide open-ended behavior from a finite specification.

### 3. Machines can be encoded as data

Turing converts each instruction table into a *standard description* and then into a numerical *description number*. A machine is therefore representable on the same tape alphabet that machines themselves manipulate.

```text
machine description -> symbol string -> input to another machine
```

A rule system can now inspect, copy, transform, and simulate another rule system. The modern stored-program computer was not built in this paper, but its central abstraction is visible: instructions and ordinary data can share a representational medium.

### 4. One universal machine can simulate every encoded machine

In §6, Turing states that a single machine can compute any computable sequence when supplied with the standard description of the corresponding machine. The universal machine reads an encoded instruction table and reproduces that machine's successive configurations.

```text
universal machine + encoded program + input
                  -> simulated program's behavior
```

The same physical substrate can therefore become many different effective machines. Compilers, interpreters, operating systems, virtual machines, and general-purpose computers all inherit this separation between a fixed executor and replaceable symbolic instructions.

Modern foundation models echo the pattern only at a higher and less exact level: a fixed learned executor receives tokens describing a task and conditionally realizes many behaviors. That is not proof that an LLM is literally Turing's universal machine in every practical setting; it is a descendant of the same **instructions-as-input** architecture.

### 5. Not every property of computation is mechanically decidable

Turing calls a machine *circle-free* when it continues producing infinitely many output figures; a *circular* machine eventually stops producing new figures, whether because it halts or because it loops without productive output. He then asks whether a general procedure can inspect any machine description and decide which case holds.

Assume such a tester exists. Combine it with the universal machine, enumerate the descriptions judged circle-free, and construct a machine whose output depends diagonally on the corresponding machines' own outputs. When the constructed machine reaches its own description, the assumed tester forces a contradiction.

This is closely related to what later textbooks teach through the halting problem, but historical precision matters: Turing's paper directly frames the key test in terms of *circular versus circle-free* machines and whether a machine ever prints a designated symbol. “The halting problem” is a later standard formulation, not Turing's own label here.

The boundary is not that particular problems are merely difficult. It is that **no single algorithm can correctly decide the property for every encoded machine**.

### 6. The *Entscheidungsproblem* has no general solution

Turing maps the behavior of a machine into formulas of the functional calculus. If a general procedure could decide whether each such formula were provable, it could decide whether the represented machine ever prints a designated symbol. Section 8 has already shown that no such machine-behavior decider exists. Therefore the assumed proof decider cannot exist either.

The conclusion is carefully scoped:

```text
there is no general decision procedure for all formulas
```

It does **not** mean that no logical fragment is decidable, no theorem can be proved automatically, or computation is futile. Turing explicitly notes that restricted systems may have decision procedures. The impossibility concerns a universal solver for the unrestricted problem.

### 7. The model is justified by an analysis of human calculation

The paper does not merely announce an arbitrary automaton. In §9, Turing argues that a human calculator can distinguish only finitely many immediately recognizable states, can attend to only a bounded local portion of written work at once, and proceeds through elementary operations. Any compound operation can be decomposed into such simple steps.

This is the paper's bridge from formal model to the informal notion of effective calculation. The appendix also outlines equivalence with Alonzo Church's independently developed lambda-definability. Two very different formalisms capture the same class of functions, strengthening what became the Church–Turing thesis: effectively calculable functions are exactly the computable ones.

The thesis is not a theorem equating two already formal definitions; one side—“effectively calculable”—begins as an informal human concept. The mathematical equivalence of multiple precise models is evidence for the identification.

## Historical Context

Turing wrote amid the foundational crisis created by Hilbert's program and Gödel's incompleteness results. Gödel had shown that sufficiently strong consistent formal systems contain statements they cannot prove or refute internally. Church then proved the unsolvability of the *Entscheidungsproblem* using lambda calculus.

Turing independently reached the same boundary through a radically concrete route: model the person carrying out a calculation. Max Newman recognized the importance of the work and helped bring it to publication. After learning of Church's result, Turing added an appendix outlining the equivalence between computability and lambda-definability.

The paper was received on 28 May 1936 and read on 12 November 1936; the journal volume is conventionally dated 1937. Calling it “Turing 1936” follows the work's submission and standard historical shorthand while the bibliographic record remains explicit.

Turing's machine was not a blueprint for a particular electronic computer. Its historical force was more fundamental: it defined the space of algorithms, introduced universal simulation, and proved limits from within that space. Stored-program architecture, developed later by several people under wartime and postwar engineering pressures, made the abstract program/data relation physical.

For this lineage, the crucial downstream use is already explicit in [[McCulloch-Pitts - Neural Networks as Logical Calculus 1943|McCulloch–Pitts 1943]]. Their conclusion relates threshold networks to Turing computation: neural circuits become an alternative physical architecture for realizing mechanically computable behavior.

## Lineage

### What came before

```text
Viète 1591
  -> symbolic transformations become general mathematical procedure
Boole 1854
  -> logical classes and conditions become algebraically manipulable
later formal logic
  -> proofs become finite symbol strings governed by exact rules
Hilbert's Entscheidungsproblem
  -> asks for one mechanical procedure deciding all formulas
Turing 1936
  -> defines mechanical procedure as an encoded machine
```

- **Built on:** [[Boole - Algebra of Logic 1854|Boole 1854]] and later formal logic, which made reasoning symbolic enough for mechanical execution to become a precise question.
- **Solved the next problem:** “algorithm” and “mechanical method” lacked a general mathematical definition.
- **Added a new boundary:** some well-posed questions about computations cannot be decided by any general algorithm.

### What Turing enabled

```text
Turing 1936
  -> finite-state control + symbolic memory + local transitions
  -> machine descriptions become machine-readable data
  -> one universal executor simulates arbitrary encoded procedures
  -> undecidability limits universal prediction and proof search
  -> McCulloch-Pitts 1943 realizes computation in threshold networks
  -> stored-program computers and software
  -> modern AI on universal digital hardware
```

- **Direct next link:** [[McCulloch-Pitts - Neural Networks as Logical Calculus 1943|McCulloch–Pitts 1943]], where simple neural units are assembled into networks capable of logical and Turing-style computation.
- **Then:** [[Hebb - Learning as Weight Change 1949|Hebb 1949]], where fixed computational connections acquire a mechanism for experience-dependent change.
- **Modern branch:** interpreters, program synthesis, neural Turing machines, tool-using models, recurrent state, and questions about which computations an architecture can express efficiently.

## Why It Matters for Mathematics

Turing makes process itself available to mathematics. Before this move, a proof could establish what an individual algorithm did. After it, machines can be quantified over as a class, encoded as integers, and used as inputs to other machines.

That produces three foundational distinctions:

1. **Computable versus definable:** a number or function may have a finite mathematical description without a machine that computes it.
2. **Existence versus decidability:** a question can have a definite yes/no answer for every instance while lacking one algorithm that always finds it.
3. **Program versus universal executor:** many procedures can be represented as data for one fixed simulator.

Diagonalization, inherited from Cantor and transformed through Gödel, now applies to algorithms acting on encoded algorithms. The technique exposes limits without appealing to machine size, speed, or engineering defects.

## Why It Matters for AI

### AI requires a substrate that can realize many procedures

Modern AI runs on general-purpose computers. Training code, tensor kernels, model weights, token sequences, and tool instructions are all encoded data interpreted by layered executors. Universal computation explains why one hardware family can host image models, language models, optimizers, simulators, and agents without being rebuilt for each task.

### Neural computation needed a target notion of computation

Boole explains how logical expressions can be transformed. Turing explains what it means for any rule-following process to be mechanically computable. McCulloch and Pitts can then ask whether networks of threshold units realize that class of behavior.

```text
Boole:             reasoning has symbolic algebra
Turing:            symbolic procedure has a general machine model
McCulloch-Pitts:   neural networks can realize logical computation
Hebb:              experience can alter connection strength
Rosenblatt onward: data can train computational networks
```

### Universality is not intelligence

A universal machine can simulate any algorithm given the right description, but universality says almost nothing about whether the description can be found, whether the computation is efficient, or whether the result generalizes from data.

Those missing pieces become later AI problems:

- **learning:** acquire useful behavior instead of receiving a complete program;
- **optimization:** search enormous parameter spaces;
- **sample efficiency:** infer from limited evidence;
- **computational complexity:** distinguish possible computation from affordable computation;
- **alignment and verification:** reason about behavior when complete prediction is impossible.

Turing therefore gives AI neither learning nor intelligence. He gives it the space of mechanically realizable procedures and proves that this space cannot contain a perfect general predictor of itself.

### Prompted models revive the universal-interpreter pattern

A foundation model is one fixed parameterized system whose behavior changes with an instruction encoded in its context. The analogy to a universal machine is useful at the architectural level:

```text
fixed executor + task description -> task-conditioned behavior
```

But the differences are equally important. Turing simulation is exact and defined by formal transition rules; language-model behavior is learned, probabilistic, resource-bounded, and not guaranteed to interpret every prompt correctly. The lineage is conceptual, not an identity claim.

## References

- Turing, A. M. “On Computable Numbers, with an Application to the Entscheidungsproblem.” Local primary source: `corpus/raw/library-papers/turing-computable-numbers-1936.pdf`.
- Turing paper DOI: `https://doi.org/10.1112/plms/s2-42.1.230`.
- Downloaded scan: `https://perso.ens-lyon.fr/aurelien.alvarez/PDF/Turing_on-computable-numbers_1936.pdf`.
- [[Boole - Algebra of Logic 1854]]
- [[McCulloch-Pitts - Neural Networks as Logical Calculus 1943]]
- [[Hebb - Learning as Weight Change 1949]]

## Status

- [x] Primary source present at `corpus/raw/library-papers/turing-computable-numbers-1936.pdf`
- [x] Downloaded with `curl -L -o`
- [x] Verified with `file` as a PDF document, version 1.5
- [x] Complete source checked: 36 pages
- [x] Turing's circle-free result distinguished from the later textbook halting-problem formulation
- [x] Wired into the chain between symbolic logic and neural computation

---

*Boole made logic calculable. Turing made calculation itself an object—and proved that no such object can fully predict the whole class it belongs to.*
