# Rumelhart, Hinton & Williams — Learning Representations by Back-Propagating Errors (1986)

*The paper that made hidden-layer credit assignment feel usable: errors could be sent backward, so internal representations could be learned instead of hand-designed.*

**← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**David E. Rumelhart, Geoffrey E. Hinton & Ronald J. Williams, "Learning representations by back-propagating errors" (1986).**

- Paper: `corpus/raw/library-papers/rumelhart1986.pdf`
- Published in *Nature*, Vol. 323, October 9, 1986, pp. 533-536.

## Key Ideas

1. **Hidden units need assigned responsibility.**
   Output units have an obvious error signal: compare the model output with the desired output. Hidden units do not. The paper's central move is to compute how each hidden unit contributed to downstream error, then update its incoming weights accordingly.

2. **Backpropagation is the chain rule turned into a learning algorithm.**
   The method performs a forward pass to compute activations, then a backward pass to compute derivatives of the error with respect to every weight. Weight changes are proportional to the negative gradient of the error.

3. **Differentiable units replace hard threshold logic.**
   McCulloch-Pitts and perceptron-style threshold units were not suitable for gradient-based hidden-layer learning because hard steps are non-differentiable. Smooth nonlinearities made it possible to move credit assignment through layers.

4. **The point is representation learning, not just curve fitting.**
   The paper's family-tree experiment showed hidden units organizing internal structure such as generation and nationality without being explicitly told those categories. The important claim was that networks could discover useful intermediate features.

5. **The idea was older than the famous paper, but this publication made it legible.**
   [[Werbos Beyond Regression 1974|Werbos 1974]] had already developed ordered derivatives for composed systems, and related rediscoveries happened before 1986. Rumelhart, Hinton, and Williams made the method short, vivid, empirical, and impossible for the neural-network community to ignore.

## What Came Before → What It Enabled

### What came before

- [[../00-Foundations/McCulloch-Pitts 1943|McCulloch & Pitts 1943]] made neurons into formal computational units, but the wiring was fixed.
- [[../01-Perceptrons/Rosenblatt Perceptron 1958|Rosenblatt 1958]] showed that a single layer could learn from examples, but only within linear-separability limits.
- [[../03-Optimization/Widrow ADALINE 1960|Widrow-Hoff 1960]] tied learning to an explicit error signal and gradient-like updates, but still at the single-layer level.
- [[../01-Perceptrons/Minsky-Papert Perceptrons 1969|Minsky & Papert 1969]] made the limitation impossible to ignore: single-layer perceptrons could not solve key problems such as XOR.
- [[Werbos Beyond Regression 1974|Werbos 1974]] supplied a deeper mathematical route through ordered derivatives, but it did not yet become the shared neural-network recipe.

### What broke

Single-layer learning broke on problems whose solution required intermediate features. You could say "use multiple layers," but that only moved the difficulty: the output layer had targets, while hidden layers had no direct teaching signal.

Hand-designing features did not solve the deeper problem either. If every useful representation had to be engineered in advance, the system was not really learning the structure of the task. Neural networks needed a way to discover internal representations from error alone.

Backpropagation repaired that break by making hidden credit assignment mechanical. The error at the output could be decomposed backward through the computation graph, giving every weight a local update derived from the global error.

### What it enabled

- [[../11-CNNs/LeCun Backpropagation Zip Codes 1989|LeCun 1989]]: convolutional networks trained by backpropagation become practical for handwritten zip-code recognition.
- [[Hochreiter Schmidhuber LSTM 1997|LSTM 1997]]: recurrent networks could be trained through time, even though vanilla backpropagation exposed the vanishing-gradient problem.
- [[Hinton Deep Belief Nets 2006|deep belief nets]] and the 2006 revival: backpropagation returned as the fine-tuning step after unsupervised pretraining.
- [[../11-CNNs/Krizhevsky-Sutskever-Hinton AlexNet 2012|AlexNet 2012]] and the modern deep-learning wave: with enough data, GPUs, ReLUs, normalization, and regularization, backpropagation became the default training engine.
- [[../12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] and generative AI: the architectures changed radically, but the basic training fact remained the same: differentiable networks are optimized by gradients computed backward.

## Why It Matters

Rumelhart, Hinton, and Williams matters because it closes the practical gap between **networks can represent complex functions** and **networks can learn those functions**.

The field could not just keep using perceptron-style learning. Perceptrons only adjusted weights from inputs directly to outputs; they had no way to learn latent features. The moment a task required a hidden representation, the old recipe ran out of machinery.

Backpropagation supplied that machinery. It made depth trainable by turning a network into a differentiable program whose parameters could all receive error signals. That is the through-line from early multilayer nets to CNNs, RNNs, LSTMs, Transformers, diffusion models, and frontier language models.

It also explains why later papers were still necessary. Backpropagation solved credit assignment in principle, but not all practical training failures. Deep networks still suffered from vanishing gradients, poor conditioning, fragile initialization, overfitting, slow hardware, and brittle optimization. The next decades of deep learning can be read as repairs to those remaining breaks.

## Lineage

- **Built on:** [[../00-Foundations/McCulloch-Pitts 1943|McCulloch-Pitts threshold neurons]], [[../01-Perceptrons/Rosenblatt Perceptron 1958|perceptron learning]], [[../03-Optimization/Widrow ADALINE 1960|LMS / error-driven updates]], [[Werbos Beyond Regression 1974|Werbos ordered derivatives]]
- **Repairs:** the hidden-layer credit-assignment failure exposed by single-layer perceptrons
- **Leads to:** [[../11-CNNs/LeCun Backpropagation Zip Codes 1989|LeCun 1989 CNNs]], [[Hochreiter Schmidhuber LSTM 1997|LSTMs]], [[Hinton Deep Belief Nets 2006|deep learning revival]], [[../11-CNNs/Krizhevsky-Sutskever-Hinton AlexNet 2012|AlexNet]], [[../12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]]

---

*Backpropagation is the moment hidden layers stop being an aspiration and become trainable machinery.*
