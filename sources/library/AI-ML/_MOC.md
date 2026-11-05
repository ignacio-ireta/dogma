# 🤖 AI/ML Lineage

*From the first artificial neuron to frontier AI systems.*

**← [[../_MOC|Library]] ← [[../../_INDEX|Knowledge Map]]**

---

## The Thread

### 00 — Foundations
- [[00-Foundations/McCulloch-Pitts 1943]] — The first mathematical neuron. Binary threshold units made neural activity a formal computation, while their fixed wiring exposed the need for Hebbian and perceptron learning.
- [[00-Foundations/Hebb Learning Rule 1949]] — Connection strengths change through repeated co-activation. The missing learning-rule bridge from logical neurons to perceptrons.

### 01 — Perceptrons
- [[01-Perceptrons/Rosenblatt Perceptron 1958]] — The first learned threshold classifier. Fixed neural logic and pure Hebbian association were not enough; Rosenblatt made supervised weight adaptation the bridge to convergence proofs, ADALINE, perceptron limits, and hidden-layer credit assignment.
- [[01-Perceptrons/Novikoff Perceptron Convergence 1962]] — The finite-mistake theorem. If a separator with margin exists, perceptron learning will not wander forever — and the condition exposes exactly why single-layer models later break.
- [[01-Perceptrons/Minsky-Papert Perceptrons 1969]] — The limitation theorem for shallow perceptrons. Local evidence plus one weighted vote could not capture global structure such as connectedness, making learned internal representations and backpropagation logically necessary.

### 02 — Linear Algebra
- [[02-Linear-Algebra/Eckart-Young Best Low-Rank Approximation 1936]] — Truncated SVD is the optimal low-rank approximation. The mathematical core of PCA, latent factors, and matrix compression.
  - Connected to: [[12-Frontiers/Radford CLIP 2021|CLIP]] (embedding geometry)
- [[02-Linear-Algebra/Golub Reinsch SVD and Least Squares 1970]] — The computational bridge that made SVD dependable in practice. Bidiagonalization, numerical stability, least squares, and the reason low-rank methods became everyday tools instead of just theorems.
  - Connected to: [[12-Frontiers/Lewis Retrieval-Augmented Generation 2020|RAG / embeddings]] (spectral geometry under vector spaces)

### 03 — Optimization
- [[03-Optimization/Optimizer Failure Modes Interview Bridge]] — Interview diagnostic: separate objective mismatch, stochastic noise, ill-conditioning, and generalization before reaching for a different optimizer.
- [[03-Optimization/Cauchy Steepest Descent 1847]] — The root of gradient-based learning. Replace brittle symbolic elimination with a scalar residual objective, then move opposite its derivatives to make the error smaller.
- [[03-Optimization/Robbins Monro Stochastic Approximation 1951]] — The mathematical hinge from deterministic optimization to noisy sample-based learning. Sequential updates can converge even when the function is only visible through random observations.
- [[03-Optimization/Widrow ADALINE 1960]] — ADALINE and the LMS rule. The bridge from perceptron-style mistake correction to explicit squared-error minimization: one example, one error signal, one gradient-like weight update.
- [[03-Optimization/Polyak Heavy-Ball Momentum 1964]] — The missing memory step between gradient descent and Adam. A two-step update carries velocity through ill-conditioned valleys, improving the dependence on condition number from order κ to order √κ.
- [[03-Optimization/Duchi Hazan Singer AdaGrad 2011]] — Per-coordinate adaptive learning rates. The optimizer remembers which features have already spoken often and which rare but predictive coordinates still need larger steps.
- [[03-Optimization/Hinton RMSProp 2012]] — Recent squared-gradient memory. AdaGrad's lifetime accumulator becomes an exponential moving average, giving mini-batch neural nets adaptive scaling without permanently shrinking learning rates.
- [[03-Optimization/Kingma Ba Adam 2015]] — Adaptive moment estimation. Momentum's directional memory plus RMSProp's recent squared-gradient scale, corrected for initialization bias, becomes the deep-learning default optimizer.
  - Connected to: [[03-Optimization/Hinton RMSProp 2012]] (second-moment ancestor), [[03-Optimization/Loshchilov AdamW 2019]] (regularization fix)
- [[03-Optimization/Loshchilov Hutter SGDR 2017]] — Cosine annealing with warm restarts. The learning-rate schedule as an optimization strategy: settle, kick, explore again.
- [[03-Optimization/Loshchilov AdamW 2019]] — Fixed Adam's weight decay, now standard for Transformers

### 04 — Probability & Information
- [[04-Probability/Shannon Information Theory 1948]] — Messages become measurable. Entropy, bits, source modeling, compression, and the deep reason next-token prediction is a serious learning objective rather than a party trick.
  - Connected to: [[12-Frontiers/Kaplan Scaling Laws 2020|Scaling laws]] (cross-entropy loss), [[12-Frontiers/Radford GPT-2 2019|GPT-2]] (language modeling), [[12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] (sequence modeling at scale)

### 05 — Classification
- [[05-Classification/Fisher Linear Discriminant 1936]] — The foundational labeled-classification note. Maximize between-class separation while minimizing within-class spread. The supervised cousin of PCA.
  - Connected to: [[../Mathematics/07-Multivariate-Statistics/Pearson - Principal Components 1901|Pearson 1901]] (same geometry, labels added)
- [[05-Classification/Friedman Gradient Boosting 2001]] — The missing conceptual bridge from generic boosting to modern tree ensembles. Boosting reinterpreted as gradient descent in function space.
  - Connected to: [[05-Classification/Chen Guestrin XGBoost 2016|XGBoost]] (the industrial descendant)
- [[05-Classification/Chen Guestrin XGBoost 2016]] — The canonical tabular-ML workhorse. Gradient-boosted trees made fast, regularized, sparse-aware, and production-grade.
- [[05-Classification/XGBoost Interview Bridge]] — Short synthesis for turning Friedman + XGBoost into a practical interview story: tabular risk prediction, loss-aware thresholding, calibration, and production-minded model choice.
  - Connected to: [[05-Classification/Friedman Gradient Boosting 2001|Friedman 2001]] (loss and functional gradients), [[05-Classification/Chen Guestrin XGBoost 2016|XGBoost 2016]] (regularized sparse-aware implementation)

### 07 — Kernels
- [[07-Kernels/Boser Guyon Vapnik Optimal Margin Classifiers 1992]] — The original support vector machine paper. Maximum-margin classification, support vectors, and the kernel trick all crystallize here.
  - Connected to: [[05-Classification/Fisher Linear Discriminant 1936|Fisher 1936]] (supervised geometry)
- [[07-Kernels/Cortes Vapnik Support-Vector Networks 1995]] — Soft-margin SVMs. Maximum-margin classification becomes robust to overlap, noisy labels, and non-separable training data.
  - Connected to: [[07-Kernels/Boser Guyon Vapnik Optimal Margin Classifiers 1992|hard-margin SVMs]] (the separable ancestor), [[05-Classification/Fisher Linear Discriminant 1936|Fisher 1936]] (classification geometry)

### 10 — Backpropagation
- [[10-Backpropagation/LSTM Memory Management Interview Bridge]] — Interview bridge from LSTM's forget gate to Transformers and long-context systems: recurrent state had to become managed memory before attention made the past directly addressable.
- [[10-Backpropagation/Hopfield Neural Networks 1982]] — Energy-based models, physics meets neural networks, content-addressable memory
- [[10-Backpropagation/Hinton Sejnowski Boltzmann Machines 1983]] — Stochastic Hopfield nets + hidden units = first principled hidden-layer learning. The generative tradition begins.
- [[10-Backpropagation/Hinton Contrastive Divergence 2002]] — The practical RBM training trick. Replace slow equilibrium sampling with a cheap data-vs-reconstruction contrast.
- [[10-Backpropagation/Werbos Beyond Regression 1974]] — Reverse-mode differentiation before backprop became famous. Ordered derivatives turn hidden-variable credit assignment into a backward traversal of the computation.
- [[10-Backpropagation/Rumelhart Hinton Williams Backpropagation 1986]] — Hidden-layer credit assignment becomes trainable. Backpropagation turns the chain rule into error signals for every weight, allowing internal representations to be learned instead of hand-designed.
- [[10-Backpropagation/Bengio Simard Frasconi Long-Term Dependencies 1994]] — The missing diagnosis between backpropagation and LSTM. Robust attractor-based memory contracts away noise, but that same contraction makes derivatives of distant events vanish; representational capacity is not enough when credit cannot cross time.
- [[10-Backpropagation/Hinton Deep Belief Nets 2006]] — Greedy layer-wise pretraining made depth trainable again. RBM stacks revived multilayer representation learning before the GPU/ImageNet explosion.
- [[10-Backpropagation/Glorot Bengio Initialization 2010]] — Xavier initialization. Deep random networks could compute gradients yet still saturate or distort signal scale layer by layer; balancing fan-in and fan-out keeps forward activations and backward gradients alive enough for end-to-end training.
- [[10-Backpropagation/Nair Hinton ReLU 2010]] — Rectified units keep positive-side gradients and feature magnitude alive. The missing bridge from DBN-era layer-wise pretraining to AlexNet's fast end-to-end supervised training.
- [[10-Backpropagation/He Rectifier Initialization 2015]] — Kaiming initialization makes weight scale respect ReLU's half-rectified statistics, allowing very deep rectified networks to train from scratch. Its 30-layer plain-network degradation exposes the next problem that ResNet must solve.
  - Connected to: [[10-Backpropagation/Glorot Bengio Initialization 2010|Xavier initialization]] (linear-activation ancestor), [[11-CNNs/He ResNets 2015|ResNet]] (depth-degradation repair)
- [[10-Backpropagation/Hochreiter Schmidhuber LSTM 1997]] — Constant Error Carousel solves vanishing gradients in RNNs. Gated memory cells. Dominant sequence model for 20 years.
- [[10-Backpropagation/Sutskever Vinyals Le Sequence to Sequence 2014]] — Two deep LSTMs turn variable-length input into variable-length output end to end. The fixed-vector encoder–decoder makes neural generation practical—and exposes the compression bottleneck that attention must repair.
- [[10-Backpropagation/Gers Learning to Forget LSTM 2000]] — The forget gate. LSTM could preserve memory across long gaps, but continual streams still broke it when no external reset marked sequence boundaries. Learned forgetting turned memory into managed state.
- [[10-Backpropagation/Hinton Dropout 2012]] — Regularization breakthrough
- [[10-Backpropagation/LeCun Bengio Hinton Deep Learning 2015]] — Deep learning as a coherent paradigm: hierarchical representation learning trained end-to-end with backpropagation, not isolated neural-net tricks.
- [[10-Backpropagation/Ioffe Szegedy Batch Normalization 2015]] — Trainability breakthrough. Normalize activations inside the computation graph so deep networks can use larger learning rates, depend less on fragile initialization, and train much faster.
  - Connected to: [[11-CNNs/He ResNets 2015]] (normalization + residual connections make extreme depth practical), [[10-Backpropagation/Ba Kiros Hinton Layer Normalization 2016]] (the sequence-model repair)
- [[10-Backpropagation/Ba Kiros Hinton Layer Normalization 2016]] — Normalize each example across its hidden features, not each feature across a mini-batch. This removes batch-size, time-step, and train–test-statistics dependencies, making normalization fit RNNs and later the Transformer’s Add & Norm blocks.
  - Connected to: [[10-Backpropagation/Ioffe Szegedy Batch Normalization 2015|BatchNorm]] (the normalization ancestor), [[12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] (the architectural descendant)
- [[10-Backpropagation/Cho Encoder-Decoder GRU 2014]] — Sequence-to-sequence, GRU cells (simplified LSTM)

### 11 — CNNs
- [[11-CNNs/LeCun Backpropagation Zip Codes 1989]] — First practical CNN. Local receptive fields + weight sharing + backpropagation = handwritten digit recognition. Architecture as inductive bias.
- [[11-CNNs/LeCun Bottou Bengio Haffner LeNet-5 1998]] — The mature CNN before AlexNet. LeNet-5 turned convolution, subsampling, and learned features into a deployed document-recognition system, with graph transformer networks connecting recognition to segmentation and check-level decisions.
- [[11-CNNs/Krizhevsky-Sutskever-Hinton AlexNet 2012]] — The ImageNet breakthrough. CNNs scaled with GPUs, ReLU, dropout, and data augmentation. The paper that made deep learning undeniable.
- [[11-CNNs/He ResNets 2015]] — Residual learning repairs the depth-degradation problem: preserve an identity path and make each block learn only a correction. This turns depth from an optimization liability into useful capacity and supplies the residual stream later inherited by Transformers.
  - Connected to: [[10-Backpropagation/Ioffe Szegedy Batch Normalization 2015|BatchNorm]] (healthy signals reveal the deeper optimization failure), [[12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] (the `Add` in Add & Norm)

### 12 — Frontiers
- [[12-Frontiers/Bahdanau Attention 2015]] — Attention repairs the fixed-vector seq2seq bottleneck by letting each decoder step retrieve a learned weighted combination of encoder states.
- [[12-Frontiers/Sennrich Haddow Birch Subword BPE 2016]] — Learned subword units repair neural models' closed word vocabularies: a finite symbol inventory can encode and generate rare or unseen forms without `UNK`, dictionary backoff, or character-length sequences.
  - Connected to: [[12-Frontiers/Bengio Neural Probabilistic Language Model 2003|word-level neural language models]] (the closed-vocabulary ancestor), [[12-Frontiers/Bahdanau Attention 2015|attention-based NMT]] (the system whose rare-word failure it repaired), [[12-Frontiers/Radford GPT-1 2018|GPT-1]] (40,000-merge BPE), [[12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] (subword sequence modeling)
- [[12-Frontiers/Tokenization Economics Interview Bridge]] — Interview synthesis for treating a context window as a tokenizer-specific coverage, cost, and latency budget; measure the real multilingual corpus before choosing chunk size, retrieval depth, or model.
  - Connected to: [[12-Frontiers/Sennrich Haddow Birch Subword BPE 2016|subword BPE]] (coverage/efficiency mechanism), [[12-Frontiers/Radford GPT-2 2019|byte-level BPE]] (modern open-text interface)
- [[12-Frontiers/Luong Pham Manning Attention 2015]] — Attention becomes a reusable design language: global vs. local retrieval, multiple scoring functions, and input feeding bridge Bahdanau’s alignment mechanism to the Transformer’s parallel dot-product attention.
- [[12-Frontiers/Kingma Welling Auto-Encoding Variational Bayes 2013]] — The trainable-latent-variable hinge. Amortized inference, the ELBO, and the reparameterization trick let nonlinear probabilistic encoders and decoders learn jointly with minibatch backpropagation.
  - Connected to: [[02-Linear-Algebra/Eckart-Young Best Low-Rank Approximation 1936|low-rank latent representations]] (the linear ancestor), [[10-Backpropagation/Hinton Sejnowski Boltzmann Machines 1983|Boltzmann machines]] (the iterative-inference predecessor), [[12-Frontiers/van den Oord VQ-VAE 2017|VQ-VAE]] (the discrete-latent repair), [[12-Frontiers/Goodfellow GANs 2014|GANs]] (the implicit-generation response), [[12-Frontiers/Ho Denoising Diffusion Probabilistic Models 2020|DDPM]] (variational generative modeling through a long latent chain)
- [[12-Frontiers/van den Oord VQ-VAE 2017]] — A learned codebook and straight-through estimator make discrete latent representations trainable, prevent a powerful decoder from quietly discarding the bottleneck, and move expensive generation from raw pixels or waveforms into shorter symbol sequences.
  - Connected to: [[12-Frontiers/Kingma Welling Auto-Encoding Variational Bayes 2013|VAE]] (continuous-latent ancestor), [[12-Frontiers/Esser Rombach Ommer VQGAN 2021|VQGAN]] (visual perceptual-tokenizer branch), [[12-Frontiers/Zeghidour SoundStream Neural Audio Codec 2021|SoundStream]] (audio residual-tokenizer branch), [[12-Frontiers/Rombach Latent Diffusion Models 2022|latent diffusion]] (compressed-space descendant)
- [[12-Frontiers/Esser Rombach Ommer VQGAN 2021]] — Perceptual and adversarial reconstruction turn VQ-VAE’s codes into context-rich visual parts, shortening image sequences enough for Transformers to model high-resolution global composition. The stronger tokenizer fixes the pixel-literal bottleneck but leaves autoregressive sampling serial, setting up latent diffusion.
  - Connected to: [[12-Frontiers/van den Oord VQ-VAE 2017|VQ-VAE]] (discrete-code ancestor), [[12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] (global-composition prior), [[12-Frontiers/Rombach Latent Diffusion Models 2022|latent diffusion]] (same compression insight, parallel spatial prior)
- [[12-Frontiers/Zeghidour SoundStream Neural Audio Codec 2021]] — Residual vector quantization turns an impossible high-rate audio codebook into small successive refinements; quantizer dropout makes one causal codec span 3–18 kbps. Learned audio symbols become practical bitstreams—and the acoustic vocabulary for later audio language models.
  - Connected to: [[12-Frontiers/van den Oord VQ-VAE 2017|VQ-VAE]] (learned discrete-latent ancestor), [[12-Frontiers/Borsos et al. AudioLM 2022|AudioLM]] (semantic/acoustic hierarchy), [[12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] (later codec-token prior), [[12-Frontiers/Esser Rombach Ommer VQGAN 2021|VQGAN]] (parallel perceptual-tokenizer branch)
- [[12-Frontiers/Borsos et al. AudioLM 2022]] — One audio alphabet cannot preserve both long-range meaning and waveform fidelity economically. AudioLM predicts low-rate semantic tokens first, then coarse and fine SoundStream tokens, turning learned compression into a textless generative hierarchy for coherent speech and music.
  - Connected to: [[12-Frontiers/Zeghidour SoundStream Neural Audio Codec 2021|SoundStream]] (reconstructable acoustic vocabulary), [[12-Frontiers/Agostinelli et al. MusicLM 2023|MusicLM]] (text-conditioned music descendant), [[12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] (autoregressive sequence model)
- [[12-Frontiers/Agostinelli et al. MusicLM 2023]] — AudioLM gains a language interface without requiring captions for 280,000 hours of training music. Frozen MuLan embeddings let audio condition the generator during training and text substitute for audio at inference; text-plus-melody control and minute-scale continuation follow, while efficiency, song-level planning, and provenance become the next breaks.
  - Connected to: [[12-Frontiers/Borsos et al. AudioLM 2022|AudioLM]] (semantic/acoustic generative hierarchy), [[12-Frontiers/Radford CLIP 2021|CLIP]] (shared cross-modal geometry), [[12-Frontiers/Zeghidour SoundStream Neural Audio Codec 2021|SoundStream]] (codec-token output), [[12-Frontiers/Copet et al. MusicGen 2023|MusicGen]] (single-stage multi-codebook repair)
- [[12-Frontiers/Copet et al. MusicGen 2023]] — A delay interleaving pattern collapses MusicLM-style cascades into one Transformer over four EnCodec streams. Thirty seconds becomes 1,500 autoregressive steps rather than 6,000 under exact flattening, while text, unsupervised melody control, and stereo remain in one model.
  - Connected to: [[12-Frontiers/Agostinelli et al. MusicLM 2023|MusicLM]] (text-controlled hierarchical predecessor), [[12-Frontiers/Zeghidour SoundStream Neural Audio Codec 2021|SoundStream]] (residual-codebook ancestor), [[12-Frontiers/Dao et al. FlashAttention 2022|FlashAttention]] (long-sequence execution), [[12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] (single-stage sequence model)
- [[12-Frontiers/Goodfellow GANs 2014]] — Generative adversarial networks
- [[12-Frontiers/Radford DCGAN 2016]] — Making GANs work with convolutions
- [[12-Frontiers/Ho Denoising Diffusion Probabilistic Models 2020]] — The hinge from adversarial image generation to modern diffusion. A fixed Gaussian corruption process creates known training targets; one time-conditioned U-Net learns to predict noise and reverse the process step by step, trading GAN instability for slow iterative sampling.
  - Connected to: [[12-Frontiers/Goodfellow GANs 2014|GANs]] (the dominant but fragile image-generation predecessor), [[12-Frontiers/Radford DCGAN 2016|DCGAN]] (the convolutional generative baseline), [[10-Backpropagation/Rumelhart Hinton Williams Backpropagation 1986|backpropagation]] (ordinary regression training for the denoiser)
- [[12-Frontiers/Song Meng Ermon DDIM 2021]] — DDPM’s loss does not require DDPM’s thousand-step Markov sampler. Preserve the trained denoiser, choose a non-Markovian deterministic trajectory, and skip most noise levels for a reported 10–50× sampling speedup.
  - Connected to: [[12-Frontiers/Ho Denoising Diffusion Probabilistic Models 2020|DDPM]] (same training objective and network), [[12-Frontiers/Rombach Latent Diffusion Models 2022|latent diffusion]] (complementary reduction in cost per denoising step)
- [[12-Frontiers/Rombach Latent Diffusion Models 2022]] — Perceptual compression repairs pixel-space diffusion's high-resolution compute bill. A frozen autoencoder removes mostly imperceptible detail once; DDPM-style denoising models the smaller spatial code, while cross-attention supplies a general interface for text, layouts, and other conditions.
  - Connected to: [[12-Frontiers/Kingma Welling Auto-Encoding Variational Bayes 2013|VAEs]] (trainable latent spaces), [[12-Frontiers/van den Oord VQ-VAE 2017|VQ-VAE]] (quantized compressed representations), [[12-Frontiers/Ho Denoising Diffusion Probabilistic Models 2020|DDPM]] (stable denoising objective), [[12-Frontiers/Song Meng Ermon DDIM 2021|DDIM]] (faster inference trajectory), [[12-Frontiers/Vaswani Attention Is All You Need 2017|cross-attention]] (conditioning interface)
  - **Interview bridge — move compute to the representation that preserves the decision signal.** LDM's senior-level lesson is not merely “compress images”: separate a reusable representation stage from the expensive task model, then choose the bottleneck by measuring quality, latency, and cost end to end. Ask what information the compressor may discard, validate that downstream decisions survive the bottleneck, and name the residual constraint—the decoder cannot recover lost evidence, while iterative sampling remains serial. This is a concrete counterexample to treating a larger model or more infrastructure as the first architecture move. Source note: `AI-ML/12-Frontiers/Rombach Latent Diffusion Models 2022.md`; source PDF: `corpus/raw/library-papers/2112.10752.pdf`.
- [[12-Frontiers/Socher Zero-Shot Learning 2013]] — Learning without examples
- [[12-Frontiers/Vaswani Attention Is All You Need 2017]] — Transformers. Everything changed.
- [[12-Frontiers/Dai et al. Transformer-XL 2019]] — Segment-level recurrence repairs the Transformer's fixed-window amnesia: cache prior hidden states, attend into them without recomputation, and use relative positions so reused memory keeps its temporal meaning.
  - Connected to: [[12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] (the fixed-segment ancestor), [[12-Frontiers/Child Sparse Transformer 2019|Sparse Transformer]] (the parallel cost-reduction branch), [[13-Scaling-and-Alignment/DeepSeek-V4 — DeepSeek AI 2026|DeepSeek-V4]] (compressed million-token memory)
- [[12-Frontiers/Jacobs et al. — Adaptive Mixtures of Local Experts 1991]] — The first real MoE note. Learned gating + competing specialists. The foundational conditional-computation instinct.
  - Connected to: [[12-Frontiers/Jordan Jacobs Hierarchical Mixtures of Experts 1994]] (the hierarchical refinement), [[12-Frontiers/Shazeer Sparsely-Gated Mixture-of-Experts 2017]] (the sparse scaling breakthrough)
- [[12-Frontiers/Jordan Jacobs Hierarchical Mixtures of Experts 1994]] — Hierarchical MoE. Soft tree-structured gates and EM turn flat expert competition into probabilistic conditional computation.
  - Connected to: [[12-Frontiers/Jacobs et al. — Adaptive Mixtures of Local Experts 1991]] (the flat learned-specialist ancestor), [[12-Frontiers/Shazeer Sparsely-Gated Mixture-of-Experts 2017]] (the sparse scaling descendant)
- [[12-Frontiers/Shazeer Sparsely-Gated Mixture-of-Experts 2017]] — Sparse conditional computation. Massive capacity without dense compute. The canonical MoE paper.
  - Connected to: [[13-Scaling-and-Alignment/DeepSeek — DeepSeek AI 2024-2025]] (MoE efficiency)
- [[12-Frontiers/Lepikhin GShard 2020]] — The systems bridge in the MoE story. Automatic sharding made giant sparse models trainable across huge TPU fleets.
  - Connected to: [[12-Frontiers/Fedus Switch Transformer 2021]] (the next simplification step), [[13-Scaling-and-Alignment/Dai et al. — DeepSeekMoE 2024]] (later sparse-model efficiency lineage)
- [[12-Frontiers/Fedus Switch Transformer 2021]] — The MoE simplification that made trillion-parameter sparse scaling practical. Top-1 routing, much less engineering pain.
  - Connected to: [[13-Scaling-and-Alignment/DeepSeek — DeepSeek AI 2024-2025]] (sparse scaling lineage), [[13-Scaling-and-Alignment/Nemotron 3 — NVIDIA 2025]] (modern hybrid sparse models)
- [[12-Frontiers/Du et al. — GLaM 2021]] — The clearest early proof that sparse LLMs could beat dense giants on the cost-quality curve. 1.2T total parameters, only ~97B active per token.
  - Connected to: [[13-Scaling-and-Alignment/Dai et al. — DeepSeekMoE 2024]] (later MoE specialization), [[12-Frontiers/Brown GPT-3 2020]] (the dense comparison point)
- [[12-Frontiers/Zoph ST-MoE 2022]] — Stable and Transferable MoE. Router z-loss, sparse-specific fine-tuning, capacity-factor tradeoffs, and a 269B sparse model make MoE dependable enough for downstream transfer.
  - Connected to: [[12-Frontiers/Du et al. — GLaM 2021]] (cost-quality sparse scaling), [[13-Scaling-and-Alignment/Dai et al. — DeepSeekMoE 2024]] (later expert-specialization recipe)
- [[12-Frontiers/Devlin BERT 2018]] — Bidirectional pretraining
- [[12-Frontiers/Liu RoBERTa 2019]] — BERT optimized
- [[12-Frontiers/Kaplan Scaling Laws 2020]] — Power laws, compute budgets, the theory behind GPT-3
- [[12-Frontiers/Child Sparse Transformer 2019]] — The efficiency bridge between the original Transformer and GPT-3. Structured sparse attention made much longer contexts computationally survivable.
- [[12-Frontiers/Dao et al. FlashAttention 2022]] — The hardware-execution bridge for long-context Transformers. IO-aware tiling, online softmax, fusion, and recomputation preserve exact dense attention while avoiding quadratic-size intermediates in GPU HBM.
  - Connected to: [[12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] (the dense-attention primitive), [[12-Frontiers/Child Sparse Transformer 2019|Sparse Transformer]] (the complementary connectivity-saving branch), [[13-Scaling-and-Alignment/DeepSeek-V4 — DeepSeek AI 2026|DeepSeek-V4]] (hybrid million-token context)
- [[12-Frontiers/Katz Backoff Language Model 1987]] — The sparse-data hinge between Shannon and neural language models. Discount rare counts, reserve probability for unseen phrases, and recursively back off to shorter histories—making finite-data language modeling usable while exposing the limits of exact symbolic contexts.
  - Connected to: [[04-Probability/Shannon Information Theory 1948|Shannon]] (probabilistic source modeling), [[12-Frontiers/Kneser Ney Backoff Language Model 1995|Kneser–Ney]] (continuation-based repair), [[12-Frontiers/Bengio Neural Probabilistic Language Model 2003|Bengio 2003]] (continuous-space repair)
- [[12-Frontiers/Kneser Ney Backoff Language Model 1995]] — Backoff learns the difference between frequency and generality. Distinct predecessor contexts, not raw token counts, estimate whether a word is plausible somewhere new—perfecting count-based novelty estimates while exposing the need for semantic sharing.
  - Connected to: [[12-Frontiers/Katz Backoff Language Model 1987|Katz backoff]] (discounting and recursive retreat), [[12-Frontiers/Bengio Neural Probabilistic Language Model 2003|Bengio 2003]] (continuous-space repair)
- [[12-Frontiers/Bengio Neural Probabilistic Language Model 2003]] — The n-gram-to-neural-language-model hinge. Learned word vectors and a smooth next-word probability function jointly replace brittle phrase tables, exposing the compute bottleneck later attacked by word2vec, recurrent models, and Transformers.
  - Connected to: [[04-Probability/Shannon Information Theory 1948|Shannon]] (probabilistic source modeling), [[10-Backpropagation/Rumelhart Hinton Williams Backpropagation 1986|backpropagation]] (jointly learned representations), [[12-Frontiers/Mikolov Recurrent Neural Language Model 2010|RNN language models]] (the fixed-window repair), [[12-Frontiers/Mikolov Word2Vec 2013|word2vec]] (the efficiency response), [[12-Frontiers/Radford GPT-1 2018|GPT-1]] (generative pretraining at Transformer scale)
- [[12-Frontiers/Mikolov Recurrent Neural Language Model 2010]] — Neural language modeling gains learned variable-length state. Recurrence removes Bengio's hard context cutoff, then exposes the long-range credit, compressed-state, serial-training, and vocabulary-cost failures repaired by LSTM, attention, efficient embeddings, and Transformers.
  - Connected to: [[12-Frontiers/Bengio Neural Probabilistic Language Model 2003|Bengio 2003]] (fixed-window ancestor), [[10-Backpropagation/Bengio Simard Frasconi Long-Term Dependencies 1994|long-range gradient diagnosis]], [[10-Backpropagation/Sutskever Vinyals Le Sequence to Sequence 2014|seq2seq]] (recurrent generation), [[12-Frontiers/Bahdanau Attention 2015|attention]] (compressed-state repair)
- [[12-Frontiers/Mikolov Word2Vec 2013]] — Distributed representations, semantic arithmetic
- [[12-Frontiers/Radford GPT-1 2018]] — Generative pretraining turns the decoder-only Transformer into a reusable language model. Raw text supplies general linguistic knowledge; task labels become a lightweight adaptation step.
  - Connected to: [[12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] (the architecture), [[12-Frontiers/Radford GPT-2 2019|GPT-2]] (fine-tuning begins to disappear), [[12-Frontiers/Devlin BERT 2018|BERT]] (the bidirectional pretraining branch)
- [[12-Frontiers/Radford GPT-2 2019]] — The hinge from language models as pretraining components to language models as zero-shot task performers. WebText + scale made unsupervised multitask learning visible.
  - Connected to: [[12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] (decoder-only architecture), [[12-Frontiers/Brown GPT-3 2020|GPT-3]] (scaled in-context learning), [[12-Frontiers/Ouyang InstructGPT RLHF 2022|InstructGPT]] (completion model → assistant)
- [[12-Frontiers/Brown GPT-3 2020]] — 175B parameters, in-context learning, the moment scale became engineering
- [[12-Frontiers/Socher Zero-Shot Learning 2013]] — Learning without examples
- [[12-Frontiers/Radford CLIP 2021]] — Vision + language in one geometric space
- [[12-Frontiers/Hoffmann Chinchilla 2022]] — "Scale data equally with model" — revised Kaplan, reshaped the industry
- [[12-Frontiers/Lewis Retrieval-Augmented Generation 2020]] — Parametric + non-parametric memory. The founding RAG paper, and the bridge from giant models to grounded enterprise systems.
  - Connected to: [[13-Scaling-and-Alignment/Claude — Anthropic 2023-2025]] (why grounded assistants matter)
- [[12-Frontiers/Wei Chain-of-Thought 2022]] — Step-by-step reasoning from prompting alone, emergent at scale
- [[12-Frontiers/Christiano et al. — Deep RL from Human Preferences 2017]] — The missing alignment precursor. Human pairwise preferences train a reward model, which then trains the policy. The conceptual backbone of RLHF.
  - Connected to: [[13-Scaling-and-Alignment/Rafailov et al. — Direct Preference Optimization 2023|DPO]] (what gets simplified later), [[13-Scaling-and-Alignment/Bai et al. — Constitutional AI 2022|Constitutional AI]] (what gets partially replaced with AI feedback)
- [[12-Frontiers/Ziegler et al. — Fine-Tuning Language Models from Human Preferences 2019]] — The first serious RLHF-for-language bridge. Human preferences become a reward signal for text generation itself, with KL-regularized policy updates and early lessons about proxy gaming.
  - Connected to: [[12-Frontiers/Stiennon et al. — Learning to Summarize from Human Feedback 2020]] (the next tightening of the recipe), [[12-Frontiers/Ouyang InstructGPT RLHF 2022|InstructGPT / RLHF]] (the later general-assistant version)
- [[12-Frontiers/Stiennon et al. — Learning to Summarize from Human Feedback 2020]] — The crucial bridge from Christiano-style preference learning to RLHF for language models. Reward modeling moved from RL environments into text generation.
  - Connected to: [[12-Frontiers/Ouyang InstructGPT RLHF 2022|InstructGPT / RLHF]] (the next scale-up step)
- [[12-Frontiers/Ouyang InstructGPT RLHF 2022]] — Teaching language models to follow instructions with human feedback. The bridge from text completion to useful assistants.
- [[12-Frontiers/Gu Goel Ré S4 2021]] — The structured state space breakthrough. The paper that made SSMs computationally practical and reopened the long-sequence architecture race.
  - Connected to: [[12-Frontiers/Gu Dao Mamba 2023|Mamba]] (direct lineage)
- [[12-Frontiers/Gu Dao Mamba 2023]] — Selective state space models. Linear-time complexity O(n) vs. Transformer's O(n²). First serious post-Transformer paradigm. 5× faster inference, constant memory.
- [[12-Frontiers/Dao Gu Mamba-2 2024]] — State space duality: attention and SSMs meet in one framework. Mamba-2 refines Mamba with cleaner theory and 2–8× faster training.
  - Connected to: [[12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] (the comparison class)

### 13 — Scaling & Alignment Era (2022–2026)
- [[13-Scaling-and-Alignment/Reasoning RL Interview Bridge - Test-Time Compute]] — Interview synthesis for "what came after RLHF?": RLHF → DPO → verifiable-reward RL (GRPO/R1), and why test-time compute became a second scaling axis. The senior-signal answer, current to 2025.
  - Connected to: [[13-Scaling-and-Alignment/Rafailov et al. — Direct Preference Optimization 2023|DPO]] (cheap post-training), [[13-Scaling-and-Alignment/Schulman et al. — Proximal Policy Optimization 2017|PPO]] (the RL workhorse)
- [[13-Scaling-and-Alignment/Schulman et al. — Proximal Policy Optimization 2017]] — The practical RL optimizer inside early RLHF. Clipped policy updates let reward-guided post-training move a model without letting it drift wildly.
- [[13-Scaling-and-Alignment/Bai et al. — Constitutional AI 2022]] — Teaching AI to self-govern with principles. RL from AI Feedback (RLAIF). The foundation of Claude.
- [[13-Scaling-and-Alignment/ChatGPT — OpenAI 2022-2025]] — The product that made AI mainstream. GPT-3.5 → GPT-4o → o1/o3.
- [[13-Scaling-and-Alignment/Claude — Anthropic 2023-2025]] — Constitutional AI. Safety-first.
- [[13-Scaling-and-Alignment/Gemini — Google DeepMind 2023-2025]] — Natively multimodal. 1M+ context. Baked into everything Google.
- [[13-Scaling-and-Alignment/Touvron et al. — LLaMA 2023]] — Open and efficient foundation models. Smaller GPT-style models trained longer on public data made frontier-adjacent capability portable and seeded the open-weight ecosystem.
- [[13-Scaling-and-Alignment/Llama — Meta 2023-2025]] — Open weights that democratized LLMs. LLaMA → Llama 4.
- [[13-Scaling-and-Alignment/Qwen — Alibaba 2023-2025]] — Multilingual excellence. Chinese-first, globally competitive.
- [[13-Scaling-and-Alignment/Dai et al. — DeepSeekMoE 2024]] — Fine-grained + shared experts. The architectural paper behind DeepSeek's specialization and efficiency story.
  - Connected to: [[13-Scaling-and-Alignment/DeepSeek — DeepSeek AI 2024-2025]] (the model family this enables)
- [[13-Scaling-and-Alignment/DeepSeek — DeepSeek AI 2024-2025]] — Efficiency breakthroughs. MoE + $5.6M training. The constraint catalyst.
- [[13-Scaling-and-Alignment/DeepSeek-V4 — DeepSeek AI 2026]] — Million-token context efficiency. V4-Pro (1.6T total / 49B active) and V4-Flash (284B / 13B active) combine hybrid CSA/HCA attention, mHC residual stabilization, Muon optimization, DeepSeekMoE, FP4/FP8-aware training, and specialist-then-consolidate post-training.
- [[13-Scaling-and-Alignment/GLM — Zhipu AI 2023-2025]] — Bilingual architecture.
- [[13-Scaling-and-Alignment/Manik Wang Gemma Phi Qwen Tradeoffs 2026]] — Deployment-aware benchmarking for compact dense and MoE reasoning models. Accuracy, latency, VRAM, task family, and prompting regime all become one model-selection problem.
- [[13-Scaling-and-Alignment/Kimi — Moonshot AI 2024-2025]] — Long context as product. 2M tokens.
- [[13-Scaling-and-Alignment/Nemotron 3 — NVIDIA 2025]] — Hybrid Mamba-Transformer MoE. The GPU maker's own models. LatentMoE, NVFP4, 1M context.
- [[13-Scaling-and-Alignment/NVIDIA Nemotron 3 Super 2026]] — The source-paper hinge for agentic throughput: hybrid Mamba-Attention MoE, LatentMoE, MTP, NVFP4 pretraining, 1M context, and multi-environment RL for tool-using agents.

---

## Stats
- 📄 Notes: 122 (added MusicGen — September 24, 2026; excludes the three MOC files)
- 📑 PDFs downloaded: 124 (filesystem count)
- 🔗 Connected to: [[../Mathematics/_MOC|Mathematics]]

---

#moc #ai-ml #library #lineage
