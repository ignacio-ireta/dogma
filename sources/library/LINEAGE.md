# 🧬 Lineage Map — The Genealogy of Ideas

## AI/ML Thread

```
McCulloch-Pitts 1943 (logical neurons) ✅ noted
├── Hebb 1949 (learning rule) ✅ noted
│   └── Rosenblatt 1958 (Perceptron — learned classification) ✅ noted
│       ├── Block / Novikoff 1962 (convergence proof; finite-mistake theorem) ✅ noted
│       ├── Widrow-Hoff 1960 (ADALINE, LMS, gradient descent) ✅ noted
│       │   ├── Cauchy 1847 (steepest descent — connects to Math thread!) ✅ noted
│       │   │   └── Robbins-Monro 1951 (stochastic approximation — noisy sample-based updates) ✅ noted
│       │   │       ├── Polyak 1964 (heavy-ball momentum — optimizer memory) ✅ noted
│       │   │       └── AdaGrad 2011 (coordinate-wise adaptive learning rates) ✅ noted
│       │   │           └── RMSProp 2012 (recent squared-gradient memory for minibatches) ✅ noted
│       │   └── Loshchilov-Hutter 2017 (SGDR — cosine annealing + warm restarts) ✅ noted
│       └── Minsky-Papert 1969 (Perceptrons book — limitations) ✅ noted
│           └── Werbos 1974 (ordered derivatives; hidden-variable credit assignment) ✅ noted
│               └── AI WINTER (1970s-80s; the method existed before broad adoption)
│                   └── Rumelhart-Hinton-Williams 1986 (backpropagation) ✅ noted
│                   ├── LeCun 1989 (CNNs, handwriting recognition) ✅ noted
│                   │   ├── LeNet-5 / LeCun et al. 1998 (mature CNN + document recognition) ✅ noted
│                   │   │   └── Krizhevsky-Sutskever-Hinton 2012 (AlexNet — ImageNet) ✅ noted
│                   │   └── LeCun-Bengio-Hinton 2015 (Deep Learning review) ✅ noted
│                   ├── Hopfield 1982 (energy-based models — physics connection) ✅ noted
│                   │   └── Boltzmann Machines (Hinton-Sejnowski 1983) ✅ noted
│                   │       └── Hinton 2002 (Contrastive Divergence / RBMs) ✅ noted
│                   │           └── efficient RBM training enables Hinton 2006 DBN pretraining
│                   ├── Bengio-Simard-Frasconi 1994 (robust recurrent memory makes long-range gradients vanish) ✅ noted
│                   │   └── Hochreiter-Schmidhuber 1997 (LSTM) ✅ noted
│                   │       ├── Gers-Schmidhuber-Cummins 2000 (forget gate — learned reset for continual streams) ✅ noted
│                   │       └── neural encoder-decoder sequence learning 2014
│                   │           ├── Cho et al. 2014 (Encoder-Decoder, GRU) ✅ noted
│                   │           └── Sutskever-Vinyals-Le 2014 (deep LSTM seq2seq; fixed-vector bottleneck) ✅ noted
│                   │               └── Bahdanau, Cho, Bengio 2015 (Attention repairs the fixed-vector bottleneck) ✅ noted
│                   │                   ├── Sennrich, Haddow & Birch 2016 (subword BPE — finite vocabulary without closed language) ✅ noted
│                   │                   │   └── GPT/BERT-era subword tokenization (tokens become the units of pretraining and compute)
│                   │                   └── Luong, Pham & Manning 2015 (global/local attention, scoring functions, input feeding) ✅ noted
│                   │                       └── Vaswani et al. 2017 (Attention Is All You Need) ✅ noted
│                   │                           ├── Dai et al. 2019 (Transformer-XL — segment recurrence + relative positions repair fixed-window amnesia) ✅ noted
│                   │                           └── Child et al. 2019 (Sparse Transformer — structured attention reduces the long-context cost) ✅ noted
│                   │                               └── Dao et al. 2022 (FlashAttention — IO-aware tiling makes exact attention practical without materializing N×N intermediates) ✅ noted
│                   │                   ├── BERT 2018 (Devlin et al.) ✅ noted
│                   │                   │   └── RoBERTa 2019 (Liu et al.) ✅ noted
│                   │                   ├── DCGAN 2016 (Radford et al.) ✅ noted
│                   │                   ├── GPT series (Radford et al. 2018→2023)
│                   │                   │   └── GPT-1 2018 (generative pretraining + supervised fine-tuning) ✅ noted
│                   │                   │       └── GPT-2 2019 (unsupervised multitask learning; fine-tuning begins to disappear) ✅ noted
│                   │                   ├── Jacobs et al. 1991 (adaptive mixtures of local experts) ✅ noted
│                   │                   │   └── Jordan & Jacobs 1994 (hierarchical MoE + EM) ✅ noted (jordan-jacobs-hierarchical-mixtures-experts-1994.pdf)
│                   │                   │       └── Shazeer et al. 2017 (sparsely-gated MoE) ✅ noted
│                   │                   │           └── Switch Transformer 2021 (Fedus et al.) ✅ noted
│                   │                   │               └── GShard 2020 / GLaM 2021 ✅ noted
│                   │                   │                   └── Zoph et al. ST-MoE 2022 ✅ noted (router z-loss; stable transferable sparse experts)
│                   │                   │                       └── practical trillion-parameter MoE becomes stable enough to transfer
│                   │                   └── Scaling Laws 2020 (Kaplan et al.) ✅ noted
│                   │                       └── Brown et al. GPT-3 2020 ✅ noted (2005.14165)
│                   │                           ├── Hoffmann et al. Chinchilla 2022 ✅ noted (2203.15556)
│                   │                           │   └── Touvron et al. LLaMA 2023 ✅ noted (2302.13971)
│                   │                           ├── Wei et al. Chain-of-Thought 2022 ✅ noted (2201.11903)
│                   │                           ├── Christiano et al. 2017 (human preferences) ✅ noted
│                   │                           │   ├── Schulman et al. 2017 (PPO — stable policy updates for RLHF) ✅ noted (1707.06347)
│                   │                           │   └── Ziegler et al. 2019 (language-model RLHF bridge) ✅ noted (1909.08593)
│                   │                           │       └── Stiennon et al. 2020 (summarization from human feedback) ✅ noted (2009.01325)
│                   │                           │           └── Ouyang et al. InstructGPT RLHF 2022 ✅ noted (2203.02155)
│                   │                           │               ├── Bai et al. Constitutional AI 2022 ✅ noted (2212.08073)
│                   │                           │               └── Rafailov et al. DPO 2023 ✅ noted (2305.18290)
│                   │                           │                   └── DeepSeek-R1 2025 ✅ noted (2501.12948; pure-RL reasoning via GRPO + verifiable rewards, no SFT/no human traces — test-time compute becomes a scaling axis)
│                   │                           └── DeepSeekMoE 2024 ✅ noted
│                   │                               └── DeepSeek-V4 2026 ✅ noted (deepseek-v4-2026.pdf; 1M context via CSA/HCA, mHC, Muon, FP4/FP8-aware training)
│                   │                                   └── NVIDIA Nemotron 3 Super 2026 ✅ noted (2604.12374; hybrid Mamba-Attention MoE for agentic throughput)
│                   │                                       └── Manik & Wang 2026 ✅ noted (2604.07035; prompt-conditioned accuracy-efficiency frontier for dense/MoE reasoning models)
│                   └── Hinton 2006 (deep belief nets — "deep learning" revival) ✅ noted
│                       ├── Kingma-Welling 2013 (VAE — amortized inference + reparameterized stochastic latents) ✅ noted
│                       │   └── van den Oord et al. 2017 (VQ-VAE — learned discrete codes + straight-through estimation prevent posterior collapse) ✅ noted
│                       │       ├── Esser-Rombach-Ommer 2021 (VQGAN — perceptual/adversarial compression makes visual-token Transformers practical) ✅ noted
│                       │       │   └── Rombach et al. 2022 (Latent Diffusion — keep the perceptual compressor, replace the serial autoregressive prior with latent denoising) ✅ noted
│                       │       └── Zeghidour et al. 2021 (SoundStream — residual codec tokens make general audio compact, scalable, and streamable) ✅ noted
│                       │           └── Borsos et al. 2022 (AudioLM — semantic planning plus hierarchical acoustic realization) ✅ noted
│                       │               └── Agostinelli et al. 2023 (MusicLM — MuLan bridges text control to audio-only generative training) ✅ noted
│                       │                   └── Copet et al. 2023 (MusicGen — delay-interleaved codec streams collapse the cascade into one LM) ✅ noted
│                       │                       └── speech-to-speech models (codec tokens become a conditioned generative vocabulary)
│                       ├── Glorot-Bengio 2010 (Xavier initialization — preserve activation/gradient scale) ✅ noted
│                       │   └── Nair-Hinton 2010 (ReLU — non-saturating rectifiers) ✅ noted
│                       │       ├── Krizhevsky-Sutskever-Hinton 2012 (AlexNet — ImageNet) ✅ noted
│                       │       ├── He et al. 2015 (rectifier-aware/Kaiming initialization + PReLU) ✅ noted
│                       │       │   └── He et al. 2015/2016 (ResNets — identity paths repair the remaining depth degradation) ✅ noted
│                       │       └── Ioffe-Szegedy 2015 (Batch Normalization — internal activation stabilization) ✅ noted
│                       │           ├── He et al. 2015/2016 (ResNets — identity paths repair depth degradation) ✅ noted
│                       │           └── Ba-Kiros-Hinton 2016 (Layer Normalization — per-example sequence-model stabilization) ✅ noted
│                       │               └── Vaswani et al. 2017 (Transformer Add & Norm) ✅ noted
│                       ├── Hinton et al. 2012 (Dropout) ✅ noted
│                       ├── Goodfellow 2014 (GANs) ✅ noted
│                       │   └── Ho, Jain & Abbeel 2020 (DDPM — stable noise-prediction training replaces the minimax game, but sampling becomes iterative) ✅ noted
│                       │       └── Song, Meng & Ermon 2021 (DDIM — same denoiser, non-Markovian deterministic path, 10–50× faster sampling) ✅ noted
│                       │           └── Rombach et al. 2022 (Latent Diffusion — move iterative denoising into a mildly compressed spatial code; cross-attention enables text conditioning) ✅ noted
│                       ├── Mikolov 2013 (word2vec — semantic arithmetic) ✅ noted
│                       │   └── Socher 2013 (Zero-Shot Learning via word vectors) ✅ noted
│                       │       └── Radford 2021 (CLIP — contrastive language-image pre-training) ✅ noted

Shannon 1948 (information theory — entropy, bits, source modeling) ✅ noted
└── probabilistic language modeling as compression / uncertainty reduction
    ├── Katz 1987 (Good-Turing discounting + recursive backoff for sparse n-grams) ✅ noted
    │   └── Kneser-Ney 1995 (continuation-based backoff rewards contextual diversity) ✅ noted
    │       └── exact short contexts remain combinatorially sparse and cannot share semantic evidence
    │           └── Bengio et al. 2003 (neural probabilistic language model) ✅ noted
    │               ├── Mikolov et al. 2010 (RNN language model — learned variable-length state) ✅ noted
    │               │   └── simple recurrence breaks on long-range credit + serial compression
    │               │       └── LSTM language models / seq2seq → attention → Transformer pretraining
    │               ├── fixed word vocabularies still collapse rare and unseen forms into UNK
    │               │   └── Sennrich, Haddow & Birch 2016 (subword BPE — open text through a finite learned inventory) ✅ noted
    │               └── Mikolov 2013 (word2vec — cheaper large-scale word geometry) ✅ noted
    ├── Kaplan et al. 2020 (scaling laws over cross-entropy loss) ✅ noted
    └── GPT-style next-token prediction (GPT-2 → GPT-3 → assistants) ✅ noted

Fisher 1936 (linear discriminant — supervised classification geometry) ✅ noted
└── Boser-Guyon-Vapnik 1992 (optimal-margin classifiers / hard-margin SVM) ✅ noted
    └── Cortes-Vapnik 1995 (soft-margin support-vector networks) ✅ noted
```

### Notes created (September 24, 2026)
- **Copet et al. 2023 / Simple and Controllable Music Generation (MusicGen)** ✅ noted — closes MusicLM’s multi-model efficiency gap. MusicLM proved text and melody could steer high-fidelity music, but its semantic, coarse-acoustic, and fine-acoustic cascade accumulated latency and irreversible handoffs; exactly flattening four residual-codebook streams also multiplied autoregressive length. MusicGen formalizes codebook interleaving and chooses a delay pattern that lets later RVQ levels depend on earlier coarse decisions while predicting one group per 50 Hz frame. Thirty seconds requires 1,500 autoregressive steps instead of 6,000 under full flattening, enabling one Transformer to handle text, unsupervised melody guidance, and stereo. The repair makes codec-token generation simpler and open through AudioCraft, while leaving long-form composition, fine-grained control, metric validity, cultural coverage, and artist compensation unresolved. Source PDF: `corpus/raw/library-papers/2306.05284.pdf`

### Notes created (September 22, 2026)
- **Agostinelli et al. 2023 / MusicLM: Generating Music From Text** ✅ noted — closes AudioLM’s conditioning-and-data gap. AudioLM could model coherent semantic and acoustic token hierarchies, but direct text conditioning appeared to require captions for generative-scale music data; one global caption was also a weak description of a long temporal object. MusicLM freezes MuLan’s shared music–text embedding, conditions on audio embeddings across 280,000 hours of uncaptioned music during training, then substitutes nearby text embeddings at inference. A semantic Transformer plans at 25 tokens per second before coarse and fine SoundStream stages realize 600 acoustic tokens per second; removing the semantic stage preserved broad audio quality but weakened caption adherence and long-term structure. The model generates 24 kHz music over minutes, accepts combined text-plus-hummed-melody control, and introduces the 5,521-example MusicCaps benchmark. Its three serial autoregressive stages, weak negation/event ordering, local sliding-window continuation, and unresolved provenance costs establish the next problems for MusicGen and later audio models. Source PDF: `corpus/raw/library-papers/2301.11325.pdf`

### Notes created (September 20, 2026)
- **Borsos et al. 2022 / AudioLM: a Language Modeling Approach to Audio Generation** ✅ noted — closes the SoundStream → audio-language-model gap. A high-rate codec stream reconstructs sound well but makes long-range modeling expensive and lets local waveform detail swamp syntax, semantics, melody, and rhythm; low-rate self-supervised speech units preserve that structure but cannot reconstruct a voice. AudioLM refuses the false choice: a 25 Hz w2v-BERT token stream plans global content, four coarse SoundStream codebooks recover identity and recording conditions, and eight fine residual codebooks restore local fidelity through three staged decoder-only Transformers. Trained on 60,000 hours of untranscribed speech, it reached 83.7% in-vocabulary sWUGGY, 64.7% sBLIMP, and 92.6% prompt-speaker retention; human real-versus-generated judgments were 51.2%, statistically indistinguishable from chance. The repair makes learned compression a generative hierarchy and establishes the semantic-planning → acoustic-realization pattern inherited by MusicLM, MusicGen, and speech-to-speech systems, while exposing serial generation and voice-provenance risks as the next breaks. Source PDF: `corpus/raw/library-papers/2209.03143.pdf`

### Notes created (September 19, 2026)
- **Zeghidour et al. 2021 / SoundStream: An End-to-End Neural Audio Codec** ✅ noted — closes the unresolved VQ-VAE → neural audio codec bridge. A flat vector quantizer cannot scale to useful audio rates: at 24 kHz, 75 latent frames per second, and 6 kbps, each frame needs 80 bits, implying an impossible `2⁸⁰`-entry codebook. SoundStream replaces that one decision with residual vector quantization, where eight 1,024-entry codebooks successively encode the remaining error; structured quantizer dropout lets one causal encoder/decoder select 3–18 kbps at inference without changing latent shape or retraining. Multi-scale spectral, feature-matching, and adversarial losses preserve perceived sound rather than literal samples, while the default model runs faster than real time on one Pixel 4 CPU thread. At 3 kbps it outperformed Opus at 12 kbps and approached EVS at 9.6 kbps across speech, music, and general audio. The repair turns learned discrete latents into a deployable bitstream and the acoustic vocabulary inherited by EnCodec, AudioLM, MusicGen, and multimodal token models; it also exposes the next break—long acoustic token streams still need semantic hierarchy. Source PDF: `corpus/raw/library-papers/2107.03312.pdf`

### Notes created (September 17, 2026)
- **Esser, Rombach & Ommer 2021 / Taming Transformers for High-Resolution Image Synthesis (VQGAN)** ✅ noted — closes the unresolved bridge from VQ-VAE’s trainable discrete codes to practical high-resolution visual generation. Quantizing pixels or using a reconstruction-heavy codebook left Transformers with prohibitively long sequences; compressing harder destroyed visible detail. VQGAN trains a convolutional codebook with perceptual reconstruction and a patch discriminator so each index represents a context-rich visual part, then models the shorter index grid with a causal Transformer. A 256×256 image can become 256 tokens; the paper reports ImageNet reconstruction FID 7.94 with a 1,024-entry codebook, versus DALL-E’s 32.01 with four times as many tokens, and a controlled CIFAR-10 comparison found 14.08× faster sampling than a quantized-pixel Transformer. Sliding-window attention reaches megapixel synthesis but exposes the remaining serial and global-coherence limits. The learned perceptual compressor directly enables latent diffusion, which keeps the cheaper spatial representation and replaces next-token sampling with denoising. Source PDF: `corpus/raw/library-papers/2012.09841.pdf`

### Notes created (September 15, 2026)
- **van den Oord, Vinyals & Kavukcuoglu 2017 / Neural Discrete Representation Learning (VQ-VAE)** ✅ noted — closes the discrete-bottleneck gap after the VAE. Gaussian reparameterization made continuous latents trainable, but powerful autoregressive decoders could optimize the ELBO by ignoring them, while genuinely categorical choices broke ordinary backpropagation. VQ-VAE maps encoder outputs to the nearest vector in a learned finite codebook, copies decoder gradients through that hard choice with a straight-through estimator, and stabilizes the division of labor with separate dictionary and commitment losses. Training the representation first and an autoregressive prior over its indices second moves expensive modeling from raw pixels or waveform samples into shorter semantic symbol sequences; the paper reports roughly 42.6× image-bit compression and unsupervised speech codes reaching 49.3% phoneme classification versus a 7.2% baseline. This establishes the learned-tokenizer pattern behind VQ-VAE-2, VQGAN, neural audio codecs, visual tokens, and the compressed-space economics later inherited by latent diffusion. Source PDF: `corpus/raw/library-papers/1711.00937.pdf`

### Notes created (September 13, 2026)
- **Song, Meng & Ermon 2021 / Denoising Diffusion Implicit Models** ✅ noted — closes DDPM’s serial-sampling gap without discarding its stable training objective. DDPM’s noise-prediction loss constrains each noisy marginal but does not uniquely require the thousand-step Markov chain used to derive it. DDIM preserves those marginals and the same pretrained denoiser while selecting a non-Markovian, optionally deterministic reverse trajectory over a sparse timestep subsequence. The result reaches roughly 1,000-step quality in 20–100 evaluations for a reported 10–50× wall-clock speedup, while deterministic trajectories also make the initial noise a consistent latent code for inversion and semantic interpolation. The repair separates model training from sampler design and exposes the next two costs: remaining serial evaluations and full-resolution processing, answered by better solvers/distillation and latent diffusion respectively. Source PDF: `corpus/raw/library-papers/2010.02502.pdf`

### Notes created (September 12, 2026)
- **Rombach et al. 2022 / High-Resolution Image Synthesis with Latent Diffusion Models** ✅ noted — closes the compute gap after DDPM. Pixel-space diffusion stabilized generative training but repeatedly evaluated a U-Net over perceptually redundant RGB dimensions, making high-resolution training and sampling prohibitively expensive. LDM separates perceptual compression from semantic generation: a frozen, mildly compressive autoencoder removes mostly imperceptible detail; DDPM-style noise prediction runs in its smaller spatial code; and cross-attention connects text, layouts, or other conditions to the denoiser. This preserves diffusion quality while cutting controlled inpainting training/sampling cost by at least 2.7× and establishes the architecture behind Stable Diffusion. Source PDF: `corpus/raw/library-papers/2112.10752.pdf`

### Notes created (September 8, 2026)
- **Kingma & Welling 2013 / Auto-Encoding Variational Bayes** ✅ noted — closes the gap between nonlinear autoencoders and scalable probabilistic generation. Deterministic reconstruction codes did not define a reliable distribution to sample, while directed latent-variable models hid learning behind an intractable marginal-likelihood integral and expensive per-datapoint posterior inference. AEVB learns one amortized probabilistic encoder q_phi(z|x), jointly trains it with a decoder by maximizing the reconstruction-minus-KL ELBO, and rewrites Gaussian sampling as z = mu + sigma times epsilon so low-variance gradients pass through stochastic latent choices. The repair makes continuous latent-variable models compatible with minibatch backpropagation, enabling the VAE family and the compressed-latent lineage that later converges with diffusion in latent diffusion models. Source PDF: `corpus/raw/library-papers/1312.6114.pdf`

### Notes created (September 5, 2026)
- **DeepSeek-AI 2025 / DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** ✅ noted — closes the reasoning-source gap after Chain-of-Thought and human-demonstration post-training. CoT and SFT-on-reasoning-traces improved reasoning but bottlenecked it twice: human annotations don't scale, and imitating human thought caps the model at human performance while forbidding non-human strategies. Meanwhile pure pretraining scale (Kaplan/Chinchilla) was hitting diminishing returns. R1 opens a second axis: freeze DeepSeek-V3-Base and apply pure RL (GRPO — group-normalized advantage, no critic) against a rule-based *verifiable* reward (final-answer correctness on math/code), deliberately skipping SFT. R1-Zero then self-evolves longer responses with emergent self-verification, reflection, and "aha" backtracking that nobody demonstrated; the production R1 wraps this in cold-start SFT → reasoning RL → rejection sampling → final SFT+RL for readability and general use, and distills the behavior into smaller dense models that beat their instruction-tuned originals. The repair makes **test-time compute** a first-class scaling axis and reframes evaluation as an accuracy-latency-memory frontier (Manik & Wang 2026). Source PDF: `corpus/raw/library-papers/2501.12948.pdf`

### Notes created (September 3, 2026)
- **Ho, Jain & Abbeel 2020 / Denoising Diffusion Probabilistic Models** ✅ noted — closes the adversarial-generation gap between GANs and modern visual generative AI. GANs produced sharp images but coupled generator progress to a moving discriminator, leaving mode collapse, vanishing gradients, and unstable two-player optimization. DDPM fixes a Gaussian noising process, samples any corruption level directly, and trains a time-conditioned U-Net with a simple known-noise regression target; generation reverses the process from noise through 1,000 learned denoising transitions. The result reached unconditional CIFAR-10 FID 3.17 and made diffusion competitive with GANs, while exposing its own next break: excellent samples required hundreds or thousands of sequential network evaluations, motivating DDIM, guidance, latent diffusion, distillation, and consistency/flow methods. Source PDF: `corpus/raw/library-papers/2006.11239.pdf`

### Notes created (September 1, 2026)
- **Dao et al. 2022 / FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** ✅ noted — closes the hardware-execution gap after dense and sparse Transformer attention. Lower-FLOP approximations often failed to become faster because GPU memory traffic, not arithmetic alone, dominated runtime. FlashAttention tiles `Q`, `K`, and `V` through on-chip SRAM, computes softmax incrementally, and recomputes blocks during backpropagation so the full `N × N` attention matrix never has to be materialized in HBM. It preserves exact dense-attention semantics while reducing IO and activation memory, giving longer context and faster training; the remaining quadratic arithmetic cost explains why sparse, compressed, and hybrid attention still follow. Source PDF: `corpus/raw/library-papers/2205.14135.pdf`

### Notes created (August 29, 2026)
- **Sennrich, Haddow & Birch 2016 / Neural Machine Translation of Rare Words with Subword Units** ✅ noted — closes the vocabulary-interface gap between word-level neural models and token-based generative pretraining. Fixed word lists made embedding and output matrices manageable only by collapsing rare and unseen forms into `UNK`; merely enlarging them consumed parameters without making language closed, while pure characters made sequences needlessly long. Frequency-learned BPE keeps common forms compact and composes the tail from reusable subword pieces, giving a finite symbol inventory productive open-text coverage. The repair enables Transformer and GPT-style pretraining over manageable tokens while exposing tokenization itself as a systems tradeoff among vocabulary size, sequence length, multilingual sharing, context capacity, and compute. Source PDF: `corpus/raw/library-papers/1508.07909.pdf`

### Notes created (August 27, 2026)
- **Dai et al. 2019 / Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context** ✅ noted — closes the fixed-segment memory gap after the original Transformer. Parallel self-attention could connect every token inside one training window, but practical language models discarded all hidden state at arbitrary segment boundaries, losing longer dependencies and starving each segment's opening tokens of context. Transformer-XL caches prior hidden states as stop-gradient memory and pairs their reuse with relative positional attention, preserving temporal order without recomputing the past. The repair exposes the next wall: retained memory remains finite and attention over it still costs compute, motivating sparse, compressed, retrieved, and state-space context mechanisms. Source PDF: `corpus/raw/library-papers/1901.02860.pdf`

### Notes created (August 25, 2026)
- **Reinhard Kneser & Hermann Ney 1995 / Improved Backing-Off for M-Gram Language Modeling** ✅ noted — closes the lower-order-distribution gap inside Katz-style backoff. Katz knew how much probability to reserve and when to retreat, but ordinary lower-order frequencies could make a word look broadly plausible merely because it appeared many times in a few rigid phrases. Kneser and Ney optimize the backoff distribution for unseen events, replacing raw frequency with contextual diversity: how many distinct histories licensed a continuation. Two derivations—marginal constraints and leave-one-out estimation—reach closely related distributions, reducing reported perplexity by up to about 10% and word error rate by about 5%. The repair perfects exact-count novelty estimates while exposing the remaining inability to share evidence across semantically similar symbols, the break answered by Bengio's continuous representations. Source PDF: `corpus/raw/library-papers/kneser-ney-backing-off-1995.pdf`

### Notes created (August 23, 2026)
- **Slava M. Katz 1987 / Estimation of Probabilities from Sparse Data for the Language Model Component of a Speech Recognizer** ✅ noted — closes the Shannon → count-based language-model gap. Maximum-likelihood n-grams assign zero probability to every phrase absent from a finite corpus; Katz discounts unreliable low counts, reserves the freed probability mass for unseen events, and recursively backs off to shorter conditional histories. The repair made finite-data language modeling practical but could generalize only by forgetting context, exposing the exact-symbol and combinatorial-sparsity failures that Bengio's learned continuous representations later addressed. Source PDF: `corpus/raw/library-papers/katz-backoff-language-model-1987.pdf`

### Notes created (August 22, 2026)
- **Mikolov, Karafiát, Burget, Černocký & Khudanpur 2010 / Recurrent Neural Network Based Language Model** ✅ noted — closes the fixed-window gap after Bengio's feed-forward neural language model. A recurrent hidden state replaces an ad hoc list of the last five to ten words with a learned variable-length summary, producing large perplexity and speech-recognition gains over heavily trained n-grams. The repair is deliberately incomplete: truncated credit assignment, a single compressed state, serial computation, and vocabulary-wide output cost expose why LSTM language models, seq2seq, attention, efficient embeddings, and Transformers were still necessary. Source PDF: `corpus/raw/library-papers/mikolov-rnn-language-model-2010.pdf`

### Notes created (August 20, 2026)
- **Bengio, Ducharme, Vincent & Jauvin 2003 / A Neural Probabilistic Language Model** ✅ noted — closes the n-gram → neural language model gap. Exact short-context tables could smooth unseen phrases but could not share evidence across semantically neighboring sentences, and increasing context length made sparsity explode combinatorially. Bengio et al. jointly learn continuous word vectors and a smooth next-word probability function, letting one sentence inform a combinatorial neighborhood while keeping parameters linear rather than exponential in context size. The repair exposes the next bottlenecks—fixed windows and vocabulary-wide softmax cost—answered by recurrent language models, word2vec, efficient output objectives, Transformers, and GPT-style generative pretraining. Source PDF: `corpus/raw/library-papers/bengio-neural-probabilistic-language-model-2003.pdf`

### Notes created (August 18, 2026)
- **He, Zhang, Ren & Sun 2015 / Delving Deep into Rectifiers** ✅ noted — closes the activation-initialization mismatch between Xavier/ReLU and ResNet. Xavier's linear derivation did not account for ReLU zeroing roughly half of its activations and local derivatives, so signals could shrink exponentially through depth. Kaiming initialization restores forward and backward variance with `Var(w)=2/fan_in`, allowing a 30-layer rectified network to train from scratch; PReLU learns the negative slope at negligible cost. Crucially, that convergent 30-layer plain network still performed worse than a 14-layer one, cleanly exposing the depth-degradation problem ResNet repaired. Source PDF: `corpus/raw/library-papers/1502.01852.pdf`

### Notes upgraded (August 16, 2026)
- **He et al. 2015/2016 / Deep Residual Learning for Image Recognition** ✅ source-grounded and rebuilt — closes the depth-degradation gap between AlexNet-era deep CNNs and the residual streams inherited by Transformers. Initialization and BatchNorm kept signals healthy, yet deeper plain networks still produced higher training error because SGD could not reliably discover the identity-extended shallow solution. ResNet protects an identity path and asks each block to learn only a correction, making 50/101/152-layer networks optimizable without adding parameters to matching-dimension shortcuts. Source PDF: `corpus/raw/library-papers/1512.03385.pdf`

### Notes created (August 15, 2026)
- **Ba, Kiros & Hinton 2016 / Layer Normalization** ✅ noted — closes the normalization-axis gap between BatchNorm and Transformers. BatchNorm stabilized deep feed-forward networks but depended on mini-batch statistics, required separate handling at inference, and became awkward for variable-length recurrent computation. LayerNorm computes statistics across one example’s hidden features at each time step, making normalization independent of batch composition and identical at training and inference. This is the missing numerical bridge into the Transformer’s Add & Norm blocks. Source PDF: `corpus/raw/library-papers/1607.06450.pdf`

### Notes created (August 13, 2026)
- **Glorot & Bengio 2010 / Understanding the Difficulty of Training Deep Feedforward Neural Networks** ✅ noted — closes the end-to-end trainability gap between DBN-era layer-wise pretraining and the ReLU/AlexNet era. Backpropagation could compute every derivative, but logistic saturation and layerwise variance drift made forward activations and backward gradients decay or become ill-conditioned through depth. Xavier initialization balances fan-in and fan-out so both streams retain useful scale, turning initialization from arbitrary randomness into part of the architecture. Source PDF: `corpus/raw/library-papers/glorot-bengio-initialization-2010.pdf`

### Source Strategy for AI/ML Papers

| Paper | Year | arXiv? | Alt Source |
|-------|------|--------|-----------|
| Cauchy steepest descent | 1847 | ❌ | Comptes Rendus English translation ✅ noted |
| McCulloch-Pitts | 1943 | ❌ | Semantic Scholar / original journal ✅ noted |
| Shannon | 1948 | ❌ | Bell System Technical Journal reprint / Harvard-hosted corrected reprint ✅ noted |
| Katz back-off language model | 1987 | ❌ | IEEE paper scan / CiteSeerX mirror ✅ noted |
| Kneser-Ney improved backoff | 1995 | ❌ | ICASSP paper / RWTH Aachen institutional scan ✅ noted |
| Hebb | 1949 | ❌ | Book: "The Organization of Behavior" ✅ noted |
| Rosenblatt | 1958 | ❌ | Psychological Review 65(6), 386–408 / `rosenblatt-perceptron-1958.pdf` (Wayback capture of UPenn scan, 23pp verified) ✅ noted |
| Robbins-Monro | 1951 | ❌ | Annals of Mathematical Statistics / JSTOR scan ✅ noted |
| Widrow-Hoff | 1960 | ❌ | Stanford-hosted WESCON paper scan ✅ noted |
| Novikoff | 1962 | ❌ | Mathematical Theory of Automata proceedings scan ✅ noted |
| Minsky-Papert | 1969 | ❌ | Book (MIT Press) / local scanned PDF excerpt ✅ noted |
| Werbos | 1974 | ❌ | Harvard PhD thesis scan ✅ noted |
| Rumelhart et al. | 1986 | ❌ | Nature 323, 533–536 ✅ noted |
| Bengio, Simard & Frasconi | 1994 | ❌ | IEEE Transactions on Neural Networks 5(2), 157–166 ✅ noted |
| Hopfield | 1982 | ❌ | PNAS 79(8), 2554–2558 / `hopfield1982.pdf` ✅ noted |
| LeCun et al. | 1989 | ❌ | Neural Computation / `lecun1989.pdf` ✅ noted |
| LeCun et al. / LeNet-5 | 1998 | ❌ | Proceedings of the IEEE / HAL mirror ✅ noted |
| Hochreiter-Schmidhuber | 1997 | ❌ | Neural Computation 9(8) / `hochreiter-schmidhuber-lstm-1997.pdf` ✅ noted |
| Gers-Schmidhuber-Cummins forget gate | 2000 | ❌ | IDSIA / Neural Computation PDF mirror ✅ noted |
| Sutskever-Vinyals-Le seq2seq | 2014 | ✅ | 1409.3215 ✅ noted |
| Luong-Pham-Manning attention | 2015 | ✅ | 1508.04025 ✅ noted |
| Hinton contrastive divergence | 2002 | ❌ | Neural Computation 14(8) ✅ noted |
| Hinton deep belief | 2006 | ❌ | Neural Computation 18(7) ✅ noted |
| Bengio et al. neural probabilistic language model | 2003 | ❌ | JMLR 3, 1137–1155 ✅ noted |
| Mikolov et al. recurrent neural language model | 2010 | ❌ | ISCA Archive / Interspeech 2010 ✅ noted |
| Glorot-Bengio initialization | 2010 | ❌ | PMLR proceedings PDF ✅ noted |
| Nair-Hinton ReLU | 2010 | ❌ | ICML author-hosted paper ✅ noted |
| He et al. rectifier initialization / PReLU | 2015 | ✅ | 1502.01852 ✅ noted |
| LeCun-Bengio-Hinton Deep Learning review | 2015 | ❌ | Nature / Hinton-hosted PDF ✅ noted |
| Batch Normalization | 2015 | ✅ | 1502.03167 ✅ noted |
| Layer Normalization | 2016 | ✅ | 1607.06450 ✅ noted |
| AlexNet | 2012 | ✅ | NeurIPS 2012 / ImageNet workshop PDF |
| GANs | 2014 | ✅ | 1406.2661 |
| Latent Diffusion Models | 2022 | ✅ | 2112.10752 ✅ noted |
| Attention | 2017 | ✅ | 1706.03762 |
| BERT | 2018 | ✅ | 1810.04805 |
| GPT-2 | 2019 | ✅ | OpenAI technical report PDF ✅ noted |
| MusicGen | 2023 | ✅ | 2306.05284 ✅ noted |
| Jordan & Jacobs HME | 1994 | ❌ | IJCN / Hinton-hosted PDF ✅ noted |
| PPO | 2017 | ✅ | 1707.06347 ✅ noted |
| ResNets | 2015 | ✅ | 1512.03385 ✅ upgraded |
| Cortes & Vapnik Support-Vector Networks | 1995 | ❌ | Springer / Machine Learning PDF ✅ noted |
| LLaMA | 2023 | ✅ | 2302.13971 ✅ noted |
| NVIDIA Nemotron 3 Super | 2026 | ✅ | 2604.12374 ✅ noted |
| Gemma 4 / Phi-4 / Qwen3 tradeoffs | 2026 | ✅ | 2604.07035 ✅ noted |

### Papers available on arXiv (immediate download queue)
- `1406.2661` — GANs (Goodfellow 2014) ✅ noted
- `1706.03762` — Attention Is All You Need (Vaswani 2017) ✅ noted
- `1810.04805` — BERT (Devlin 2018) ✅ noted
- `1512.03385` — Deep Residual Learning (He 2015) ✅ noted
- `1207.0580` — Dropout (Hinton 2012) ✅ noted
- `1502.03167` — Batch Normalization (Ioffe & Szegedy 2015) ✅ noted
- `1607.06450` — Layer Normalization (Ba, Kiros & Hinton 2016) ✅ noted
- `2203.02155` — InstructGPT RLHF (Ouyang 2022) ✅ noted
- `2212.08073` — Constitutional AI (Bai 2022) ✅ noted

### PDFs present but notes not yet created (next session queue)
- ~~`2005.14165` — GPT-3 / Brown et al. 2020~~ ✅ noted (March 10, 2026)
- ~~`1711.05101` — AdamW / Loshchilov & Hutter 2019~~ ✅ noted (March 10, 2026)
- ~~`1111.6088` — Buchmann survey on Quaternions / Hamilton 1843~~ ✅ noted (March 10, 2026) → Mathematics thread
- `math_0606400` — Kędra, Rudyak & Tralle (2006), symplectic manifolds / fundamental groups — **off-lineage** (differential topology; no note created; see Hamilton note for context)

### PDFs present, notes now created (April 11, 2026)
- ~~`2203.15556` — Hoffmann et al. Chinchilla 2022~~ ✅ noted
- ~~`2201.11903` — Wei et al. Chain-of-Thought 2022~~ ✅ noted
- ~~`2212.08073` — Bai et al. Constitutional AI 2022~~ ✅ noted
- ~~`deepseek-v4-2026.pdf` — DeepSeek-V4 2026~~ ✅ noted (May 17, 2026)

### Remaining unnoted PDFs
- None in the active AI/ML queue.

### Notes created (August 11, 2026)
- **Luong, Pham & Manning 2015 / Effective Approaches to Attention-based Neural Machine Translation** ✅ noted — closes the architectural-design gap between Bahdanau attention and the Transformer. Bahdanau proved learned retrieval could repair seq2seq’s fixed-vector bottleneck; Luong et al. separated attention into global vs. local scope, alternative compatibility scores, context construction, and input-fed alignment history. That turns attention from one successful NMT mechanism into a reusable layer while exposing the retrieval-quality vs. retrieval-cost tradeoff inherited by long-context systems. Source PDF: `corpus/raw/library-papers/1508.04025.pdf`

### Notes created (August 9, 2026)
- **Sutskever, Vinyals & Le 2014 / Sequence to Sequence Learning with Neural Networks** ✅ noted — closes the causal gap between LSTM as a memory mechanism and attention as a retrieval mechanism. Two deep LSTMs make variable-length conditional generation work end to end, but compressing every source detail into one fixed-dimensional vector creates the bottleneck that Bahdanau attention must repair. Source PDF: `corpus/raw/library-papers/1409.3215.pdf`

### Notes created (July 23, 2026)
- **Bengio, Simard & Frasconi 1994 / Learning Long-Term Dependencies with Gradient Descent Is Difficult** ✅ noted — closes the causal gap between backpropagation and LSTM. A recurrent network can represent robust memory with stable attractors, but the contraction that protects state from noise also drives derivatives of distant events toward zero. The result separates representational capacity from trainability and makes a purpose-built gradient-preserving memory path logically necessary. Source PDF: `corpus/raw/library-papers/bengio-long-term-dependencies-1994.pdf`

### Notes created (July 21, 2026)
- **Zoph et al. 2022 / ST-MoE: Designing Stable and Transferable Sparse Expert Models** ✅ noted — closes the stability-transfer gap after Switch Transformer and GLaM. Sparse MoE had become scalable and energy-efficient, but training instabilities and dense-style fine-tuning protocols could erase downstream gains. ST-MoE adds router z-loss, sparse-specific fine-tuning analysis, capacity-factor and precision tradeoffs, and a 269B sparse model that transfers across reasoning, summarization, closed-book QA, and adversarial tasks. Source PDF: `corpus/raw/library-papers/2202.08906.pdf`

### Notes upgraded (July 19, 2026)
- **Jordan & Jacobs 1994 / Hierarchical Mixtures of Experts and the EM Algorithm** ✅ sourced and wired into the MoE chain — closes the flat-specialization gap after Jacobs et al. 1991. A single soft gate could create local specialists, but nested tasks needed structured routing and principled credit assignment. HME turns MoE into a tree of soft gates and GLIM experts trained by maximum likelihood / EM, bridging early learned specialists to Shazeer-style sparse conditional computation. Source PDF: `corpus/raw/library-papers/jordan-jacobs-hierarchical-mixtures-experts-1994.pdf`

### Notes upgraded (July 18, 2026)
- **Minsky & Papert 1969 / Perceptrons** ✅ sourced and upgraded — closes the shallow-representation failure after Rosenblatt and Novikoff. Perceptron learning could converge when a separating solution existed, but local predicates plus one weighted vote could not represent global structure such as connectedness. That break made hidden representations and hidden-layer credit assignment logically necessary. Source PDF: `corpus/raw/library-papers/minsky-papert-perceptrons-1969.pdf`

### Notes upgraded (July 16, 2026)
- **DeepSeek-V4 2026 / Towards Highly Efficient Million-Token Context Intelligence** ✅ source-grounded and rechecked against the local PDF — closes the long-context efficiency gap after DeepSeekMoE, sparse attention, and DeepSeek-V3-style efficient MoE scaling. The note now records the two preview models (V4-Pro 1.6T total / 49B active; V4-Flash 284B / 13B active), the local source path, PDF verification, 1M-token CSA/HCA attention, mHC stabilization, Muon optimization, DeepSeekMoE continuity, FP4/FP8-aware training, and specialist-then-consolidate post-training. Source PDF: `corpus/raw/library-papers/deepseek-v4-2026.pdf`

### Notes upgraded (July 14, 2026)
- **Frank Rosenblatt 1958 / The Perceptron** ✅ upgraded — closes the learning-machine gap after McCulloch-Pitts and Hebb. Fixed threshold circuits could compute but not adapt; pure Hebbian association could strengthen co-activity but not fully explain supervised discrimination. Rosenblatt made weight adaptation from examples the bridge to convergence proofs, ADALINE's error minimization, Minsky-Papert limits, and hidden-layer credit assignment. Source PDF: `corpus/raw/library-papers/rosenblatt-perceptron-1958.pdf` (sourced Sep 6, 2026 via Wayback capture of the UPenn scan; 23pp full text verified — the earlier public "PDF" was lecture slides). DOI `10.1037/h0042519`, PMID `13602029`.

### Notes upgraded (July 12, 2026)
- **Rumelhart, Hinton & Williams 1986 / Learning representations by back-propagating errors** ✅ sourced and upgraded — closes the hidden-layer credit-assignment gap after perceptrons and Werbos. Single-layer learning could not discover intermediate representations; backpropagation turns the chain rule into a backward error signal for every weight, making multilayer representation learning trainable. Source PDF: `corpus/raw/library-papers/rumelhart1986.pdf`

### Notes upgraded (July 11, 2026)
- **McCulloch & Pitts 1943 / A Logical Calculus of Ideas Immanent in Nervous Activity** ✅ sourced and upgraded — closes the root computation gap before Hebb and Rosenblatt. Neural tissue becomes a formal threshold-logic network, but the wiring is fixed; that limitation makes the next learning steps necessary. Source PDF: `corpus/raw/library-papers/mcculloch-pitts-logical-calculus-1943.pdf`

### Notes created (July 9, 2026)
- **Manik & Wang 2026 / Gemma 4, Phi-4, and Qwen3 tradeoffs** ✅ noted — closes the deployment-evaluation gap after compact dense models, MoE efficiency work, DeepSeek-V4, and Nemotron 3 Super. Once many open reasoning models are plausible, the old leaderboard question breaks: model choice depends on accuracy, latency, VRAM, prompting regime, and task family together. Source PDF: `corpus/raw/library-papers/manik-wang-gemma4-phi4-qwen3-accuracy-efficiency-2026.pdf`

### Notes upgraded (July 7, 2026)
- **Ziegler et al. 2019 / Fine-Tuning Language Models from Human Preferences** ✅ sourced and wired into the RLHF chain — closes the language bridge between Christiano-style reward learning and Stiennon/Ouyang-style assistant post-training. GPT-style pretraining could produce fluent text, but fixed supervised labels and automatic metrics could not specify fuzzy human preferences like helpfulness, vividness, faithfulness, or usefulness. Source PDF: `corpus/raw/library-papers/1909.08593.pdf`

### Notes upgraded (July 5, 2026)
- **Kingma & Ba 2015 / Adam** ✅ upgraded — closes the usable-adaptivity gap between RMSProp and AdamW. AdaGrad remembered gradient scale forever; RMSProp learned recent scale; momentum carried direction. Adam combines first-moment direction, second-moment scale, and bias correction into the default optimizer grammar inherited by Transformer-era training. Source PDF: `corpus/raw/library-papers/1412.6980.pdf`

### Notes created (July 2, 2026)
- **Gers, Schmidhuber & Cummins 2000 / Learning to Forget with LSTM** ✅ noted — closes the practical memory-management gap after original LSTM. The Constant Error Carousel can preserve information across long delays, but continual streams with no external sequence reset can make cell states grow without bound. The forget gate replaces the fixed self-connection with a learned multiplicative reset, allowing LSTM to keep useful state, erase stale state, and self-segment continuous input. Source PDF: `corpus/raw/library-papers/gers-learning-to-forget-lstm-2000.pdf`

### Notes upgraded (June 30, 2026)
- **Schulman et al. 2017 / Proximal Policy Optimization** ✅ sourced and wired into the lineage — closes the mechanical gap inside the RLHF chain. Preference models can say which behavior humans prefer, but a language-model policy still needs a stable way to optimize against that learned reward without drifting into nonsense. PPO's clipped objective supplied the practical update rule used by early RLHF systems, and DPO later simplified the pipeline by removing this explicit RL stage. Source PDF: `corpus/raw/library-papers/1707.06347.pdf`

### Notes created (June 28, 2026)
- **Hinton, Srivastava & Swersky 2012 / RMSProp** ✅ noted — closes the stale-scale-memory gap between AdaGrad and Adam. AdaGrad's lifetime squared-gradient sum works well for sparse convex learning, but deep neural networks need adaptive learning rates that can forget old gradient scales. RMSProp uses an exponential moving average of squared gradients, making mini-batch adaptive optimization practical and supplying Adam's second-moment ancestor. Source PDF: `corpus/raw/library-papers/hinton-rmsprop-2012.pdf`

### Notes created (June 27, 2026)
- **Cortes & Vapnik 1995 / Support-Vector Networks** ✅ noted — closes the hard-margin SVM gap after Boser, Guyon & Vapnik 1992. The original optimal-margin classifier assumed separable training data; real labels overlap and contain noise. Soft margins add slack and a regularization parameter `C`, preserving maximum-margin geometry while allowing controlled violations. Source PDF: `corpus/raw/library-papers/cortes-vapnik-support-vector-networks-1995.pdf`

### Notes created (June 25, 2026)
- **NVIDIA Nemotron 3 Super 2026** ✅ noted — closes the agentic-throughput gap after Mamba-2, MoE scaling, and long-context frontier models. The old dense-Transformer recipe broke when assistants became long-horizon tool users: context, generation length, and interactive rollouts made serving cost part of capability. Nemotron 3 Super combines hybrid Mamba-Attention, LatentMoE, MTP, NVFP4 pretraining, and multi-environment RL into one open model recipe. Source PDF: `corpus/raw/library-papers/2604.12374.pdf`

### Notes created (June 23, 2026)
- **Touvron et al. 2023 / LLaMA** ✅ noted — closes the open-weight foundation-model gap after GPT-3 and Chinchilla. The old dense-scaling story broke at inference cost, reproducibility, closed data, and access. LLaMA kept the Transformer recipe but trained smaller models longer on public data, making frontier-adjacent capability portable enough for research, fine-tuning, and local deployment. Source PDF: `corpus/raw/library-papers/2302.13971.pdf`

### Notes created (June 21, 2026)
- **LeCun, Bottou, Bengio & Haffner 1998 / LeNet-5 and gradient-based document recognition** ✅ noted — closes the mature-CNN deployment gap between the 1989 zip-code CNN and AlexNet. The note explains why isolated character recognition was not enough: real document systems broke at feature engineering, segmentation, module-local tuning, and check-level correctness. LeNet-5 plus Graph Transformer Networks made more of the pipeline trainable. Source PDF: `corpus/raw/library-papers/lecun-document-recognition-1998.pdf`

### Notes created (June 20, 2026)
- **Yann LeCun, Yoshua Bengio & Geoffrey Hinton 2015 / Deep Learning review** ✅ noted — closes the synthesis gap after AlexNet and DBN-era revival. The field could no longer treat neural networks as scattered tricks; the shared story was hierarchical representation learning trained by backpropagation, replacing brittle hand-engineered features across vision, speech, language, and scientific data. Source PDF: `corpus/raw/library-papers/lecun-bengio-hinton-deep-learning-2015.pdf`

### Notes created (June 18, 2026)
- **Sergey Ioffe & Christian Szegedy 2015 / Batch Normalization** ✅ noted — closes the trainability-stability gap after AlexNet. ReLU and GPUs made deep CNNs viable, but deeper networks still required careful learning rates and initialization because internal activation distributions shifted during training. BatchNorm made normalization differentiable inside the graph, enabling much larger learning rates, faster convergence, regularization from mini-batch statistics, and the normalization habit later inherited by ResNets and Transformers. Source PDF: `corpus/raw/library-papers/1502.03167.pdf`

### Notes created (June 13, 2026)
- **Augustin-Louis Cauchy 1847 / Steepest Descent** ✅ noted — closes the root gap in the optimizer lineage. When elimination becomes impossible or explosively complicated, combine residuals into one nonnegative objective and repeatedly step opposite its partial derivatives. This creates the deterministic template later repaired for noise, ill-conditioning, coordinate imbalance, and scale by Robbins-Monro, momentum, AdaGrad, Adam, and AdamW. Source PDF: `corpus/raw/library-papers/cauchy-steepest-descent-1847.pdf`

### Notes created (June 11, 2026)
- **Nair & Hinton 2010 / Rectified Linear Units** ✅ noted — closes the trainability gap between DBN-era greedy pretraining and AlexNet's end-to-end supervised recipe. Rectifiers preserve positive-side gradients and activation magnitude, and the paper showed that changing the hidden unit could outperform binary units even without pretraining. Source PDF: `corpus/raw/library-papers/nair-relu-2010.pdf`

### Notes created (June 9, 2026)
- **Paul Werbos 1974 / Beyond Regression** ✅ noted — closes the hidden-variable credit-assignment gap between single-layer LMS and multilayer backpropagation. Ordered derivatives reuse sensitivities backward through a composed computation, anticipating reverse-mode automatic differentiation while leaving the later neural-network presentation and empirical adoption to Rumelhart, Hinton & Williams. Source PDF: `corpus/raw/library-papers/werbos-beyond-regression-1974.pdf`

### Notes upgraded (June 2, 2026)
- **Widrow & Hoff 1960 / ADALINE and LMS** ✅ noted — upgrades the seed note into a sourced lineage note. Closes the gap between perceptron-style mistake correction and explicit squared-error minimization: learning becomes one example, one error signal, one gradient-like weight update. Source PDF: `corpus/raw/library-papers/widrow-hoff-adaptive-switching-1960.pdf`

### Notes created (June 7, 2026)
- **Radford et al. 2018 / GPT-1** ✅ noted — closes the break between the Transformer as a supervised sequence architecture and GPT-2 as a zero-shot language model: generative pretraining learns a reusable language representation, while task-specific fine-tuning exposes the remaining adaptation bottleneck. Source PDF: `corpus/raw/library-papers/radford-generative-pretraining-2018.pdf`

### Notes created (June 6, 2026)
- **Boris Polyak 1964 / Heavy-Ball Momentum** ✅ noted — closes the optimizer-memory gap between one-step gradient descent and Adam. By carrying the previous displacement, the method accelerates ill-conditioned optimization from order κ to order √κ iterations and establishes momentum as a stateful first-order method. Source PDF: `corpus/raw/library-papers/polyak-heavy-ball-1964.pdf`

### Notes created (June 4, 2026)
- **Claude Shannon 1948 / A Mathematical Theory of Communication** ✅ noted — adds the information-theoretic measurement layer beneath language modeling: entropy, bits, source modeling, compression, and the reason cross-entropy / next-token prediction is a coherent objective. Source PDF: `corpus/raw/library-papers/shannon-information-theory-1948.pdf`

### Notes created (May 31, 2026)
- **A. B. J. Novikoff 1962 / Perceptron Convergence** ✅ noted — closes the proof gap after Rosenblatt: if the data are bounded and linearly separable with margin, mistake-driven perceptron learning converges in finitely many errors. Source PDF: `corpus/raw/library-papers/novikoff-convergence-perceptrons-1962.pdf`

### Notes created (May 30, 2026)
- **Duchi, Hazan & Singer 2011 / AdaGrad** ✅ noted — closes the gap between plain SGD's single global learning rate and sparse, high-dimensional data where rare predictive features need larger coordinate-wise steps. Source PDF: `corpus/raw/library-papers/duchi-adagrad-2011.pdf`

### Notes created (May 28, 2026)
- **Robbins & Monro 1951 / Stochastic Approximation** ✅ noted — closes the gap between deterministic steepest descent and machine-learning-style noisy updates: the target function is an expectation, but each step only sees a random sample. Source PDF: `corpus/raw/library-papers/robbins-monro-stochastic-approximation-1951.pdf`

### Notes created (May 26, 2026)
- **Loshchilov & Hutter 2017 / SGDR** ✅ noted — cosine annealing with warm restarts closes the practical gap between plain SGD and modern schedule-aware training recipes. Source PDF: `corpus/raw/library-papers/1608.03983.pdf`

### Notes created (April 25, 2026)
- AlexNet / Krizhevsky, Sutskever & Hinton 2012 ✅ noted

### Notes created (May 24, 2026)
- **Hinton 2002 / Contrastive Divergence** ✅ noted — one-step data-vs-reconstruction learning makes RBMs practical, closing the technical gap between full Boltzmann machines and Deep Belief Nets. Source PDF: `corpus/raw/library-papers/hinton-contrastive-divergence-2002.pdf`

### Notes created (May 23, 2026)
- **Hinton, Osindero & Teh 2006 / Deep Belief Nets** ✅ noted — greedy RBM pretraining closes the practical gap between backprop's trainable hidden layers and the deep-learning revival. Source PDF: `corpus/raw/library-papers/hinton2006-dbn.pdf`

### Notes created (May 21, 2026)
- **GPT-2 2019** ✅ noted — the hinge from Transformer/GPT-style pretraining to GPT-3-style in-context learning. Source PDF: `corpus/raw/library-papers/radford-gpt2-2019.pdf`

### Notes created (May 19, 2026)
- **Hebb 1949** ✅ noted — the missing learning-rule bridge from McCulloch-Pitts logical neurons to Rosenblatt's learned perceptron. Source PDF: `corpus/raw/library-papers/hebb-organization-behavior-1949.pdf`

### Pre-arXiv papers: alternative sources
- **Semantic Scholar** — free PDFs for many classic papers
- **Google Scholar** — direct PDF links
- **Author websites** — many researchers host their papers
- **Sci-Hub** — as last resort
- **Project Gutenberg / Internet Archive** — historical texts

---

## Mathematics Thread

```
Zeno ~450 BCE (paradoxes of infinity)
├── Eudoxus ~370 BCE (proportion for incommensurables; exhaustion) ✅ noted
│   └── Euclid ~300 BCE (Elements — definitions, postulates, proof as a system) ✅ noted
│       └── Archimedes ~250 BCE (exhaustion, mechanical discovery, parabola quadrature) ✅ noted
│           └── "The Method" — mechanical discovery -> formal proof
│
Al-Khwarizmi ~820 (algebra as sentences)
├── Viète 1591 (symbolic notation — letters for general quantities) ✅ noted
│   └── Descartes 1637 (analytic geometry — algebra + geometry)
│       ├── Fermat ~1630s (adequality, tangent lines, maxima/minima — early derivatives) ✅ noted
│       │   └── General integral power rule
│       ├── Hamilton 1843 (quaternions — first non-commutative algebra) ✅ noted
│       │   ├── Frobenius 1877 (only ℝ, ℂ, ℍ as real division algebras)
│       │   ├── Clifford 1878 (geometric algebra — generalization)
│       │   └── Modern: 3D rotation, computer graphics, robotics
│       └── Grassmann 1844 (Ausdehnungslehre — vector spaces, independence, dimension) ✅ noted
│           └── Cayley 1858 (matrices as algebraic objects / transformations) ✅ noted
│               └── Cauchy spectral theorem
│                   └── Jacobi 1846 (plane rotations compute symmetric eigenvalues/eigenvectors) ✅ noted
│                       ├── Schmidt 1907 (paired singular modes; best finite-rank kernel approximation) ✅ noted
│                       │   └── Eckart-Young 1936 (finite-matrix best low-rank approximation) ✅ noted
│                       └── Pearson 1901 (principal axes of a data cloud) ✅ noted
│                           └── Hotelling 1933 (principal components as ordered uncorrelated variables) ✅ noted
│                               └── Eckart-Young 1936 (best low-rank approximation) ✅ noted
│                                   └── Golub-Kahan 1965 (stable numerical SVD + pseudoinverse) ✅ noted
│                                       └── numerical SVD / low-rank compression / latent factor models
│
├── Baghdad scholars (volumes of revolution — early integration!)
│
├── Aryabhata 500s (d/dx sin = cos — Indian calculus!)
│
├── Cavalieri 1635 (indivisibles — continuous figures compared by slices) ✅ noted
│   └── Barrow 1670 (geometrical lectures — tangents and areas as inverse problems) ✅ noted
│       └── Torricelli (Gabriel's Horn — infinite surface, finite volume)
│
├── Galileo ~1600 (falling bodies — integration ↔ differentiation hint)
├── Kepler ~1610 (equal area law — same hint from planetary orbits)
│
├── Barrow 1670 (geometric tangent-area reciprocity — pre-FTC hinge) ✅ noted
│   └── NEWTON ~1666–1671 (fluents/fluxions) + LEIBNIZ 1675–1686 (d, dy/dx, ∫) ✅ source-grounded
│       └── FUNDAMENTAL THEOREM (direct/inverse problems; local rate ↔ global accumulation) ✅ rebuilt
│           └── Leibniz 1676 (differential substitution / chain rule — local changes compose through nested dependencies) ✅ noted
│               └── Jacobians / automatic differentiation / backpropagation
    │
    Euler 1744 (calculus of variations — optimize curves/functions) ✅ noted
    │   └── Lagrange 1788 (analytical mechanics, virtual work, multipliers — constraints stay in the equations) ✅ noted
    │       └── Cauchy 1847 (steepest descent — derivatives become an iterative downhill algorithm) ✅ noted
    │           └── Robbins-Monro 1951 (stochastic approximation — noisy sample-based updates) ✅ noted
    │               └── SGD / backprop / Adam
    │
    │       Hamilton/Jacobi → optimal control / variational inference
    │
    Boole 1854 (reasoning becomes algebraic; logical classes obey x²=x over 0/1) ✅ noted
    └── later Boolean/propositional logic
        └── Turing 1936 (mechanical procedure, encoded programs, universal simulation, undecidability) ✅ noted
            └── Turing computability + all-or-none neurophysiology
                └── McCulloch-Pitts 1943 (binary threshold networks realize temporal logic) ✅ noted
                    └── Hebb 1949 (learning as weight change — co-activity writes memory into parameters) ✅ noted
                        └── Rosenblatt 1958 (probabilistic perceptron — reinforcement trains discrimination and generalization) ✅ noted
                            ├── Block / Novikoff 1962 (separability → finite-mistake convergence) ✅ noted
                            ├── Widrow-Hoff 1960 (pre-threshold residual + squared loss → LMS/online gradient learning) ✅ noted
                            │   └── delta rule / adaptive filtering / differentiable loss
                            │       └── backpropagation extends graded credit through hidden layers
                            └── Minsky-Papert 1969 (restricted representations expose limits)
    │
    Euler 1748 (functions become the object; infinite series become representations) ✅ source-grounded
    └── Fourier 1807–1822 (arbitrary functions as trigonometric modes; solve heat by superposition) ✅ noted
        ├── spectral representation / orthogonal bases / signal analysis
        └── convergence crisis: when does an infinite representation equal the function?
            ├── Cauchy 1821 (limits, continuity, series convergence, and the tail criterion become explicit) ✅ source-grounded
            │   └── pointwise convergence is not enough to preserve continuity; completeness remains implicit
            └── Dirichlet 1829 (piecewise-regular Fourier convergence; midpoint at jumps) ✅ noted
                ├── function separates from analytic formula
                └── Riemann 1854 (tagged-sum integral; discontinuity controlled by local oscillation) ✅ noted
                    ├── Weierstrass 1861 (ε–δ dependency makes quantified error control explicit) ✅ source-grounded
                    │   └── pointwise vs uniform control; completeness of the continuum becomes unavoidable
                    └── interval-based integration still fails on persistent dense oscillation
        │
        Dedekind 1872 (cuts — arithmetic construction of the real continuum) ✅ noted
        └── Cantor 1874 (countability vs continuum — infinity has sizes) ✅ noted
            ├── DEBATES CONTINUE (constructivism, finitism, non-standard analysis...)
            └── Lebesgue 1902 (measurable exceptional sets + integration stable under limits) ✅ noted
                ├── Lᵖ spaces / almost-everywhere analysis / functional analysis
                └── Kolmogorov 1933 (probability as normalized measure) ✅ noted

Kolmogorov 1933 (probability as measure, built on Lebesgue integration) ✅ noted
├── Shannon 1948 (information theory — entropy, bits, source modeling) ✅ noted
│   └── Kullback-Leibler 1951 (expected log-ratio; information lost by processing) ✅ noted
│       ├── cross-entropy = entropy + distribution-mismatch cost
│       ├── data-processing inequality / sufficient representations
│       └── probabilistic language modeling / variational inference / natural gradient
└── rigorous probability for statistical models

Gauss 1809 (Gaussian errors + least squares) ✅ noted
└── Fisher 1922 (likelihood, MLE, sufficiency, Fisher information) ✅ noted
    ├── negative log-likelihood as a general optimizable loss
    ├── Gaussian / Bernoulli / categorical NLL → MSE / binary CE / softmax CE
    └── Fisher 1922 + Shannon 1948
        └── Kullback-Leibler 1951 → cross-entropy training and information geometry ✅ noted
```

### Notes created (September 23, 2026)

- **Bernard Widrow & Marcian Hoff 1960 / Adaptive Switching Circuits** ✅ noted in the Mathematics learning-theory thread — closes the graded-objective gap after Rosenblatt and beside Novikoff. A threshold learner could correct mistakes, and separable data could support a finite convergence guarantee, but noisy or contradictory data need a best compromise while a binary decision discards error magnitude. ADALINE measures the desired-minus-linear-response residual before quantization, making expected squared error a convex quadratic surface; the one-pattern derivative $-2ex$ yields the LMS update $w\leftarrow w+\mu ex$. The note preserves the paper’s measured-error versus neuron-error distinction, connects its adaptation time and misadjustment analysis to stochastic gradients, and marks the remaining representation/hidden-credit problem that leads toward backpropagation. Source PDF: `corpus/raw/library-papers/widrow-hoff-adaptive-switching-1960.pdf` (nine-page Stanford-hosted primary-source scan; verified PDF 1.3).

### Notes created (September 21, 2026)

- **A. B. J. Novikoff 1962 / On Convergence Proofs on Perceptrons** ✅ noted in the Mathematics learning-theory thread — closes the proof gap after Rosenblatt. An adaptive threshold classifier existed, but it was unknown whether mistake-driven corrections had to terminate even when a correct separator existed. Novikoff's argument pairs linear progress toward a positive-margin separator with square-root growth of the weight norm, yielding the finite mistake bound $M\le(R/\gamma)^2$ for bounded separable examples. The note explicitly separates training convergence from maximum-margin selection and statistical generalization, then traces the failed-assumption case toward graded losses, representational critiques, kernels, and hidden-layer feature learning. Source PDF: `corpus/raw/library-papers/novikoff-convergence-perceptrons-1962.pdf` (eight-page primary-source scan; verified PDF 1.6).

### Notes created (September 18, 2026)

- **Frank Rosenblatt 1958 / The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain** ✅ noted in the Mathematics learning-theory thread — closes the gap between Hebbian plasticity and supervised classification. Hebb made a connection capable of storing co-activity, but did not specify a complete discrimination task, correction mechanism, or generalization test. Rosenblatt combines randomly connected association features, adjustable values, competing responses, and positive/negative reinforcement; he explicitly distinguishes correct recall from correct classification of unseen examples and makes statistical class separability in the representation the condition for performance above chance. The note preserves the important boundary between Rosenblatt's broad 1958 probabilistic architecture, the later textbook perceptron update, and Block/Novikoff's convergence theorem. Source PDF: `corpus/raw/library-papers/rosenblatt-perceptron-1958.pdf` (23-page original article; verified PDF 1.5).

### Notes created (September 16, 2026)

- **Alan M. Turing 1936 / On Computable Numbers, with an Application to the Entscheidungsproblem** ✅ noted in the Mathematics learning-theory thread — closes the explicit symbolic-logic-to-neural-computation gap. Boole and later formal logic made reasoning transformable by exact rules, but the phrase “mechanical procedure” remained informal. Turing models a calculator as finite-state local action over symbolic tape, encodes each machine as data, constructs one universal machine that simulates any encoded machine, and diagonalizes against a hypothetical procedure that classifies every machine's productive behavior. He then reduces the *Entscheidungsproblem* to an impossible machine-behavior test. This supplies the target notion of computation that McCulloch–Pitts can realize with threshold networks, while distinguishing universality from learning, efficiency, and intelligence. Source PDF: `corpus/raw/library-papers/turing-computable-numbers-1936.pdf` (36-page original article scan; verified PDF 1.5).

### Notes created (September 14, 2026)

- **George Boole 1854 / An Investigation of the Laws of Thought** ✅ noted in the Mathematics learning-theory thread — closes the unnamed propositional-logic gap before McCulloch–Pitts. Traditional logic catalogued verbal syllogisms but lacked a general algorithm for complex combinations of classes and exclusions. Boole represents classes by symbols, interprets multiplication as simultaneous membership and `1-x` as complement, and identifies the idempotent law $x^2=x$, connecting logical membership to the binary values 0 and 1. Expansion and elimination make inference a symbolic procedure. The note distinguishes Boole's original partial class algebra from modern Boolean algebra, then traces the repaired chain through switching theory to threshold-neuron circuits and learned weights. Source PDF: `corpus/raw/library-papers/boole-laws-of-thought-1854.pdf` (complete 343-page Project Gutenberg transcription; verified PDF 1.5)

### Notes created (September 11, 2026)

- **Solomon Kullback & Richard Leibler 1951 / On Information and Sufficiency** ✅ noted in the Mathematics information-theory thread — closes the explicit Fisher/Shannon-to-modern-loss gap. Shannon measured uncertainty within one source and Fisher made sample probability an estimation objective, but neither supplied a population-level price for modeling data from $P$ as though they came from $Q$. Kullback and Leibler define the expected log-likelihood ratio, prove that it is nonnegative and additive, and show that statistical processing cannot increase discriminatory information, with equality exactly for a sufficient statistic. The modern identity $H(P,Q)=H(P)+D_{KL}(P\|Q)$ then makes cross-entropy minimization equivalent to reducing distribution mismatch, while the local quadratic expansion recovers Fisher information geometry. Source PDF: `corpus/raw/library-papers/kullback-leibler-information-sufficiency-1951.pdf` (8 pages; extracted from the verified Internet Archive issue scan, journal pp. 79–86)

### Notes created (September 9, 2026)

- **R. A. Fisher 1922 / On the Mathematical Foundations of Theoretical Statistics** ✅ noted in the Mathematics probability thread — closes the probability-to-learning-objective gap after Gauss. Least squares connected Gaussian errors to one fitting rule, but statistics lacked a general principle for choosing parameters in arbitrary probability models. Fisher fixes the observed sample, treats its probability as likelihood over candidate parameters, and maximizes the additive log likelihood; consistency, efficiency, sufficiency, and likelihood curvature then measure what an estimator retains and how precisely it can distinguish nearby models. Gaussian, Bernoulli, categorical, and autoregressive likelihoods become MSE, binary cross-entropy, softmax cross-entropy, and next-token negative log-likelihood. The note also preserves Fisher's explicit caveat that his proposed universal link from MLE to sufficiency lacked a complete proof. Source PDF: `corpus/raw/library-papers/fisher-theoretical-statistics-1922.pdf` (61 pages; assembled from the repository's two verified scan parts)

### Notes created (September 4, 2026)

- **Warren McCulloch & Walter Pitts 1943 / Neural Networks as Logical Calculus** ✅ noted in the Mathematics learning-theory thread — closes the fixed-computation gap immediately before Hebb. Formal logic could specify propositions and Turing could bound mechanical computation, but neither showed how a network of simple physical units could realize those operations. McCulloch and Pitts model firing as a binary proposition, construct delayed logical operations from threshold-and-inhibition circuits, prove a two-way correspondence between acyclic nets and temporal propositional expressions, and show how cycles introduce state. The model still lacks a data-driven learning rule, making Hebbian plasticity logically necessary. Source PDF: `corpus/raw/library-papers/mcculloch-pitts-logical-calculus-1943.pdf`

### Notes created (September 2, 2026)

- **Gottfried Wilhelm Leibniz 1676 / Differential Substitution and the Chain Rule** ✅ noted in the Mathematics calculus thread — closes the Newton–Leibniz → backpropagation composition gap. A general derivative operator was still insufficient for roots, quotients, implicit relations, and other nested dependencies; Leibniz’s manuscripts introduce an intermediate letter, find the component differentials separately, and substitute them. The note distinguishes this operational discovery from the later modern theorem, interprets the chain rule as composition of local linear maps and Jacobians, and follows it into reverse-mode automatic differentiation, backpropagation, and gradient pathologies. Source PDF: `corpus/raw/library-papers/child-leibniz-chain-rule-1917.pdf`

### Notes upgraded (August 28, 2026)

- **Karl Weierstrass 1861 / The ε–δ Method and Arithmetization of Analysis** ✅ source-grounded and rebuilt — closes the Cauchy/Dirichlet/Riemann → complete-continuum hinge. Nineteenth-century analysis could define limits and manipulate convergent processes, but it still needed a reusable grammar stating which neighborhood must answer an arbitrarily chosen error tolerance and whether that choice could depend on the point. The ε–δ method makes this dependency explicit; quantifier order then separates pointwise from uniform control. The note documents that the earliest known formulation survives in H. A. Schwarz’s notes from Weierstrass’s 1861 lectures, not in a Weierstrass-authored 1861 paper, and traces the remaining completeness problem into Dedekind, Cantor, Lebesgue, probability, and AI convergence/generalization theory. Source PDF: `corpus/raw/library-papers/1502.06942.pdf`

### Notes upgraded (August 26, 2026)

- **Augustin-Louis Cauchy 1821 / Limits and Convergence** ✅ source-grounded and rebuilt — closes the Euler/Fourier → rigorous-analysis hinge. Infinite series had become indispensable representations, but algebraic manipulation did not say whether finite partial sums approached a value or whether operations survived the limit. Cauchy makes limits, null infinitesimals, continuity, series convergence, and the tail criterion explicit; his harmonic-series example separates vanishing terms from convergent accumulation. His false theorem on continuous-function series then exposes pointwise versus uniform convergence, while the tail criterion’s tacit reliance on complete real numbers makes Dedekind’s later construction necessary. The note now cleanly separates the 1821 analysis work from Cauchy’s 1847 steepest descent. Source PDF: `corpus/raw/library-papers/cauchy-cours-analyse-1821.pdf`

### Notes upgraded (August 24, 2026)

- **Leonhard Euler 1748 / Functions and Infinite Series** ✅ source-grounded and rebuilt — closes the Newton–Leibniz → Fourier organization gap. Calculus had general methods for rates and accumulation but lacked a common object around which analysis could be organized; Euler makes functions central and uses infinite series, products, and related expansions to represent and compute transcendental dependence. That success exposes the unresolved distinction between a function and its formula, and between a formal expansion and a convergent representation—the exact problems that force Fourier, Cauchy, Dirichlet, Riemann, Weierstrass, Cantor, and Lebesgue. Source PDF: `corpus/raw/library-papers/euler-introductio-analysis-infinite-1748.pdf`

### Notes upgraded (August 21, 2026)

- **Isaac Newton & Gottfried Wilhelm Leibniz 1666–1686 / A General Calculus and the Fundamental Theorem** ✅ source-grounded and rebuilt — closes the central Barrow → Euler gap. Seventeenth-century mathematics had methods for particular tangents, velocities, areas, and sums but lacked a general system that made rate and accumulation inverse problems. Newton’s fluxions pair the direct problem “given a fluent, find its rate” with the inverse problem “given the rate, recover the fluent”; Leibniz’s `d` and `∫` notation turns that reciprocity into a portable operational language. The Fundamental Theorem converts accumulated local change into endpoint difference, creating the calculus later expanded by Euler and made rigorous by Cauchy, Riemann, and Weierstrass. The same local-to-global grammar underlies gradients, backpropagation, automatic differentiation, and expected-loss optimization. Source PDFs: `corpus/raw/library-papers/newton-method-of-fluxions-1736.pdf` and `corpus/raw/library-papers/leibniz-early-mathematical-manuscripts-1920.pdf`

### Notes created (August 19, 2026)

- **Bernhard Riemann 1854 / Integration and Trigonometric Series** ✅ noted in the Mathematics analysis thread — closes the Dirichlet → Lebesgue integration gap. Dirichlet’s Fourier theorem used integrals for piecewise-regular functions, but analysis still lacked a general test for which discontinuous functions possessed those coefficients. Riemann defines the integral by requiring every sufficiently fine tagged partition sum to approach one value and characterizes integrability through vanishing weighted local oscillation. The method admits many functions with infinitely many discontinuities but fails when dense oscillation persists across the domain, making Lebesgue’s later move from intervals to measurable exceptional sets logically necessary. Source PDF: `corpus/raw/library-papers/riemann-trigonometric-series-integration-1854.pdf`

### Notes created (August 17, 2026)

- **P. G. Lejeune Dirichlet 1829 / Fourier-Series Convergence** ✅ noted in the Mathematics analysis thread — closes the immediate truth-condition gap after Fourier. Fourier could calculate coefficients and solve heat mode by mode, but neither that success nor Cauchy's attempted proof established when the undamped series converged or what it reconstructed at a discontinuity. Dirichlet rewrites the partial sum as an oscillatory-kernel integral and uses cancellation to prove pointwise convergence for bounded piecewise monotone functions, with the limit equal to the average of the left and right values. This makes assumptions and modes of convergence part of the representation itself, opening the route to Riemann, Weierstrass, Cantor, Lebesgue, and modern harmonic analysis. Source PDF: `corpus/raw/library-papers/0806.1294.pdf`

### Notes created (August 14, 2026)

- **Joseph Fourier 1807–1822 / Heat Equation and Trigonometric Representation** ✅ noted in the Mathematics analysis thread — closes the Euler → Cauchy/Weierstrass representation gap. Euler made functions and infinite series central, but Fourier faced the harder physical problem of solving one PDE for arbitrary measured initial profiles. He decomposed those profiles into orthogonal sine/cosine modes, used linear superposition to evolve each mode independently, and recovered the solution by recombination. His extension to irregular and discontinuous functions made convergence and the meaning of infinite representation unavoidable, forcing the later rigor of Cauchy, Dirichlet, Weierstrass, Riemann, and Lebesgue. Source PDF: `corpus/raw/library-papers/fourier-analytical-theory-heat-1822.pdf`

### Notes created (August 12, 2026)

- **Erhard Schmidt 1907 / Best Finite-Rank Approximation of Kernels** ✅ noted in the Mathematics linear-algebra thread — closes the missing Cauchy spectral theorem → Eckart-Young hinge. Symmetric operators can use one orthogonal eigenbasis, but general two-sided kernels need paired left/right systems; Schmidt proves that the first `m` adjoint pairs give the best separable approximation under integrated squared error. This is the infinite-dimensional ancestor of truncated SVD, later made a finite-matrix theorem by Eckart-Young and a stable algorithm by Golub-Kahan. Source PDF: `corpus/raw/library-papers/schmidt-integral-equations-1907.pdf`

### Notes created (August 10, 2026)

- **Henri Lebesgue 1902 / Measure and Integration** ✅ noted in the Mathematics set-theory-to-probability thread — closes the missing Cantor/Dedekind → Kolmogorov hinge. Set theory made subsets of the continuum legitimate objects, but measure-theoretic probability still needed countably additive size, measurable functions, and an integral stable under limiting operations. Lebesgue supplies that machinery; Kolmogorov later normalizes it to total mass one, and Shannon turns the resulting distributions into entropy and cross-entropy. Source PDF: `corpus/raw/library-papers/lebesgue-integrale-longueur-aire-1902.pdf`

### Notes created (July 22, 2026)

- **Eudoxus ~370 BCE / Proportion Without Irrational Numbers** ✅ noted in the Mathematics ancient-Greek thread — closes the technical gap between Zeno's challenge and Euclid's proof system. Incommensurable magnitudes cannot be reduced to whole-number ratios, so Eudoxian proportion defines ratio equality through the ordering behavior of every pair of integer multiples. No Eudoxus-authored text survives; the earliest extant presentation is Euclid's Book V. Source PDF: `corpus/raw/library-papers/eudoxus-proportion-book-v-300bce.pdf`

### Notes upgraded (July 20, 2026)

- **Archimedes ~250 BCE / Exhaustion and Mechanical Discovery** ✅ upgraded in the Mathematics ancient-Greek thread — closes the continuous-magnitude bridge after Euclid. Euclidean proof culture gave geometry a durable grammar; Archimedes used exhaustion to trap curved areas and volumes rigorously, while *The Method* reveals the mechanical slice-and-balance heuristics that helped him discover results before proving them. Source PDF: `corpus/raw/library-papers/archimedes-method-250bce.pdf`

### Notes created (July 17, 2026)

- **Euclid ~300 BCE / Elements and Axiomatic Geometry** ✅ noted in the Mathematics ancient-Greek thread — closes the weak opening before Archimedes. Zeno exposes the danger of infinity and infinite division; Eudoxus develops proportion and exhaustion; Euclid codifies definitions, postulates, common notions, constructions, and proofs into reusable mathematical infrastructure. Source PDF: `corpus/raw/library-papers/euclid-elements-300bce.pdf`

### Notes created (July 15, 2026)

- **Donald Hebb 1949 / Learning as Weight Change** ✅ noted in the Mathematics learning-theory thread — closes the bridge from fixed computation and correlation structure to learned parameters. McCulloch-Pitts makes neural circuits compute; Hebb makes connection strength plastic through repeated co-activity; later perceptrons, LMS, backpropagation, and SGD make that plasticity supervised, error-driven, and scalable. Source PDF: `corpus/raw/library-papers/hebb-organization-behavior-1949.pdf`

### Notes created (July 13, 2026)

- **Jacobi 1846 / Eigenvalue Algorithm by Plane Rotations** ✅ noted in the Mathematics linear-algebra thread — closes the spectral theorem-to-computation gap after Cauchy. Cauchy makes symmetric matrices' hidden axes mathematically legitimate; Jacobi gives a geometry-preserving iterative method for finding them numerically by rotating away off-diagonal coupling. Source PDF: `corpus/raw/library-papers/jacobi-eigenvalue-algorithm-1846.pdf`

### Notes created (July 10, 2026)

- **Robbins & Monro 1951 / Stochastic Approximation** ✅ noted in the Mathematics optimization thread — closes the noisy-update gap after Cauchy's steepest descent. Cauchy makes descent algorithmic when the objective is known; Robbins-Monro makes approximation statistical when the target is an expectation and every observation is noisy. Source PDF: `corpus/raw/library-papers/robbins-monro-stochastic-approximation-1951.pdf`

### Notes created (July 8, 2026)
- **Barrow 1670** ✅ noted — closes the too-large jump from Cavalieri/Fermat to Newton-Leibniz. Cavalieri made accumulated area operational; Fermat made tangency and local change operational; Barrow's geometrical lectures make their reciprocity visible before symbolic calculus turns it into the fundamental theorem. Source PDF: `corpus/raw/library-papers/barrow-geometrical-lectures-1670.pdf`

### Notes created (July 6, 2026)
- **Cauchy 1847** ✅ noted — closes the algorithmic optimization gap after Euler and Lagrange. Calculus and constraints explain what an optimum means; Cauchy turns derivative information into an iterative process: define scalar wrongness, follow a downhill direction, choose a step, and repeat. Source PDF: `corpus/raw/library-papers/cauchy-steepest-descent-1847.pdf`

### Notes created (June 29, 2026)
- **Lagrange 1788** ✅ noted — closes the constraint gap after Euler. Euler makes optimization range over curves/functions; Lagrange keeps constraints inside the analytic machinery through virtual displacements and multipliers, giving modern constrained optimization one of its durable patterns. Source PDF: `corpus/raw/library-papers/lagrange-analytical-mechanics-1788.pdf`

### Notes created (June 26, 2026)
- **Euler 1744** ✅ noted — closes the optimization-width gap between early extrema and modern optimization. Fermat shows how to optimize quantities; Newton-Leibniz supplies calculus; Euler makes optimization range over whole curves/functions and turns variational problems into differential equations. Source PDF: `corpus/raw/library-papers/euler-calculus-of-variations-1744.pdf`

### Notes created (June 22, 2026)
- **Golub-Kahan 1965** ✅ noted — closes the theorem-to-computation gap after Eckart-Young. Low-rank approximation was optimal on paper, but practical AI and statistics need stable floating-point SVD and pseudoinverse algorithms for rectangular, noisy, rank-deficient matrices. Source PDF: `corpus/raw/library-papers/golub-kahan-svd-pseudoinverse-1965.pdf`

### Notes created (June 19, 2026)
- **Eckart-Young 1936** ✅ noted — closes the Hotelling-to-SVD gap in the multivariate/statistical linear-algebra thread. Pearson gives best-fit subspaces and Hotelling gives ordered principal components; Eckart and Young prove that keeping the largest canonical components gives the best rank-constrained reconstruction under squared-error matrix distance. Source PDF: `corpus/raw/library-papers/eckart-young-low-rank-1936.pdf`

### Notes created (June 5, 2026)
- **Shannon 1948** ✅ noted — closes the mathematics-side bridge from rigorous probability to information-theoretic prediction: probability distributions become entropy budgets, messages become source outputs, and cross-entropy becomes a meaningful objective for language modeling. Source PDF: `corpus/raw/library-papers/shannon-information-theory-1948.pdf`

### Notes created (June 3, 2026)
- **Hotelling 1933** ✅ noted — closes the Pearson-to-modern-PCA gap: Pearson gives best-fit subspace geometry, but Hotelling turns correlated variables into ordered, uncorrelated principal components and makes dimensionality reduction a general multivariate method. Source PDF: `corpus/raw/library-papers/hotelling-principal-components-1933.pdf`

### Notes created (May 29, 2026)
- **Grassmann 1844** ✅ noted — closes the vector-space gap in the linear-algebra thread: coordinates/quaternions/matrices are not enough unless directed quantities, independence, dimension, and spaces themselves become algebraic. Source PDF: `corpus/raw/library-papers/grassmann-ausdehnungslehre-1844.pdf`

### Notes created (May 27, 2026)
- **Dedekind 1872** ✅ noted — cuts close the post-Weierstrass foundation gap from the other side: rigorous limits quantify over real numbers, so the continuum needs an arithmetic construction before Cantor can measure its size. Source PDF: `corpus/raw/library-papers/dedekind-continuity-irrational-numbers-1872.pdf`

### Notes created (May 25, 2026)
- **Cantor 1874** ✅ noted — countability vs continuum closes the post-Weierstrass foundation gap: rigorous limits made the real line unavoidable, and Cantor showed that dense/countable infinity is not the same as the continuum. Source PDF: `corpus/raw/library-papers/cantor-set-theory-1874.pdf`

### Notes created (May 22, 2026)
- **Cavalieri 1635** ✅ noted — indivisibles close the integration-side gap between Archimedes' exhaustion and Newton-Leibniz calculus: continuous magnitudes become comparable through parallel slices. Source PDF: `corpus/raw/library-papers/cavalieri-indivisibles-1635.pdf`

### Notes created (May 20, 2026)
- **Fermat 1637** ✅ noted — adequality closes the gap between Descartes' equations-as-curves and Newton-Leibniz calculus: local perturbation makes maxima/minima and tangents algebraic. Source PDFs: `corpus/raw/library-papers/fermat-maxima-minima-1637.pdf`, `corpus/raw/library-papers/stromholm-fermat-1965.pdf`, `corpus/raw/library-papers/1210.7750.pdf`

### Notes created (May 18, 2026)
- **Viète 1591** ✅ noted — symbolic notation closes the gap between Al-Khwarizmi's rhetorical algebra and Descartes' analytic geometry. Source PDF: `corpus/raw/library-papers/viete-symbolic-notation-1591.pdf`

### Source Strategy for Math
- **Primary texts in translation:** Internet Archive, Project Gutenberg, MacTutor
- **Modern references:**
  - David Bressoud, "Calculus Reordered" (2019) — our seed video references this
  - Rudy Rucker, "Infinity and the Mind" — referenced in video
  - Silvanus Thompson, "Calculus Made Easy" (1914) — referenced in Ananthaswamy's notes

---

### Optimization Lineage (AI-ML thread internal)

```
Euler 1744 (calculus of variations: optimize curves/functions) ✅ noted
  └── Lagrange 1788 (variational mechanics, virtual work, multipliers) ✅ noted
      └── Cauchy 1847 (steepest descent) ✅ noted
          └── Robbins-Monro 1951 (stochastic approximation) ✅ noted
              └── SGD (applied to neural nets via backprop)
                  ├── Polyak 1964 / Momentum SGD ✅ noted
                  ├── AdaGrad 2011 (Duchi et al.) ✅ noted
                  ├── RMSProp 2012 (Hinton, unpublished Coursera) ✅ noted
                  ├── Adam 2015 (Kingma-Ba) ✅ noted
                  │   └── AdamW 2019 (Loshchilov-Hutter) ✅ noted (1711.05101)
                  └── SGDR 2017 (Loshchilov-Hutter) ✅ noted (1608.03983) — cosine schedules + warm restarts for SGD
```

---

## Cross-Thread Connections 🔗

These are the exciting parts — where math and ML share DNA:

1. **Cauchy (1847) steepest descent → Robbins-Monro (1951) → Adam (2015)** — The same core algorithm forced into a statistical world. Cauchy gives downhill motion; Robbins-Monro proves noisy sample-based steps can converge; SGD → Momentum → RMSProp → Adam refine the bargain.
2. **Euler's functions → Feature spaces** — Thinking in functions rather than geometry
3. **Linear algebra (Hamilton's quaternions/vectors) → ML everywhere** — dot products, projections, weight matrices. Hamilton's quaternions (1843) were the first non-commutative algebra — the idea that algebraic structure could violate familiar axioms eventually enables weight matrices, operations that don't commute in general. Also: quaternions are the modern standard for 3D rotation in graphics and robotics (rotation-equivariant neural networks).
15. **Barrow's tangent-area reciprocity → loss functions and gradients** — Cavalieri accumulates slices; Fermat perturbs local quantities; Barrow shows that accumulated quantities and local tangents are inverse problems. Modern training loops inherit the same structure: accumulated error has a local slope, and local slope tells the optimizer how to move.
10. **AdamW as optimization lineage closer** — Cauchy 1847 (steepest descent) → Robbins-Monro 1951 (stochastic approximation) → Widrow-Hoff 1960 (delta rule) → SGD → AdaGrad 2011 → RMSProp 2012 → Adam 2015 → AdamW 2019. Each step fixes a specific failure mode of the previous: direction, noise, online error correction, adaptivity, and decoupled regularization.
11. **GPT-3 as the scaling laws made flesh** — Kaplan et al. (Jan 2020) wrote the theory; Brown et al. (May 2020) built the experiment. 8-month gap. 175B parameters. In-context learning *emerged* — it wasn't trained for explicitly. When is an emergent capability a phase transition vs. just smooth scaling? Open question.
4. **Probability (Bayes 1763 → Gauss 1809) → Naive Bayes, Gaussian processes, Bayesian ML** — Bayes gives the update rule; Gauss gives the error model and least-squares fitting principle. Together they explain why uncertainty and optimization are so entangled in modern ML.
5. **Infinity/limits → Convergence proofs** — Perceptron convergence theorem uses the same epsilon-delta spirit
6. **Statistical mechanics (Boltzmann) → Hopfield nets → Boltzmann machines** — Hopfield (1982) explicitly borrowed the energy function from spin glass physics. This is not a metaphor — it's the same mathematics. The Ising model (1920s) → spin glasses (1975) → Hopfield networks (1982) → Boltzmann machines (1983) is a direct physics → ML pipeline. Nobel Prize 2024 awarded to Hopfield and Hinton recognizes this bridge.
7. **Calculus of variations → Optimization theory → Loss function minimization**
8. **Power laws in physics → Scaling Laws in ML (2020)** — Kaplan et al.'s scaling laws mirror physical scaling laws (fractal dimensions, critical exponents). Loss ∝ compute^(-0.05) has the same form as how surface area scales with volume. The "shape doesn't matter, scale does" finding echoes statistical mechanics.
9. **Distributed representations across modalities** — word2vec (2013) shows words have algebra; DCGAN (2016) shows image latent spaces have the same algebra; CLIP (2021) shows vision and language share one geometric space. The hypothesis: if representations are good enough, they transfer and align. Holds from vectors to vision to multimodal.
12. **The Chinchilla correction** — Kaplan (2020) said scale the model; Chinchilla (2022) said scale model AND data equally. The 0.23 exponent gap (C^0.73 vs C^0.50 for model size) has trillion-dollar consequences. Every 175B+ model trained before Chinchilla was undertrained. Llama, Mistral, and the open-source revolution followed from "smaller + more data."
13. **Emergence and the reasoning scratchpad** — Chain-of-Thought (2022) shows that reasoning is *emergent* at ~100B parameters. The Transformer's autoregressive feedback loop (each generated token enters the context) creates an external scratchpad. Without CoT, a Transformer does constant-depth computation; with CoT, it can simulate arbitrary-length computation. This connects to Turing completeness and the theoretical foundations of computation — a new thread for the Library.
14. **Sparse scaling becomes operational** — Shazeer (2017) proved MoE could explode capacity without dense compute. Switch Transformer (2021), GShard (2020), GLaM (2021), and ST-MoE (2022) made that idea trainable, efficient, and transferable: top-1 routing simplified the path, sharding made giant sparse training operational, GLaM made the energy case, and ST-MoE stabilized routers and downstream fine-tuning. That is the direct bridge to Mixtral, DeepSeek, and the modern "quality per dollar" race.
16. **Deployment-aware evaluation becomes part of the frontier** — DeepSeek-V4 and Nemotron 3 Super show how the model recipe bends around long context, hybrid architectures, sparse routing, and throughput. Manik & Wang (2026) show the matching evaluation consequence: once many dense and MoE reasoning models are viable, the serious question is no longer "which model wins?" but "which model sits on the right accuracy-latency-memory frontier for this workload and prompt regime?"
