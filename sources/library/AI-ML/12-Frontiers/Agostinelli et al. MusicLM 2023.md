# Agostinelli et al. — MusicLM (2023)

*AudioLM could compose coherent audio from audio alone, but it had no natural-language control surface. MusicLM made text useful without requiring a caption for every training song.*

**← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**“MusicLM: Generating Music From Text”** — Andrea Agostinelli, Timo I. Denk, Zalán Borsos, Jesse Engel, Mauro Verzetti, Antoine Caillon, Qingqing Huang, Aren Jansen, Adam Roberts, Marco Tagliasacchi, Matt Sharifi, Neil Zeghidour, and Christian Frank, 2023.

- Paper: `corpus/raw/library-papers/2301.11325.pdf`
- arXiv: [2301.11325](https://arxiv.org/abs/2301.11325)
- Source PDF status: local file verified as `PDF document, version 1.5` on 2026-09-22

## Key Ideas

1. **Turn AudioLM from an unconditional generator into a text-controlled one.**
   MusicLM preserves AudioLM’s coarse-to-fine stack: a semantic model generates low-rate w2v-BERT tokens, then coarse and fine acoustic models generate SoundStream codec tokens that decode to 24 kHz audio. Its new problem is conditioning—how a sentence such as “Berlin 90s techno with a low bass and strong kick” can steer that hierarchy without a temporally aligned score or an enormous captioned-music corpus.

2. **Use a shared music–text space as a modality adapter.**
   A frozen MuLan model maps music and language into the same 128-dimensional contrastive embedding space. During training, MusicLM conditions on the MuLan embedding of the target *audio*; during inference, it swaps in the MuLan embedding of the user’s *text*. Because matching music and captions occupy nearby regions, the generator can learn from audio-only examples yet accept language at inference. This is the paper’s decisive bridge: paired data teaches the adapter once; abundant unlabelled music teaches the generator.

3. **Discretize the condition as well as the output.**
   MuLan processes overlapping ten-second windows; MusicLM averages their embeddings, then quantizes the result through 12 residual vector quantizers with 1,024 entries each. Those 12 conditioning tokens precede the generated sequence. The same quantizer handles MuLan audio embeddings at training time and MuLan text embeddings at inference time, giving the autoregressive system a homogeneous discrete interface rather than a separate continuous-conditioning mechanism.

4. **Keep semantic planning between the prompt and the waveform.**
   The semantic Transformer models 25 w2v-BERT tokens per second conditioned on MuLan tokens. Coarse and fine acoustic Transformers then predict 600 SoundStream tokens per second across 12 codebooks. An ablation that removed semantic modeling retained similar audio quality but worsened caption adherence—KLD rose from 1.01 to 1.05, MCC fell from 0.51 to 0.49—and degraded long-term structure. Text-to-codec generation alone can sound plausible while losing the musical idea.

5. **Scale with unlabelled music, not scarce expert descriptions.**
   The semantic and acoustic models train on five million clips—280,000 hours of music—without requiring those clips to have captions. Each of the three decoder-only Transformers has 24 layers, 16 attention heads, a 1,024-dimensional embedding, and about 430 million parameters. Semantic, coarse-acoustic, and fine-acoustic stages use 30-second, 10-second, and 3-second crops respectively, matching context length to the abstraction level being modeled.

6. **Make long generation an overlap-and-continue operation.**
   Although semantic training examples last 30 seconds, inference retains 15 seconds as a prefix and autoregressively generates the next 15. Repeating this window produces coherent clips several minutes long. “Story mode” changes the text condition every 15 seconds while retaining the musical prefix, producing tempo-consistent transitions between descriptions. The technique extends duration, but it does not explicitly model song form.

7. **Add melody as a second, non-verbal control channel.**
   Words describe genre, mood, and instrumentation poorly when the user already knows the tune. MusicLM trains a melody embedding with pairs such as covers, instrumental/vocal versions, humming, and singing; quantized melody tokens are concatenated with the MuLan text tokens. A hummed or whistled melody can therefore supply pitch contour while text chooses the rendering style.

8. **Create an evaluation target for text-to-music.**
   MusicCaps contributes 5,521 ten-second AudioSet clips with detailed descriptions written by professional musicians plus aspect labels for genre, mood, tempo, voices, instruments, rhythm, and more. On MusicCaps, MusicLM beat Riffusion and Mubert on VGGish FAD (4.0 versus 13.4 and 9.6), caption KLD (1.01 versus 1.19 and 1.58), and MuLan cycle consistency (0.51 versus 0.34 and 0.32). Human raters awarded it 312 pairwise wins versus 158 for Riffusion and 97 for Mubert, while real references still led with 472.

9. **Measure copying, then admit that generation risk remains.**
   Exact five-second semantic-token continuations matched training data in under 0.2% of tests even after a ten-second prompt; an approximate-matching procedure found about 1% at the selected threshold. That is evidence against simple lookup, not proof of legal or cultural safety. The authors flag creator misappropriation, cultural bias, and underrepresented musical traditions and did not release model weights. Generated-media capability arrived before a settled provenance regime.

## What Came Before → What It Enabled

### What came before

- [[Borsos et al. AudioLM 2022|AudioLM]] established the semantic → coarse acoustic → fine acoustic hierarchy needed for long-range coherence and waveform fidelity, but generated from audio prompts rather than open-ended language.
- [[Zeghidour SoundStream Neural Audio Codec 2021|SoundStream]] supplied a reconstructable 24 kHz acoustic vocabulary, while w2v-BERT supplied low-rate semantic units.
- Joint contrastive models such as [[Radford CLIP 2021|CLIP]] showed that separately encoded modalities could meet in one semantic space; MuLan specialized this idea to music and language.
- Earlier text-to-audio systems relied on less than roughly 5,000 hours of paired data, short scenes, handcrafted tag lookup, or spectrogram generation. Music captions were both expensive and temporally weak: one sentence cannot align every event in a minute-long song.

### What broke

AudioLM solved *how to generate coherent sound*, not *how a person should specify the sound*. Training a text-conditioned AudioLM directly would appear to require captions for the same hundreds of thousands of music hours used to learn musical structure. Those captions do not exist, are expensive to write, and still omit timing, instrumentation changes, melody, and production detail.

Nor could a larger text encoder repair the data mismatch by itself. The generator needed a conditioning signal during every audio-only training example. MusicLM’s answer is **modality substitution**: condition on MuLan’s music embedding while learning, then substitute a nearby text embedding at inference. It decouples the quantity of generative training data from the quantity of paired music–language data.

But one global text vector still cannot specify an exact tune or ordered song structure. The paper therefore exposes its own next break: negation and temporal ordering are weak, vocal quality and lyrics remain poor, and minute-scale continuation is local extrapolation rather than explicit verse–chorus composition.

### What it enabled

- High-fidelity text-to-music generation over minutes rather than seconds, with natural-language control over genre, mood, instruments, and production qualities.
- A reusable weak-supervision pattern: train a cross-modal embedding on scarce pairs, freeze it, then train the expensive generator on abundant unpaired data by swapping modalities at inference.
- Combined text-plus-melody control, separating *what tune* to follow from *how it should sound*.
- MusicCaps, a durable benchmark for caption adherence and music quality.
- The next codec-model branch, including MusicGen, which keeps neural audio tokens and text control but attacks MusicLM’s three-model serial pipeline with a more efficient multi-codebook language-model schedule.
- A sharper provenance question for generative media: low measured memorization does not by itself settle attribution, consent, cultural representation, or creator compensation.

## Why It Matters

MusicLM is not merely AudioLM with a text encoder attached. Its key insight is that **the conditioning modality used during training does not have to be the modality supplied at inference**, provided a learned shared space makes the substitution meaningful. That lets scarce paired supervision define the interface while abundant unlabelled data teaches the world model.

The paper also clarifies why generative AI systems accumulate representation layers. Text, melody, semantic audio, and acoustic codec tokens each preserve different invariants. Forcing one representation to express all of them would either discard control or explode sequence length. MusicLM composes specialized representations in the order of the decisions they make: intent, musical structure, and waveform realization.

Its limitations are equally important. The model uses three large autoregressive stages, so generation is slow and errors cannot be jointly revised across levels. A pooled MuLan condition captures broad semantics but mishandles negation, precise event order, lyrics, and formal song structure. Sliding-window continuation extends time without truly planning an introduction, verse, chorus, and return. MusicLM therefore turns text-to-music from a toy into a credible system—and makes efficiency, compositional control, evaluation independence, and provenance the next unavoidable problems.

## Lineage

- **Built on:** AudioLM, SoundStream, w2v-BERT, MuLan, CLIP-style contrastive alignment, residual vector quantization, and decoder-only Transformers
- **Repairs:** AudioLM’s missing natural-language interface and the scarcity of captioned music at generative-training scale
- **Preserves:** semantic-first planning, hierarchical codec-token generation, and autoregressive continuation
- **Leads toward:** MusicGen, unified multi-codebook audio language models, text-plus-melody editing, long-form song planning, and music provenance safeguards

---

*AudioLM taught a hierarchy to compose sound. MusicLM taught language where to enter that hierarchy—without making every song explain itself first.*
