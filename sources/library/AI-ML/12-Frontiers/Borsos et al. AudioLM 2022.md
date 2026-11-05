# Borsos et al. — AudioLM (2022)

*SoundStream made high-fidelity audio discrete; AudioLM showed why one acoustic token stream was still not enough. Sound needs one representation for long-range meaning and another for local fidelity.*

**← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**“AudioLM: a Language Modeling Approach to Audio Generation”** — Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi, and Neil Zeghidour, 2022 (arXiv revision, 2023).

- Paper: `corpus/raw/library-papers/2209.03143.pdf`
- arXiv: [2209.03143](https://arxiv.org/abs/2209.03143)
- Source PDF status: local file verified as `PDF document, version 1.5` on 2026-09-20

## Key Ideas

1. **Turn audio generation into language modeling—but first choose the right alphabet.**
   AudioLM freezes pretrained tokenizers, maps a waveform into finite symbol sequences, trains decoder-only Transformers to predict the next symbols, and converts generated acoustic symbols back to audio. Tokenization reduces a raw 16 kHz waveform by two to three orders of magnitude before self-attention sees it. The decisive contribution is not merely applying a Transformer to audio; it is recognizing that no single audio alphabet simultaneously preserves long-range structure and waveform detail.

2. **Expose the semantic–acoustic rate–distortion conflict.**
   SoundStream acoustic tokens reconstruct well but spend most of their bitrate on speaker identity, recording conditions, and fine signal detail. At 2,000 bps they reached 3.3 ViSQOL, yet had poor across-speaker phonetic discrimination (28.7% ABX error). Discretized w2v-BERT features do the opposite: at only 250 bps they represented phonetic content well (7.6% across-speaker ABX error) but reconstructed badly (1.1 ViSQOL). Raising either representation's bitrate did not erase its bias. **Invertibility and abstraction are different objectives.**

3. **Use semantic tokens to decide what happens.**
   AudioLM extracts 1,024-way semantic tokens from an intermediate layer of a self-supervised w2v-BERT model at 25 Hz. These tokens discard much of the speaker and waveform detail but retain phonetics, syntax, linguistic content, melody, harmony, and rhythm. A first autoregressive Transformer models this low-rate sequence across the full target duration, giving global structure the shortest and cheapest token stream.

4. **Use hierarchical acoustic tokens to decide how it sounds.**
   A SoundStream codec emits 12 residual-quantizer tokens per 20 ms frame—600 tokens per second and 6 kbps. AudioLM splits them into four coarse codebooks and eight fine codebooks. A second Transformer predicts coarse acoustic tokens conditioned on the complete semantic sequence; a third predicts fine residual tokens conditioned on the coarse ones. The hierarchy moves from content, to voice and recording conditions, to local waveform detail.

5. **Match dependency range to modeling stage.**
   The semantic stage uses up to 30 seconds of context, the coarse acoustic stage 10 seconds, and the fine stage only 3-second non-overlapping chunks. Fine acoustic details need not attend across an entire utterance once semantic and coarse acoustic decisions are known. Three separate stages keep each attention problem smaller than one interleaved sequence of every token type and let local fidelity scale independently of global duration.

6. **Generate speech without text supervision.**
   All speech components were trained on 60,000 hours of untranscribed Libri-Light audio. The semantic model nevertheless learned lexical and grammatical regularities: AudioLM scored 83.7% on in-vocabulary sWUGGY and 64.7% on sBLIMP, the strongest reported text-free results in the paper and an 8% relative sBLIMP improvement over the prior text-free state of the art. Ground-truth semantic tokens converted through the acoustic stages yielded 6.0% word error rate, showing that the compressed semantic sequence preserved linguistic content.

7. **Preserve identity through the acoustic prompt, not the semantic stream.**
   Semantic tokens intentionally carried little speaker identity: acoustic samples generated from the same semantic sequence matched the original speaker only 3.2% of the time. But when the system received both semantic and acoustic tokens from a three-second prompt, its seven-second continuations retained the detected speaker in 92.6% of cases. Content and identity remain separable until conditioning recombines them.

8. **Demonstrate both fidelity and risk.**
   Human raters distinguishing real ten-second speech from three-second-real plus seven-second-generated continuations were correct only 51.2% of the time—not significantly above chance. A dedicated classifier detected generated continuations at 98.6% accuracy, but that experiment also made the misuse problem concrete: a textless model could continue an unseen voice, prosody, and recording environment from seconds of audio. The paper treats detection as part of the system, while acknowledging impersonation, biometric spoofing, bias, accent, and dialect risks.

## What Came Before → What It Enabled

### What came before

- [[Zeghidour SoundStream Neural Audio Codec 2021|SoundStream]] made general audio compact and reconstructable with residual vector quantization, a causal encoder/decoder, perceptual losses, and explicit bitrate control.
- Self-supervised speech models such as HuBERT and w2v-BERT learned low-rate units that tracked phonetic and linguistic structure without transcripts, but those units threw away too much signal detail for high-fidelity synthesis.
- Textless spoken-language models generated coherent content from semantic speech units, but usually with restricted speakers and clean conditions; high-rate acoustic-token models produced convincing timbre but tended toward structurally incoherent speech or music.
- Transformers could model long dependencies, but quadratic attention made raw waveforms—and even flattened high-rate codec streams—an uneconomical representation for long audio.

### What broke

The field could not simply feed all SoundStream tokens into a larger Transformer. At 600 acoustic tokens per second, a 30-second example becomes 18,000 tokens before accounting for prompts or longer-form generation. More importantly, most of those decisions describe fine acoustic detail. The model must spend scarce long-range attention comparing tiny waveform refinements when the hard global question is whether the next phrase, sentence, chord, or rhythm makes sense.

Nor could it generate from semantic tokens alone. Their very success at becoming speaker-invariant and content-focused makes them poorly invertible. A low-bitrate sequence can preserve *what was said* while losing the voice, room, prosody, and high-frequency structure needed to reconstruct *how it sounded*.

AudioLM repairs the conflict by refusing to force one representation to do incompatible jobs: **semantic tokens carry long-range structure; coarse acoustic tokens recover identity and conditions; fine residual tokens restore local fidelity.** Prediction proceeds in that same coarse-to-fine order, so global decisions are made before local detail consumes the context budget.

### What it enabled

- A general recipe for audio language models: represent meaning and acoustics at different rates, model them hierarchically, and decode the final codec stream to waveform audio.
- High-quality, textless continuation of an unseen speaker from a short acoustic prompt, and coherent piano continuation without MIDI or symbolic music supervision.
- Later codec language models such as MusicLM and MusicGen, which inherit the premise that neural codec indices are a generative vocabulary while refining conditioning and multi-codebook scheduling.
- Modern speech-to-speech and multimodal systems that separate semantic planning from acoustic realization, often sharing discrete units with text or other modalities.
- A concrete alignment problem for generated media: preserving identity is technically useful and simultaneously creates impersonation and biometric-spoofing risk.

## Why It Matters

AudioLM is the moment learned audio compression becomes a **generative hierarchy**. SoundStream solved the codec problem: how to represent a waveform as a compact stream of reconstructable symbols. But compression optimizes for keeping every perceptually important detail, while long-form generation needs an economical representation of structure. Those objectives point toward different bitrates and different invariances.

The paper's durable insight is therefore a division of labor across representation levels. The semantic model plans over a low-rate abstraction; acoustic models progressively realize that plan. This mirrors a broader pattern in generative AI: expensive global reasoning should happen over the smallest representation that preserves the relevant decision signal, while local decoders spend compute only after the global structure is fixed.

The repair is still incomplete. AudioLM uses three separately trained autoregressive Transformers, samples stages serially, depends on pretrained tokenizers, and cannot jointly revise a semantic decision after hearing its acoustic realization. It also demonstrates voice continuation before fully solving provenance or misuse. These limits motivate more efficient multi-codebook schedules, stronger text and multimodal conditioning, unified semantic/acoustic token spaces, parallel generation, and durable generated-media detection.

## Lineage

- **Built on:** SoundStream, residual vector quantization, w2v-BERT, self-supervised speech representation learning, decoder-only Transformers, and textless NLP
- **Repairs:** the conflict between high-fidelity acoustic tokens and low-rate long-range structure
- **Preserves:** next-token modeling, discrete learned representations, and codec-based waveform reconstruction
- **Leads toward:** MusicLM, MusicGen, neural codec language models, speech-to-speech generation, multimodal token hierarchies, and audio provenance safeguards

---

*SoundStream gave machines an acoustic alphabet. AudioLM separated the sentence from the voice, then taught the hierarchy to compose them back into sound.*
