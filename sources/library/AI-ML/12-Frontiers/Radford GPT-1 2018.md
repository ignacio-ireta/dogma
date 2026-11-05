# Radford et al. — Generative Pre-Training (GPT-1, 2018)

*The Transformer stopped being only a task architecture and became a reusable language model.*

**← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**"Improving Language Understanding by Generative Pre-Training"** — Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever, OpenAI, 2018.

- Paper: `corpus/raw/library-papers/radford-generative-pretraining-2018.pdf`
- Official source: OpenAI technical report, published June 11, 2018
- Local source verified as a PDF document

## Key Ideas

1. **Use unlabeled text to learn the model, then use labels only to steer it.**
   NLP systems had abundant raw text but scarce task labels. GPT-1 split training into two stages: first maximize next-token likelihood on BooksCorpus, then fine-tune the same parameters on each labeled task. The expensive linguistic knowledge came from text itself rather than being rebuilt from every benchmark.

2. **The transferable object was the whole Transformer, not a bag of word vectors.**
   Word2vec-style embeddings transferred a fixed vector for each word. Earlier language-model approaches often supplied contextual representations as features to a new task-specific network. GPT-1 instead transferred all 12 Transformer layers and updated them end to end. Its layer-transfer analysis showed that each additional pretrained layer helped, not just the embeddings.

3. **One decoder could consume many task structures as text.**
   Entailment has premise-hypothesis pairs; question answering has document-question-answer triples; similarity is symmetric. Instead of building a different neural architecture for each one, GPT-1 serialized these structures with delimiter tokens and passed them through the same decoder, adding only a small linear output layer.

4. **Generative pretraining and discriminative fine-tuning could cooperate.**
   During fine-tuning, the model optimized the supervised task objective plus an auxiliary language-model objective. The language objective helped preserve the general representation while labels taught the final decision boundary. The paper found this especially useful on some larger datasets.

5. **The architecture was modest; the recipe was the breakthrough.**
   GPT-1 used a 12-layer decoder-only Transformer with 768-dimensional states, 12 attention heads, 512-token sequences, and a 40,000-merge BPE vocabulary. It trained on more than 7,000 books. Later GPT models became famous for parameter counts, but the first GPT mattered because it established the training sequence:

   ```text
   broad generative pretraining -> lightweight task adaptation
   ```

6. **The ablations showed what was doing the work.**
   Removing pretraining reduced the average score from 74.7 to 59.9. Replacing the Transformer with an LSTM reduced the average by 5.6 points. The result was not merely a better classifier head: pretrained Transformer layers contained reusable task-relevant structure.

7. **Zero-shot behavior appeared before zero-shot use became the product.**
   As language-model training progressed, heuristic zero-shot performance improved on sentiment, acceptability, question answering, and pronoun resolution. GPT-1 still relied on supervised fine-tuning for its headline results, but this observation supplied the clue that [[Radford GPT-2 2019|GPT-2]] would pursue directly.

## What Came Before → What It Enabled

### What came before

- [[Vaswani Attention Is All You Need 2017|The Transformer]] made long-range sequence modeling parallelizable, but its flagship use was supervised machine translation.
- [[Mikolov Word2Vec 2013|Word embeddings]] transferred lexical knowledge, but a word received one vector regardless of context.
- LSTM language-model pretraining showed that raw text could improve classification, but recurrent models struggled with long-range structure and transfer often remained task-specific.
- NLP benchmarks were dominated by bespoke architectures trained mainly from labeled examples.

### What broke

Task-specific supervised learning could not scale with language.

Every new benchmark demanded annotation, a specialized architecture, and another model trained largely from scratch. Static embeddings carried useful lexical similarity, but not a reusable hierarchy of contextual reasoning. The Transformer solved the sequence-computation problem, yet the field was still using that machinery as if each task were a separate engineering project.

They could not keep building one supervised model per task because labels were the scarce resource and language knowledge was being relearned repeatedly.

GPT-1 made raw text the shared training substrate and task labels the final adaptation step.

### What it enabled

- [[Radford GPT-2 2019|GPT-2]] removed supervised fine-tuning from the central claim and tested whether broad web-scale language modeling itself induced zero-shot task behavior.
- [[Devlin BERT 2018|BERT]] took the pretrain-then-fine-tune recipe down the encoder-only, bidirectional path.
- [[Brown GPT-3 2020|GPT-3]] scaled the decoder-only path until task adaptation could happen through examples in the prompt rather than gradient updates.
- [[Ouyang InstructGPT RLHF 2022|InstructGPT]] later added instruction demonstrations and human preferences, turning a completion model into a cooperative assistant.

## Why It Matters

GPT-1 is the missing hinge between the Transformer and generative AI.

The Transformer paper supplied a powerful architecture, but an architecture alone does not explain why one model should know sentiment, entailment, question answering, and commonsense reasoning. GPT-1 supplied the economic and methodological answer: train once on plentiful unlabeled language, then reuse the learned network wherever labels are scarce.

Its limitation also logically necessitated the next step. GPT-1 still needed fine-tuning for each target task. The universal representation was shared, but the final model was not yet universal in use. If every task still required a labeled dataset and parameter update, supervised adaptation remained the bottleneck.

That is exactly what GPT-2 attacked:

```text
GPT-1: can one generatively pretrained model be adapted to many tasks?
GPT-2: can one generatively pretrained model perform many tasks without adaptation?
GPT-3: can examples in context replace adaptation?
InstructGPT: can human preferences make that general model reliably helpful?
```

Generative AI exists because this sequence kept removing a different form of task-specific labor: first training from scratch, then fine-tuning, then explicit task programming, and finally much of the awkwardness of specifying behavior.

## Lineage

- **Built on:** [[Vaswani Attention Is All You Need 2017|decoder-only Transformers]], language modeling, transfer learning, and semi-supervised sequence learning.
- **Branches into:** [[Devlin BERT 2018|BERT's bidirectional encoder path]] and the autoregressive GPT path.
- **Leads to:** [[Radford GPT-2 2019|GPT-2]], [[Brown GPT-3 2020|GPT-3]], instruction tuning, RLHF, and conversational assistants.

---

*The Transformer made the machine trainable. GPT-1 taught it to learn language once and reuse what it knew.*
