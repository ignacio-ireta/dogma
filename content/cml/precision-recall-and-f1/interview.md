# Interview notes: Precision, recall and F1

## Q1. Would you optimise precision or recall?

**60-second answer.** It depends on what each error costs, so I would ask that before choosing a number. A false
positive costs whatever the flag triggers, such as a review, an offer or a follow-up test. A false negative costs
whatever goes unnoticed. Once the cost ratio is known, I would fix the operating point the business can afford,
for example a cap on the number reviewed, and maximise recall within it. A precision floor is a separate constraint
on review quality. F1 balances precision and recall but does not encode equal business error costs; for known costs,
choose the threshold by held-out expected cost.

**Follow-ups.**
- *When is F1 the wrong choice?* When true negatives matter, because F1 ignores them, or when the costs are far from equal.
- *How do I choose the threshold?* On held-out scores, never on the data used to train the model.

## Q2. Why can macro F1 and micro F1 disagree on the same predictions?

**60-second answer.** Micro pools the counts, so large classes dominate it. For single-label data it equals accuracy.
Macro takes the mean of the per-class F1 values, so every class counts equally. If a rare class is failing, macro
drops sharply while micro barely moves. I would report macro when the rare class matters, and micro when the
overall error rate is what matters. Per-class scores should be shown either way.

**Follow-ups.**
- *Is macro F1 the harmonic mean of macro precision and recall?* No. The mean is taken per class, after the F1 is computed.
- *What is weighted F1 for?* It keeps per-class scores but weights each class by its share of the data, so it mirrors the frequencies.

## Common mistakes

- Choosing a threshold on the data used to train the model, which makes the reported F1 optimistic.
- Reporting F1 without the threshold, so the number cannot be reproduced or compared.
- Computing macro F1 as the harmonic mean of macro precision and macro recall, instead of averaging per-class F1.
- Splitting tied scores when sweeping thresholds, which can report an F1 no single threshold achieves.
