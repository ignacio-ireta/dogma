# Interview notes — BPE merge training

## Q1. Walk me through training a BPE vocabulary.
**60-second answer.** Start from the characters of each word with an end-of-word symbol, and keep a table of word
counts. Each step counts adjacent symbol pairs, weighted by how often each word occurs, and merges the most
frequent pair in every word. Repeat for the number of merges, then stop early if no pair is left. The
vocabulary grows by one symbol per merge. I break ties by a fixed rule so the result does not depend on dict order.
**Follow-ups.** Why learn on counts and not raw text? It is smaller, and no pair crosses a word boundary.
What is the cost? A naive recount costs time proportional to the total symbol count for every merge. Incremental pair indexes reduce that.

## Q2. What edge cases break a naive BPE trainer?
**60-second answer.** Overlapping pairs such as `aaa`, where the count depends on whether you count
occurrences or non-overlapping runs. Ties, where the answer depends on iteration order. The end-of-word
marker, which must merge with the last character. And early stop, when every word is one symbol.
**Follow-ups.** How do you test it? Compare with a naive trainer on random tables, and hand-check a worked example.

## Q3. Why is the merge order important?
**60-second answer.** Each merge is chosen on the symbols that earlier merges produced, so the list is an ordered
program. Applying the same merges in a different order can give a different segmentation. The encoder must use the
same order the trainer learned.
**Follow-ups.** What happens with a different tie rule? The vocabulary changes, and a model trained on the old
vocabulary reads the wrong ids.

## Common mistakes
- Counting pairs once per word instead of once per word occurrence.
- Forgetting to recount after each merge.
- Using dict insertion order to break ties.
- Dropping the end-of-word marker, so word endings are not learned.
