# Internal policy assistant, debrief

## Reframe

Which answer can an employee trust, with a link to its source, within 4 seconds and 0.02 dollars, and without
ever showing content the asker is not allowed to open? Model quality is measured on a frozen question set: the
share of answers reviewers rate correct and grounded, and the leak rate, the share of permission probes that return
restricted content, which must be zero. Whether the
assistant is worth launching is a decision-value question, which the Synthesize answers.

## Ask

Ask for the question mix; whether abstaining is acceptable; who can see each system's documents, and whether the
source systems expose access groups through an API; the latency and cost ceilings; and how the support desk
judges an answer today. The access answer decides the architecture. If live permissions are not available,
restricted sources cannot be in the assistant at launch.

## Structure

Start with the pipeline and name its two failure points. Ingest each document with its version, status,
effective date and access groups. Chunk by heading, keep tables whole, and check extraction on the most-asked
documents. At query time, filter by the asker's current groups, retrieve with hybrid search (keyword for error
codes, dense for paraphrase), keep 50 candidates and rerank to 5. Generate an answer that cites packed chunk
ids, and abstain when no chunk supports it. Measure retrieval recall@5 of the gold passage first. Then measure
generation: accuracy given the gold passage, and the share of claims in an answer that no packed chunk supports. Build a 400-question set,
stratified by question type, double-judged on 20 percent, frozen, and refreshed a fifth each quarter. Use a paired
test, because both systems answer the same questions.

## Synthesize

Model quality and decision value are different numbers. Model quality is recall@5, answer accuracy and the leak
rate on the frozen set. Decision value depends on how many answers the desk accepts and how many of those are
right. At about 44,000 questions a month, if 45 percent are accepted without escalation, that is about 19,800
answers. If each accepted answer replaces an escalation and saves about 10 of the 14 agent minutes, the desk saves
about 3,300 hours, worth about 148,500 dollars a month at 45 dollars an hour. Answering all 44,000 questions at the
0.02-dollar budget costs about 880 dollars a month. The prototype's 0.06 dollars an answer would cost about 2,640 a
month, so it must come inside budget before launch. Either way the savings dominate, so the assistant pays if the
accepted-answer rate holds. The gate is a different kind of test from the savings. One permission leak
into a restricted group can outweigh a month of savings, so launch needs zero leaks on adversarial permission
probes, a live access check on every query, and a staged rollout. The twist changes the
plan: pause the restricted sources, filter by live permissions from the source systems' API before generation,
exclude superseded versions, and pilot one department before widening.
