# Interview notes — RASS case framework

## Q1. Walk me through how you would approach an open-ended ML product question.
**60-second answer.** I start by naming the decision the model informs, then restate the problem with
a number and a success metric. Before choosing a model I ask about the cost of each error type, the
current process I am beating, and the data that exists. The model comes after that: a baseline first,
then a candidate with a reason, evaluated on a holdout. I finish with the business effect of the
decision, and a pilot that tests it.
**Follow-ups.** Why define the decision first? It fixes the metric, and the metric fixes the threshold.
What if the stakeholder has no numbers? State an assumption, mark it as one, and say what you would check first.

## Q2. How do you keep a case answer on time without rushing it?
**60-second answer.** I budget the four moves: about two minutes for the Reframe, four for the Ask,
twelve for the Structure and two or three for the Synthesize. I say each move's name out loud, so the
interviewer can follow, and the names give me a fixed place to return to if I lose the thread.
**Follow-ups.** What if the interviewer interrupts? Answer the question, then return to the move you were in.
What if you run out of time? Go to the Synthesize and state the recommendation with its caveat.

## Common mistakes
- Naming a model in the first minute, before the decision is clear.
- Asking twenty questions and never stating an assumption.
- Quoting accuracy as the business result.
