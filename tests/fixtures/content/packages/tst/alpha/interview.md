# Interview notes: Alpha topic

## Q1. What does the doubling function do?

**60-second answer.** It returns a new list whose entries are twice the inputs. The input is
left alone, so the function is pure and safe to call twice.

**Follow-ups.**
- *Empty input?* Returns an empty list.
- *Why not in place?* Callers may still hold the original list.

## Q2. Why is mutation a problem here?

**60-second answer.** Shared state makes results depend on call order, which breaks tests and
caching.

## Common mistakes

- Mutating the input list.
