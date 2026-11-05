# Interview notes: Stacks and queues

## Q1. For each day, how many days until a warmer day? (Next greater element)

**60-second answer.** The answer for a day is the distance to the first later day that is strictly warmer. I scan
left to right and keep a stack of days still waiting for a warmer day. Their temperatures never increase from
bottom to top. When a new day is warmer than the top of the stack, I pop the top and record the distance, and I
keep popping while the new day is warmer. Then I push the new day. Each day is pushed once and popped at most once,
so the pass is O(n). A nested loop would look at every later day for each day, which is O(n squared). Equal
temperatures do not end a wait, so the pop test is strict.

**Follow-ups.**
- *What about the previous warmer day?* Keep the same left-to-right scan, but pop while the top is at most the new
  value. The top left after the pops is the answer, and the new day is then pushed.
- *What if there are no warmer days?* The entry stays 0, which is the value the stack never resolves.

## Q2. Give the maximum of every window of width k, as the window slides across an array.

**60-second answer.** I keep a deque of indices whose values never increase from front to back. The front is the
answer for the current window. For each new index, I first remove the front if it has left the window. Then I remove
from the back every index whose value is not larger than the new value, because the new value stays in the window at
least as long and is at least as large. Then I push the new index, and once the first window is full, I record the
front value. Each index enters and leaves the deque once, so the cost is O(n). Validate the width first: it must be
between 1 and the length.

**Follow-ups.**
- *Why not a heap?* A heap gives O(n log n), and removing expired entries needs lazy deletion.
- *Is there an O(n) method without a deque?* Yes. Split the array into blocks of width k, take prefix maxima inside
  each block and suffix maxima inside each block. Each window is then the larger of one suffix and one prefix.

## Common mistakes
- Choosing a queue where the answer needs the most recent item, or a stack where it needs the oldest.
- Using `if` instead of `while` in the pop loop, which leaves waiting items unresolved.
- Using `<=` in a next-greater pop test, which lets equal values end a wait.
- Forgetting to check the front index for expiry in a sliding-window deque.
- Calling `list.pop(0)` in a loop, which turns a linear pass into a quadratic one.
