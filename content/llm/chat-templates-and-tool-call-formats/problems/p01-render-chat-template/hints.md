## Hint 1 — concept

A conversation becomes one string. Each message is wrapped in its role marker and a close marker,
and the whole string starts with the begin marker. Before any text is built, the conversation must
be checked: which roles are allowed, in which order, and whether the content may contain the
reserved strings.

## Hint 2 — formula / approach

Render as `<bos>` followed by `<msg:ROLE>content<eom>` for each message, plus `<msg:assistant>` when
the generation prompt is requested. Validate first. Check each message's keys, role and content
type, and reject reserved strings. Then check the turn order as adjacent pairs: a user turn is
followed by an assistant turn, an assistant turn by a user or tool turn, and a tool turn by a tool
or assistant turn. A system message is legal only at the start. The generation prompt is invalid
after an assistant turn.

## Hint 3 — code shape

Use a helper that checks one message and returns its role, and a helper that walks the role list.
In the walk, `itertools.pairwise` gives the adjacent pairs. Look each pair up in a dict of allowed
next roles and raise a named `ValueError` on a miss. Build the output with a list of parts and
`"".join`, so that nothing is added between the pieces.
