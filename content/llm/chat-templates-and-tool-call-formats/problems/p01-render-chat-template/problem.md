# Render chat messages into a template string

A chat model is trained on text in which every turn is wrapped in markers that say who is
speaking. The markers used at training time and at inference time must match, or the model sees
text it has never been trained on, and quality drops without any error. This problem defines one
template, renders a conversation into it, and rejects conversations whose turn order is invalid.

### The template (defined here)

Three control strings are reserved. They never appear inside a message's content.

- `<bos>` opens the rendered string. It appears once, at the start.
- `<msg:ROLE>` opens a message. `ROLE` is one of `system`, `user`, `assistant` or `tool`.
- `<eom>` closes a message.

A message with role `r` and content `c` renders as `<msg:r>c<eom>`. Nothing is added around it: no
spaces, no newlines. The content is kept byte for byte, including its whitespace.

The rendered string is `<bos>` followed by the rendered messages, in order. If
`add_generation_prompt` is true, append `<msg:assistant>`: an open assistant message with no content
and no `<eom>`. The model continues from that point.

## Contract

- `render_chat(messages, add_generation_prompt=False)` returns a `str`.
- `messages`: a `list` of `dict`. Each dict has exactly the keys `"role"` and `"content"`. A missing
  or extra key raises `ValueError`. A `messages` that is not a `list`, or an item that is not a
  `dict`, raises `TypeError`.
- `role`: a `str` in `{"system", "user", "assistant", "tool"}`. Any other value, including a `str`
  that differs in case, raises `ValueError`. A non-`str` role raises `ValueError` too.
- `content`: a `str`. Any other type raises `TypeError`. The empty string is valid. Content that
  contains `<bos>`, `<eom>` or `<msg:` raises `ValueError`, because those strings are reserved.
- `add_generation_prompt`: a `bool`. Any other type, including `int`, raises `TypeError`.
- Turn order (each breach raises `ValueError`):
  1. The list is non-empty and holds at least one message that is not `system`.
  2. `system` may appear only as the first message, and at most once.
  3. The first message that is not `system` has role `user`.
  4. A `user` message is followed by an `assistant` message, or the list ends.
  5. An `assistant` message is followed by a `user` message, a `tool` message, or the list ends.
  6. A `tool` message is followed by a `tool` message, an `assistant` message, or the list ends.
- `add_generation_prompt=True` requires the last message to be `user` or `tool`. After an
  `assistant` message the next turn cannot be an assistant turn, so the prompt is invalid there.
- Must not mutate `messages`.

## Examples

```python
>>> render_chat([{"role": "user", "content": "Hi"}], add_generation_prompt=True)
'<bos><msg:user>Hi<eom><msg:assistant>'
```

```python
>>> render_chat([
...     {"role": "system", "content": "Be brief."},
...     {"role": "user", "content": "2+2?"},
...     {"role": "assistant", "content": "4"},
... ])
'<bos><msg:system>Be brief.<eom><msg:user>2+2?<eom><msg:assistant>4<eom>'
```

```python
>>> render_chat([
...     {"role": "user", "content": "Oslo weather?"},
...     {"role": "assistant", "content": "CALL"},
...     {"role": "tool", "content": "4C"},
... ], add_generation_prompt=True)
'<bos><msg:user>Oslo weather?<eom><msg:assistant>CALL<eom><msg:tool>4C<eom><msg:assistant>'
```

Invalid: `[{"role": "assistant", "content": "Hi"}]` (the first message is not `user`), and
`[{"role": "user", "content": "a"}, {"role": "tool", "content": "b"}]` (a tool result must follow an
assistant message), both raise `ValueError`.

## Constraints

- Allowed: the standard library, for example `str` operations, `list` and `dict` operations, and
  `isinstance` checks.
- Not allowed: `jinja2`, and any other template engine. The template is the thing you are writing
  here, so the engine must not write it for you.

## Done looks like

All public tests pass. Submit then runs the hidden tests. They cover agreement with an independent
renderer on seeded random valid conversations, the byte-for-byte content (whitespace, newlines,
tabs, non-ASCII text), the empty message, the reserved strings inside content, every turn-order
family (system position, user followed by user, tool after user, assistant after assistant, a user
after a tool), the generation-prompt rules, the key, role and type rules, empty and system-only
lists, the non-list input, and inputs left unchanged.
