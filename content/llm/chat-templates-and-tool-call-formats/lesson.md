---
title: "Chat templates and tool-call formats"
minutes: 12
min_viable: "Read The system and the Worked example, then do p01 with the template written out on paper."
---

## The system

A chat model reads one flat string of tokens. Turn boundaries and roles are not a separate channel. They are text that the template writes into the string. The template has four parts:

1. **Roles.** Each message has a role: `system` (instructions), `user`, `assistant` (the model), or `tool` (the result of a tool call).
2. **Special tokens.** Markers that open and close each message. They are reserved strings the tokenizer can map to single token ids. A renderer that rejects those strings inside content keeps turn boundaries exact.
3. **Turn order.** A legal sequence: an optional single `system` first, then `user` answered by `assistant`. An `assistant` turn may be followed by a `tool` result. A `tool` result is followed by another `tool` result or by an `assistant` turn. These are the rules p01 enforces. Other templates allow other orders.
4. **Generation prompt.** When `add_generation_prompt` is true, the string ends with the open marker of the next assistant turn. The model continues from there.

The rendering is a fold over the messages: a start marker, then each message as `open + content + close`, then the optional generation prompt. Nothing is added between pieces, unless the template writes it.

## Worked example

Take one user question, an assistant tool call, and the tool's result, with the template $\langle\text{bos}\rangle$, $\langle\text{msg:}r\rangle c\langle\text{eom}\rangle$ per message, and a generation prompt:

```text
<bos><msg:user>Oslo weather?<eom><msg:assistant><call>{"name": "get_weather", "arguments": {"city": "Oslo"}}</call><eom><msg:tool>{"temp_c": 4}<eom><msg:assistant>
```

The string ends with the open assistant marker, so the model writes the next assistant turn. The call is a JSON object inside the assistant text. Parsing it is a separate step: find one block, decode the JSON, then check the name, the required keys and the types against the tool's schema. A call that fails the schema should not be run.

## Why it works

Training teaches the model a particular string shape. Instruction-tuned models learn the conversation format from demonstrations, as InstructGPT's supervised fine-tuning step did [library:AI-ML/12-Frontiers/Ouyang InstructGPT RLHF 2022.md]. At inference the same shape should appear.

Practical guidance, not a measured result in the corpus: if the template differs, for example a different marker, a missing close token, or an extra newline, the model sees text it was not trained to read. Nothing raises an error, so the quality change is easy to miss.

Tool-call format choice also affects errors. DeepSeek-V4 describes an XML-style tool-call format and reports, from its own experiments, that it "effectively mitigates escaping failures and reduces tool-call errors". It also separates string parameters, written as is, from other typed values, passed as JSON [corpus:library-papers/deepseek-v4-2026.pdf.md p.30]. That is one vendor's result for one format, not a general law.

## Signal vs. noise

Signal: the four parts above; that rendering is plain concatenation; that a tool call is structured text which must be parsed and validated; and that the model and the template are one unit. Noise: the exact marker strings of any one vendor, which change between versions. Learn the principle with one template you define, then read any real template with the same four questions.

## Where it breaks

- **Mismatch.** A template the model was not trained on can lower quality without an error. Check the template, not only the model.
- **Forged turns.** Content that contains a control string can open a new turn. Reject reserved strings in content, as p01 does.
- **Turn-order errors.** A tool result with no preceding assistant call, or two user messages in a row, makes a string the template's rules do not allow. Validate the order before rendering.
- **Tool-call errors.** A call can be valid JSON and still wrong. Name the cause with one of two errors, shown in the table below.
- **Duplicate keys.** Python's `json` keeps the last of two duplicate keys without complaint. The parser needs a hook that rejects duplicates when the format says keys are unique.
- **Booleans.** A boolean is an integer in Python, so the integer check must exclude `True` and `False` explicitly.

| Error | Causes, with examples from this topic |
|---|---|
| `ToolCallParseError`: the text is not a well-formed call | no `<call>` block; two blocks; a missing `</call>`; `NaN`; `1e400`; a repeated key; `"arguments": ["a"]` |
| `ToolSchemaError`: the call is well formed but does not match the tool | `"name": "get_wether"` for `get_weather`; a missing required `city`; `"days": "2"` (string for integer); `"days": 2.0` (float for integer); `"days": true`; an undeclared `"units"` parameter |

The split matters for the caller. A parse error means the model wrote something unreadable, so retry or repair the text. A schema error means the model read the tool wrongly, so report which argument failed.

## Interview angle

Say that the model sees one string, so the template is part of the model. Describe the render: roles, reserved markers, turn-order rules, and the generation prompt. For tool use, describe the call as structured text, the result as a `tool` turn, and the schema check before anything executes. Full versions are in `interview.md`.

## Connections

Prerequisite: [word and character tokenizers](../word-and-character-tokenizers/). The next topic, [instruction-tuning batches and loss masks](../instruction-tuning-batches-and-loss-masks/), uses the same templates to decide which tokens the loss covers.
