# Interview notes — Chat templates and tool-call formats

## Q1. How does a chat model turn a list of messages into its input?

**60-second answer.** The model reads one string. A chat template renders the messages into that string. Each message is wrapped in reserved markers that give its role and close it, and the rendering starts with a start marker. The turn-order rules say which role can follow which: a user turn is answered by an assistant turn, and a tool result follows an assistant call. If generation is needed, the string ends with the open marker of the next assistant turn. The template is part of the model: training and inference must use the same one.

**Follow-ups.**

- What goes wrong with a different template? The model reads text it was not trained on. Quality drops, and nothing raises an error.
- Why reserved markers? Each one maps to a single token. If the renderer rejects content that contains a marker, a user cannot close a turn early or open a fake one.

## Q2. How do you let a model call a tool safely?

**60-second answer.** The assistant writes a structured call in its turn, usually a JSON object with a tool name and arguments. The application parses that text, checks that there is exactly one call, decodes the JSON strictly, and validates the name, the required keys, and each value's type against the tool's schema. Only a call that passes runs. The tool's result goes back as a tool turn, and the model continues. Each failure gets its own error, so the caller can retry or report it.

**Follow-ups.**

- What checks matter most? Type checks that exclude booleans from integer slots, rejection of non-finite numbers, rejection of duplicate keys, and an exact match on the tool name.
- Why parse before running? The model's text is data, not a command. Validation is the boundary between the two.

## Common mistakes

- Rendering with a template the model was not trained on, and blaming the model.
- Letting user content contain a control string.
- Accepting a `bool` where an integer is declared.
- Trusting a tool call without checking the schema.
