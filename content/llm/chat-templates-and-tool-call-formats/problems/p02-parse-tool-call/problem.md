# Parse and validate a tool call from assistant text

When a chat model wants to use a tool, its assistant turn carries a structured call instead of
prose. This problem defines one call format, parses it out of the assistant's text, and checks it
against the tool's parameter schema. The two failure kinds are named, so a caller can tell a
malformed call (the model wrote something unparseable) from a well-formed call with the wrong
arguments (the model misread the tool).

### The call format (defined here)

An assistant turn that calls a tool has exactly one block:

```text
<call>{"name": "get_weather", "arguments": {"city": "Oslo", "days": 2}}</call>
```

- The block starts with `<call>` and ends with `</call>`. Whitespace before the opening tag and
  after the closing tag is allowed. Any other text outside the block is an error.
- Between the tags is one JSON object with exactly two keys: `name` (a string) and `arguments` (a
  JSON object).
- JSON numbers must be finite. `NaN`, `Infinity` and `-Infinity` are not valid JSON numbers here,
  and neither is a literal such as `1e400` that overflows to infinity.
- No object in the body may repeat a key. A repeated key is an error, not the last value.

### Schema

The schema describes one tool:

```python
{
    "name": "get_weather",
    "parameters": {
        "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["city"],
    },
}
```

- `name` is the tool's name. The call's `name` must equal it.
- `properties` maps each parameter to a `type`, one of `"string"`, `"integer"`, `"number"` or
  `"boolean"`. The schema is assumed well formed.
- `required` lists parameters that must appear in `arguments`.
- Type rules: `integer` accepts a JSON integer, not a float (so `2.0` is rejected) and not a
  `bool`. `number` accepts an integer or a float, but not a `bool`. `boolean` accepts `true` or
  `false` only, so `1` is rejected. `string` accepts a `str`.

## Contract

- `parse_tool_call(text, tool)` returns a `dict` with exactly the keys `"name"` and `"arguments"`.
  `arguments` holds the parsed values, with the JSON types they were parsed to.
- Errors, all subclasses of `ValueError`. The starter defines them, and you must keep them:
  - `ToolCallParseError`: the text is not one call block; the JSON is malformed; a number is
    non-finite; a key repeats; the top level is not an object with exactly `name` and `arguments`;
    `name` is not a string; or `arguments` is not an object.
  - `ToolSchemaError`: the call's `name` differs from the schema; a required parameter is missing;
    an argument is not in `properties`; or an argument value has the wrong type.
- `text` must be a `str`. Otherwise `TypeError`.
- Must not mutate `text` or `tool`.

## Examples

```python
>>> tool = {"name": "get_weather",
...         "parameters": {"properties": {"city": {"type": "string"},
...                                       "days": {"type": "integer"}},
...                        "required": ["city"]}}
>>> parse_tool_call('<call>{"name": "get_weather", "arguments": {"city": "Oslo", "days": 2}}</call>', tool)
{'name': 'get_weather', 'arguments': {'city': 'Oslo', 'days': 2}}
```

```python
>>> parse_tool_call('<call>{"name": "get_weather", "arguments": {"days": 2}}</call>', tool)
ToolSchemaError                     # the required parameter "city" is missing
```

```python
>>> parse_tool_call('Let me check the weather.', tool)
ToolCallParseError                  # there is no call block
```

## Constraints

- Allowed: the standard library, for example `json`, `re`, and `str` operations. The `json`
  module's `loads` function accepts optional hooks, which you may use.
- Not allowed: `jsonschema`, `pydantic`, and other validation libraries. Validating against the
  schema is the work of this problem.

## Done looks like

All public tests pass. Submit then runs the hidden tests. They cover a round trip on seeded random
valid calls against an independent `json.loads` reference, exact integers above `2**53`, the
integer and number type rules (booleans, floats in integer slots, integers in number slots), the
non-finite numbers (NaN, Infinity, and an overflowing literal), duplicate keys, the block shape
(missing tags, two blocks, text outside the block, surrounding whitespace), the top-level shape,
the wrong tool name, missing required parameters, unexpected parameters, the type of `name` and of
`arguments`, the `TypeError` for a non-string text, and inputs left unchanged.
