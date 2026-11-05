## Hint 1 — concept

Parsing and validating are two separate jobs. First the text must be a single, well-formed call.
Then the parsed call must match the schema: the right tool name, every required argument, no
undeclared arguments, and the declared type for every value. Keep the two kinds of failure apart,
because the caller handles them differently.

## Hint 2 — formula / approach

Strip the text and check that it holds exactly one `<call>` block and nothing else. Parse the body
with `json.loads`, passing a `parse_constant` hook that rejects `NaN` and `Infinity`, and an
`object_pairs_hook` that rejects a repeated key. Check that the result is an object with exactly
the keys `name` and `arguments`. Then check the schema in order: name, required keys, declared
keys, types. Remember that `bool` is a subclass of `int`, and that `1e400` parses to infinity.

## Hint 3 — code shape

Compute `stripped = text.strip()` and check that it starts with `<call>` and ends with `</call>`.
Slice the body between them. Call `json.loads(body, parse_constant=..., object_pairs_hook=...,
parse_float=...)` inside a try block that converts `json.JSONDecodeError` to your parse error. In
`parse_float`, reject a result that is not finite. For types, test `bool` first: a `bool` is never
an `integer` or a `number`. Then test `int` for `integer`, `(int, float)` for `number`, and `bool`
for `boolean`.
