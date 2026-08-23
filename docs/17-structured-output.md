# Structured Output

Structured output constrains the model's final answer to a schema. The
mechanism is a synthetic tool: the schema is advertised as a tool the model must
call, and the arguments it produces *are* the answer. That reuses the provider's
own constrained-decoding path, which is far more reliable than asking for JSON
in a prompt and parsing whatever comes back.

## Usage

```rust,ignore
use serde::Deserialize;
use strands_core::tool::StructuredOutputSpec;

#[derive(Deserialize)]
struct Person {
    name: String,
    age: u32,
}

let spec = StructuredOutputSpec::new(
    "Person",
    "A person record",
    json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }),
);

let person: Person = agent.prompt_structured("Describe Ada Lovelace", spec).await?;
```

## Why the schema is explicit

Upstream derives it from a pydantic model. Rust has no equivalent reflection, so
the schema is supplied directly and validation is a `serde` deserialize into
your type. If you already generate schemas (via `schemars` or similar), pass
that output straight in.

## Validation failures are recoverable

A model that gets its own schema wrong is handed the error as a **tool result**,
not a caller error, and usually fixes it on the next turn:

```text
Validation failed for Person. Please fix the following and call the tool again:
- missing field `age`
```

Failing the whole invocation over a mistake the model can correct would throw
away the run.

## Lifecycle

The synthetic tool is registered for the duration of the call only, and removed
afterwards on both success and failure — so a failed structured call cannot
leave it advertised on later, unrelated turns.

If the model finishes without calling it, one follow-up prompt asks it to format
what it just said. That single retry is deliberate: models commonly forget the
final call once and almost never twice, so retrying further mostly burns tokens.
