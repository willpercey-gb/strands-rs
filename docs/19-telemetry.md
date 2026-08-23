# Telemetry

Per-cycle and per-tool metrics for an invocation.

## Reading them

```rust,ignore
let result = agent.prompt("...").await?;

println!("{} cycles", result.telemetry.cycle_count());
println!("{:?} tokens", result.telemetry.total_usage.total());
println!("{}", result.telemetry);        // human-readable summary
```

## Tool metrics

Cycle metrics come from the loop. Tool metrics come from `AfterToolCall`, which
carries each call's measured duration:

```rust,ignore
let collector = Arc::new(Mutex::new(MetricsCollector::new()));
let sink = collector.clone();

let agent = Agent::builder()
    .model(model)
    .hook(move |event: &mut HookEvent| {
        if let HookEvent::AfterToolCall(e) = event {
            sink.lock().unwrap().record_tool(&e.tool_name, e.duration, e.is_error);
        }
    })
    .build()?;
```

`ToolMetrics::error_rate()` returns `None` before any call rather than `0.0` —
reporting zero would make a tool that never ran look reliable.

## Tracing

strands-rs is instrumented with the `tracing` crate rather than binding
OpenTelemetry directly. An OTel exporter plugs into `tracing`, so this avoids
forcing a heavy dependency tree on users who export nothing.

The GenAI semantic-convention attribute names are exported as constants, so a
subscriber can map onto OTel without guessing:

```rust,ignore
use strands_core::telemetry::attributes;

span.record(attributes::USAGE_INPUT_TOKENS, tokens);
span.record(attributes::TOOL_NAME, name);
```

## Redaction

**Span content is redacted by default.** Tool arguments and results routinely
carry secrets and personal data, so exporting them has to be a deliberate
choice:

```rust,ignore
use strands_core::telemetry::RedactionPolicy;

let policy = RedactionPolicy::default();     // everything redacted
let policy = RedactionPolicy::verbose();     // only where the backend is as trusted as the agent's inputs
```

Redaction substitutes `[redacted]` rather than dropping the field: an absent
attribute is ambiguous, whereas a marker says the field existed and was withheld.
