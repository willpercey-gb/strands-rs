# Streaming

strands-rs supports real-time streaming of model responses via callback handlers, giving you access to text deltas and tool call events as they arrive.

## Callback Handler

Set a callback handler on the agent to receive `StreamEvent`s in real-time:

```rust
use strands_core::{Agent, StreamEvent, CallbackHandler};
use strands_core::types::streaming::DeltaContent;

let agent = Agent::builder()
    .model(model)
    .callback_handler(|event: &StreamEvent| {
        match event {
            StreamEvent::ContentBlockDelta { delta: DeltaContent::TextDelta(text), .. } => {
                print!("{text}");  // Print text as it streams
            }
            StreamEvent::ContentBlockStart { content_type, .. } => {
                if let strands_core::types::streaming::ContentBlockType::ToolUse { name, .. } = content_type {
                    println!("\n[Calling tool: {name}]");
                }
            }
            StreamEvent::MessageStop { stop_reason } => {
                println!("\n[Done: {stop_reason:?}]");
            }
            _ => {}
        }
    })
    .build()?;
```

The callback fires for every event from the model stream, before the event is processed by the agent loop. This enables:
- Real-time UI updates as text streams in
- Early detection of tool usage
- Progress indicators
- Token counting

## Stream Events

| Event | When |
|-------|------|
| `MessageStart` | Model begins responding |
| `ContentBlockStart` | New text or tool_use block starts |
| `ContentBlockDelta` | Incremental content (text fragment or tool input JSON) |
| `ContentBlockStop` | Block is complete |
| `MessageStop` | Model finished with a stop reason |
| `Metadata` | Token usage statistics |

## Custom CallbackHandler Trait

For stateful handlers, implement the trait:

```rust
use strands_core::agent::callback::CallbackHandler;
use strands_core::StreamEvent;
use std::sync::atomic::{AtomicUsize, Ordering};

struct TokenCounter {
    chunks: AtomicUsize,
}

impl CallbackHandler for TokenCounter {
    fn on_stream_event(&self, event: &StreamEvent) {
        if matches!(event, StreamEvent::ContentBlockDelta { .. }) {
            self.chunks.fetch_add(1, Ordering::Relaxed);
        }
    }
}

let counter = TokenCounter { chunks: AtomicUsize::new(0) };

let agent = Agent::builder()
    .model(model)
    .callback_handler(counter)
    .build()?;
```
