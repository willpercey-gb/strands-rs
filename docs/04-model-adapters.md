# Model Adapters

The `Model` trait abstracts LLM providers into a unified streaming interface. strands-rs ships with an Ollama adapter and is designed for you to bring your own.

## The `Model` Trait

```rust
#[async_trait]
pub trait Model: Send + Sync {
    async fn stream(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
        tool_specs: &[ToolSpec],
    ) -> Result<ModelStream, StrandsError>;
}
```

Implementations normalize provider-specific APIs into the `StreamEvent` protocol:

| Event | Purpose |
|-------|---------|
| `MessageStart` | Start of assistant response |
| `ContentBlockStart` | New text or tool_use block |
| `ContentBlockDelta` | Incremental text or tool input |
| `ContentBlockStop` | Block complete |
| `MessageStop` | Response complete with stop reason |
| `Metadata` | Token usage statistics |

## Ollama Adapter (Built-In)

```rust
use strands_ollama::{OllamaModel, OllamaRequestOptions};

let model = OllamaModel::new("llama3.2")
    .with_host("http://localhost:11434")
    .with_options(OllamaRequestOptions {
        temperature: Some(0.7),
        top_p: Some(0.9),
        top_k: None,
        num_predict: Some(4096),
        seed: None,
    });
```

The Ollama adapter:
- Posts to `/api/chat` with streaming enabled
- Converts strands messages to Ollama format (text, tool_use, tool results)
- Parses newline-delimited JSON response chunks
- Handles tool calls (Ollama sends them as complete objects, not streamed)

## Writing Your Own Adapter

Implement the `Model` trait for any LLM provider:

```rust
use async_trait::async_trait;
use futures::stream;
use strands_core::*;
use strands_core::model::{Model, ModelStream};
use strands_core::types::streaming::*;

struct MyModel { /* ... */ }

#[async_trait]
impl Model for MyModel {
    async fn stream(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
        tool_specs: &[ToolSpec],
    ) -> Result<ModelStream> {
        // 1. Convert messages to your provider's format
        // 2. Make the API call
        // 3. Parse the streaming response into StreamEvents
        // 4. Return as a boxed stream

        let events = vec![
            Ok(StreamEvent::MessageStart { role: Role::Assistant }),
            Ok(StreamEvent::ContentBlockStart {
                index: 0,
                content_type: ContentBlockType::Text,
            }),
            Ok(StreamEvent::ContentBlockDelta {
                index: 0,
                delta: DeltaContent::TextDelta("Hello!".into()),
            }),
            Ok(StreamEvent::ContentBlockStop { index: 0 }),
            Ok(StreamEvent::MessageStop { stop_reason: StopReason::EndTurn }),
        ];

        Ok(Box::pin(stream::iter(events)))
    }
}
```

### Key Implementation Notes

- **Tool calls**: Emit `ContentBlockStart` with `ContentBlockType::ToolUse { tool_use_id, name }`, then `ContentBlockDelta` with `DeltaContent::ToolInputDelta(json_fragment)`, then `ContentBlockStop`
- **Stop reason**: Set to `StopReason::ToolUse` when the model wants to call tools, `StopReason::EndTurn` for final responses
- **Streaming**: If your provider doesn't stream, emit all events at once via `stream::iter`
- **Errors**: Return `StrandsError::Model(message)` for provider errors
