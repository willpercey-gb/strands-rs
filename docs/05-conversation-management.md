# Conversation Management

As conversations grow, you need strategies to manage context window limits. strands-rs provides the `ConversationManager` trait with three built-in implementations.

## Built-In Managers

### NullConversationManager

Does nothing. Useful for short conversations or testing.

```rust
use strands_core::conversation::NullConversationManager;

let agent = Agent::builder()
    .model(model)
    .conversation_manager(NullConversationManager)
    .build()?;
```

### SlidingWindowConversationManager (Default)

Keeps a fixed window of recent messages, dropping the oldest when the limit is exceeded.

```rust
use strands_core::conversation::SlidingWindowConversationManager;

let agent = Agent::builder()
    .model(model)
    .conversation_manager(SlidingWindowConversationManager::new(20))
    .build()?;
```

Default window size is 40 messages.

### SummarizingConversationManager

Uses the model to summarize older messages instead of discarding them. Preserves context while staying within token limits.

```rust
use std::sync::Arc;
use strands_core::conversation::SummarizingConversationManager;

let summarizer = SummarizingConversationManager::new(Arc::new(model))
    .with_window_size(40)           // Trigger summarization above this count
    .with_preserve_recent(10)       // Always keep last N messages verbatim
    .with_summary_ratio(0.3);       // Summarize 30% of older messages

let agent = Agent::builder()
    .model(main_model)
    .conversation_manager(summarizer)
    .build()?;
```

The summarizer:
1. Triggers when message count exceeds `window_size`
2. Preserves the most recent `preserve_recent` messages
3. Summarizes `summary_ratio` fraction of older messages via the model
4. Replaces summarized messages with a `[Previous conversation summary]` message

## Custom Conversation Managers

Implement the `ConversationManager` trait:

```rust
use async_trait::async_trait;
use strands_core::conversation::ConversationManager;
use strands_core::{Message, StrandsError};

struct MyManager;

#[async_trait]
impl ConversationManager for MyManager {
    async fn reduce_context(
        &self,
        messages: &mut Vec<Message>,
        system_prompt: Option<&str>,
    ) -> Result<(), StrandsError> {
        // Your strategy here
        // Called before each model invocation
        if messages.len() > 10 {
            messages.drain(..messages.len() - 10);
        }
        Ok(())
    }
}
```

The trait is `async` so managers can perform I/O (e.g., calling a model to summarize).
