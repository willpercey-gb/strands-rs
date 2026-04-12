# Agent Loop

The agent loop is the core execution engine. It implements a ReAct (Reason + Act) cycle that lets language models perform actions beyond their training.

## How It Works

```
1. REASONING   → Model receives input + conversation history + available tools
2. DECISION    → Model decides: respond directly or call a tool
3. EXECUTION   → If tool requested: execute it, collect results
4. FEEDBACK    → Tool results added to conversation, loop back to step 1
5. TERMINATION → Model produces final response (no more tool calls)
```

Each iteration accumulates context. The model sees the original request plus every tool call and result, enabling multi-step reasoning.

## Stop Reasons

| Reason | Behavior |
|--------|----------|
| `EndTurn` | Normal completion — model finished responding |
| `ToolUse` | Execute requested tools, then continue loop |
| `MaxTokens` | Model hit token limit — loop ends |
| `Cancelled` | External cancel signal received |
| `ContentFiltered` | Safety mechanism blocked the response |
| `GuardrailIntervention` | Policy stopped generation |

## Configuration

```rust
let agent = Agent::builder()
    .model(model)
    .max_cycles(20)           // Safety limit on ReAct iterations (default: 20)
    .retry_config(RetryConfig {
        max_retries: 3,        // Retry model calls on failure
        initial_backoff_ms: 500,
        backoff_multiplier: 2.0,
        max_backoff_ms: 30_000,
    })
    .concurrent_tools(true)   // Execute tools in parallel (default: false)
    .build()?;
```

## Cancellation

Cancel a running agent from another task:

```rust
let cancel_handle = agent.cancel.clone(); // Arc<AtomicBool>

// In another task:
cancel_handle.store(true, std::sync::atomic::Ordering::Relaxed);

// Or use the method:
agent.cancel();
```

Cancellation is cooperative — checked between model calls and before tool execution.

## Error Handling

- **Tool errors** are caught and fed back to the model as error results, allowing it to retry or adjust
- **Model errors** trigger exponential backoff retry (configurable via `RetryConfig`)
- **Max cycles** returns `StrandsError::MaxCycles` if the loop doesn't converge
- **Malformed tool JSON** is caught and fed back as an error result

## Message Flow

**User messages** contain:
- Text input
- Tool results from previous executions

**Assistant messages** contain:
- Text responses
- Tool use requests (name + JSON arguments)

The stream accumulator assembles partial streaming events into complete `ContentBlock` values before adding them to the conversation history.
