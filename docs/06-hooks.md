# Hooks

Hooks let you observe and modify agent behavior at key lifecycle points. They fire synchronously and receive mutable references to event data, allowing hooks to cancel tools, retry model calls, or override messages.

## Hook Events

| Event | When | Writable Fields |
|-------|------|-----------------|
| `AgentInitialized` | After agent constructed | — |
| `BeforeInvocation` | Start of `prompt()` call | `override_messages` |
| `AfterInvocation` | End of `prompt()` call | `resume` |
| `MessageAdded` | Message added to history | — |
| `BeforeModelCall` | Before model inference | — |
| `AfterModelCall` | After model response | `retry` |
| `BeforeToolCall` | Before tool execution | `cancel` |
| `AfterToolCall` | After tool execution | `retry` |

## Registering Hooks

### Closure-Based

```rust
use strands_core::hooks::HookEvent;

let agent = Agent::builder()
    .model(model)
    .hook(|event: &mut HookEvent| {
        match event {
            HookEvent::BeforeToolCall(e) => {
                println!("Calling tool: {}", e.tool_name);
            }
            HookEvent::AfterToolCall(e) => {
                println!("Tool {} error: {}", e.tool_name, e.is_error);
            }
            _ => {}
        }
    })
    .build()?;
```

### Via Plugins

See [Plugins](09-plugins.md) for bundling hooks into reusable units.

## Common Patterns

### Cancel a Tool Call

```rust
use strands_core::hooks::{HookEvent, events::BeforeToolCallEvent};

agent = Agent::builder()
    .model(model)
    .hook(|event: &mut HookEvent| {
        if let HookEvent::BeforeToolCall(e) = event {
            if e.tool_name == "dangerous_tool" {
                e.cancel = true;  // Tool returns an error result to the model
            }
        }
    })
    .build()?;
```

### Retry a Model Call

```rust
use strands_core::hooks::{HookEvent, events::AfterModelCallEvent};

.hook(|event: &mut HookEvent| {
    if let HookEvent::AfterModelCall(e) = event {
        if e.stop_reason == StopReason::ContentFiltered {
            e.retry = true;  // Re-invoke the model
        }
    }
})
```

### Retry a Tool Call

```rust
.hook(|event: &mut HookEvent| {
    if let HookEvent::AfterToolCall(e) = event {
        if e.is_error {
            e.retry = true;  // Re-execute the same tool
        }
    }
})
```

### Override Messages Before Invocation

```rust
use strands_core::hooks::{HookEvent, events::BeforeInvocationEvent};

.hook(|event: &mut HookEvent| {
    if let HookEvent::BeforeInvocation(e) = event {
        // Filter or modify messages before the agent processes them
        e.override_messages = Some(
            e.messages.iter()
                .filter(|m| !m.text().contains("secret"))
                .cloned()
                .collect()
        );
    }
})
```

### Limit Tool Usage

```rust
use std::sync::{Arc, Mutex};
use strands_core::hooks::HookEvent;

let tool_counts: Arc<Mutex<HashMap<String, usize>>> = Arc::new(Mutex::new(HashMap::new()));
let counts = tool_counts.clone();

.hook(move |event: &mut HookEvent| {
    if let HookEvent::BeforeToolCall(e) = event {
        let mut counts = counts.lock().unwrap();
        let count = counts.entry(e.tool_name.clone()).or_insert(0);
        *count += 1;
        if *count > 3 {
            e.cancel = true;
        }
    }
})
```

## Execution Order

- **Before** hooks fire in registration order
- **After** hooks fire in registration order (reverse ordering planned for future)
