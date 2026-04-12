# Plugins

Plugins bundle hooks and tools into reusable, composable units. They're the recommended way to package cross-cutting concerns like logging, safety, metrics, or domain-specific tool sets.

## Defining a Plugin

```rust
use strands_core::{Plugin, Tool, HookRegistry};
use strands_core::hooks::HookEvent;

struct LoggingPlugin;

impl Plugin for LoggingPlugin {
    fn name(&self) -> &str { "logging" }

    fn register_hooks(&self, registry: &mut HookRegistry) {
        registry.register(|event: &mut HookEvent| {
            match event {
                HookEvent::BeforeModelCall { cycle } => {
                    tracing::info!(cycle, "Model call starting");
                }
                HookEvent::AfterToolCall(e) => {
                    tracing::info!(tool = %e.tool_name, error = e.is_error, "Tool completed");
                }
                _ => {}
            }
        });
    }
}
```

## Plugin with Tools

```rust
struct SafetyPlugin {
    blocked_patterns: Vec<String>,
}

impl Plugin for SafetyPlugin {
    fn name(&self) -> &str { "safety" }

    fn register_hooks(&self, registry: &mut HookRegistry) {
        let patterns = self.blocked_patterns.clone();
        registry.register(move |event: &mut HookEvent| {
            if let HookEvent::BeforeToolCall(e) = event {
                let input_str = serde_json::to_string(&e.input).unwrap_or_default();
                for pattern in &patterns {
                    if input_str.contains(pattern) {
                        e.cancel = true;
                        return;
                    }
                }
            }
        });
    }

    fn tools(&self) -> Vec<Box<dyn Tool>> {
        // Optionally contribute tools
        vec![]
    }
}
```

## Using Plugins

```rust
let agent = Agent::builder()
    .model(model)
    .plugin(LoggingPlugin)
    .plugin(SafetyPlugin {
        blocked_patterns: vec!["rm -rf".into(), "DROP TABLE".into()],
    })
    .tool(my_tool)
    .build()?;
```

Plugins are applied during agent construction. Their hooks are registered before the agent processes any requests.
