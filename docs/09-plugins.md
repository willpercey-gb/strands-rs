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

## Ready-Made Plugins

`strands-core` ships four plugins for common patterns.

### Skills

Reusable capability definitions injected into the system prompt, so the set can
change without touching agent code.

```rust,ignore
use strands_core::plugin::{Skill, SkillSet};

let skills = SkillSet::new()
    .with_skill(Skill::new("search", "Find things", "Prefer search over guessing."))
    .with_skill(Skill::new("cite", "Attribute claims", "Cite every factual claim."));

let prompt = skills.inject(Some(&existing_prompt));
```

Injection lands **before** the first cache point, keeping skill text inside the
cacheable prefix. Appending after it would invalidate the cache every turn,
which for a long skill set costs more than the skills are worth.

### Context Offloader

Moves oversized tool results into [storage](18-memory-and-storage.md), leaving a
reference and a preview:

```rust,ignore
use strands_core::plugin::ContextOffloader;

let offloader = ContextOffloader::new(storage.clone()).with_threshold(8_000);

if offloader.should_offload(tool_name, &text) {
    let replacement = offloader.offload(tool_name, tool_use_id, &text).await?;
}

// Give the model a way to read it back.
agent_builder.tool(offloader.retrieval_tool());
```

Retrieval supports grep-with-context, not just a blob read — returning a 10 MB
document in one piece would re-create the problem offloading solved.

Size is not the only reason to offload; `with_should_offload` takes a predicate
for results that are small but sensitive, or large but needed in full.

### Context Injector

Adds retrieved documents, memory recall or environment facts to the prompt:

```rust,ignore
use strands_core::plugin::{ContextInjector, InjectedContent, InjectionPlacement};

let injector = ContextInjector::new()
    .with_content(InjectedContent::new("memory", "User prefers metric units."))
    .with_placement(InjectionPlacement::AfterCachePoint);
```

Content is tagged with its source. Unlabelled injected text reads to the model
as its own instructions, which is how retrieved content ends up being followed
as though it were a directive.

Placement defaults to *after* cache points, because per-turn content must not
invalidate the cached prefix. Stable content can opt into `BeforeCachePoint`.

### Goal Loop

Keeps an agent working until an objective is actually met, rather than until the
model decides it is done:

```rust,ignore
use strands_core::plugin::{ContainsJudge, GoalLoop};

let goal = GoalLoop::new("write a report with a summary", ContainsJudge::new(["summary"]))
    .with_max_attempts(3);

let outcome = goal.run(|prompt| async { Ok(agent.prompt(&prompt).await?.text()) }).await?;
```

Attempts are capped: an unreachable goal would otherwise burn the whole token
budget discovering it was unreachable. Each retry restates the goal alongside
the shortfall, since feedback alone loses the objective once the conversation
has been trimmed.
