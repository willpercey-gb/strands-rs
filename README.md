# strands-rs

A Rust port of the [AWS Strands Agents SDK](https://github.com/strands-agents/sdk-python) — a model-driven framework for building AI agents that reason and act through tool use.

## Features

- **ReAct Agent Loop** — model reasoning + tool execution in a recursive cycle
- **Extensible Model Adapters** — bring your own LLM via the `Model` trait; Ollama adapter included
- **Flexible Tool System** — `#[tool]` proc macro, closure-based `FnTool`, or manual trait impl
- **Multi-Agent Orchestration** — Swarm (autonomous handoffs), Graph (deterministic DAG), and Agents-as-Tools
- **Conversation Management** — sliding window, LLM-powered summarization, or custom strategies
- **Mutable Hook System** — cancel tools, retry model calls, override messages at lifecycle points
- **Session Persistence** — file-based or pluggable backend via `SessionRepository`
- **Streaming** — real-time callback handler for text deltas and tool events
- **Plugins** — bundle hooks + tools into reusable units
- **Concurrent Tool Execution** — sequential or parallel tool dispatch

## Quickstart

```rust
use strands_core::{Agent, FnTool, ToolOutput, ToolContext};
use strands_ollama::OllamaModel;
use serde_json::json;

#[tokio::main]
async fn main() -> strands_core::Result<()> {
    let weather = FnTool::new(
        "get_weather",
        "Get the current weather for a city",
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"]
        }),
        |input: serde_json::Value, _ctx: &ToolContext| async move {
            let city = input["city"].as_str().unwrap_or("unknown");
            Ok(ToolOutput::success(format!("22C and sunny in {city}")))
        },
    );

    let mut agent = Agent::builder()
        .model(OllamaModel::new("llama3.2"))
        .tool(weather)
        .system_prompt("You are a helpful assistant with weather access.")
        .build()?;

    let result = agent.prompt("What's the weather in London?").await?;
    println!("{}", result.text());
    Ok(())
}
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| `strands-core` | Agent loop, tools, hooks, conversation/session management, multi-agent, plugins |
| `strands-ollama` | Ollama model adapter (`/api/chat` with streaming + tool calling) |
| `strands-macros` | `#[tool]` proc macro for ergonomic tool definition |

## Multi-Agent Example

```rust
use strands_core::multiagent::Swarm;

let swarm = Swarm::builder()
    .agent("researcher", "Researches topics", researcher)
    .agent("coder", "Writes code", coder)
    .agent("reviewer", "Reviews code", reviewer)
    .entry_point("researcher")
    .max_handoffs(20)
    .build()?;

let result = swarm.run("Build a REST API for a todo app").await?;
println!("{}", result.output);
```

## Writing Your Own Model Adapter

Implement the `Model` trait to connect any LLM:

```rust
#[async_trait]
impl Model for MyModel {
    async fn stream(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
        tool_specs: &[ToolSpec],
    ) -> Result<ModelStream, StrandsError> {
        // Convert messages, call your API, return StreamEvents
    }
}
```

## Documentation

Full user guide in [`docs/`](docs/):

1. [Quickstart](docs/01-quickstart.md)
2. [Agent Loop](docs/02-agent-loop.md)
3. [Tools](docs/03-tools.md)
4. [Model Adapters](docs/04-model-adapters.md)
5. [Conversation Management](docs/05-conversation-management.md)
6. [Hooks](docs/06-hooks.md)
7. [Multi-Agent](docs/07-multi-agent.md)
8. [Session Management](docs/08-session-management.md)
9. [Plugins](docs/09-plugins.md)
10. [Streaming](docs/10-streaming.md)

## License

Apache-2.0
