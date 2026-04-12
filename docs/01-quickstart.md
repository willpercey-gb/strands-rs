# Quickstart

strands-rs is a Rust port of the AWS Strands Agents SDK — a model-driven framework for building AI agents that can reason and act through tool use.

## Installation

Add the crates to your `Cargo.toml`:

```toml
[dependencies]
strands-core = { path = "../strands-core" }
strands-ollama = { path = "../strands-ollama" }  # or your own adapter
tokio = { version = "1", features = ["full"] }
serde_json = "1"
```

## Your First Agent

```rust
use strands_core::{Agent, ToolOutput, ToolContext, FnTool};
use strands_ollama::OllamaModel;
use serde_json::json;

#[tokio::main]
async fn main() -> strands_core::Result<()> {
    let mut agent = Agent::builder()
        .model(OllamaModel::new("llama3.2"))
        .system_prompt("You are a helpful assistant.")
        .build()?;

    let result = agent.prompt("What is the capital of France?").await?;
    println!("{}", result.text());
    Ok(())
}
```

## Adding Tools

Tools let the agent take actions. Define them as closures or structs:

```rust
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
        Ok(ToolOutput::success(format!("22°C and sunny in {city}")))
    },
);

let mut agent = Agent::builder()
    .model(OllamaModel::new("llama3.2"))
    .tool(weather)
    .system_prompt("You are a helpful assistant with weather access.")
    .build()?;

let result = agent.prompt("What's the weather in London?").await?;
println!("{}", result.text());
```

## Using the `#[tool]` Macro

For cleaner tool definitions, use the proc macro:

```rust
use strands_core::tool;

#[tool]
async fn get_weather(
    /// The city to check weather for
    city: String,
) -> Result<String, strands_core::StrandsError> {
    Ok(format!("22°C and sunny in {city}"))
}

// Use: .tool(GetWeather)
```

## What's Next

- [Agent Loop](02-agent-loop.md) — how the ReAct cycle works
- [Tools](03-tools.md) — defining and using tools
- [Model Adapters](04-model-adapters.md) — connecting to LLM providers
- [Conversation Management](05-conversation-management.md) — managing context windows
- [Hooks](06-hooks.md) — extending agent behavior
- [Multi-Agent](07-multi-agent.md) — swarm and graph patterns
- [Session Management](08-session-management.md) — persisting conversations
- [Plugins](09-plugins.md) — bundling reusable extensions
- [Streaming](10-streaming.md) — real-time event callbacks
