# Tools

Tools let agents take actions — call APIs, read files, execute code, or interact with external systems. When the model decides it needs a tool, it specifies the tool name and arguments. The agent executes the tool and feeds the result back.

## Three Ways to Define Tools

### 1. `#[tool]` Proc Macro (Recommended)

The simplest approach. Decorate an async function:

```rust
use strands_core::tool;

/// Get the current weather for a city.
#[tool]
async fn get_weather(
    /// The city to check weather for
    city: String,
    /// Temperature unit (celsius or fahrenheit)
    unit: Option<String>,
) -> Result<String, strands_core::StrandsError> {
    let unit = unit.unwrap_or_else(|| "celsius".into());
    Ok(format!("22 degrees {unit} in {city}"))
}

// Creates a `GetWeather` struct implementing `Tool`.
// Use it: Agent::builder().tool(GetWeather).build()?;
```

The macro automatically:
- Generates a struct named in PascalCase (`get_weather` → `GetWeather`)
- Extracts the function doc comment as the tool description
- Extracts parameter doc comments as parameter descriptions
- Builds JSON Schema from Rust types (`String` → `"string"`, `i64` → `"integer"`, etc.)
- Handles `Option<T>` as non-required parameters

### 2. `FnTool` (Closure-Based)

For tools defined inline without a dedicated struct:

```rust
use strands_core::{FnTool, ToolOutput, ToolContext};
use serde_json::json;

let calculator = FnTool::new(
    "calculate",
    "Evaluate a math expression",
    json!({
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate"
            }
        },
        "required": ["expression"]
    }),
    |input: serde_json::Value, _ctx: &ToolContext| async move {
        let expr = input["expression"].as_str().unwrap_or("");
        // ... evaluate expression ...
        Ok(ToolOutput::success(json!(42)))
    },
);
```

### 3. Manual `Tool` Trait Implementation

For maximum control:

```rust
use async_trait::async_trait;
use strands_core::{Tool, ToolSpec, ToolOutput, ToolContext, StrandsError};
use serde_json::{json, Value};

struct DatabaseQuery {
    connection_string: String,
}

#[async_trait]
impl Tool for DatabaseQuery {
    fn name(&self) -> &str { "query_database" }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "query_database".into(),
            description: "Run a SQL query".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "sql": { "type": "string", "description": "SQL query" }
                },
                "required": ["sql"]
            }),
        }
    }

    async fn invoke(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, StrandsError> {
        let sql = input["sql"].as_str().unwrap_or("");
        // ... execute query using self.connection_string ...
        Ok(ToolOutput::success(json!({"rows": []})))
    }
}
```

## Tool Output

```rust
// Success
ToolOutput::success(json!("result text"))
ToolOutput::success(json!({"key": "value"}))

// Error — fed back to the model so it can retry
ToolOutput::error("Something went wrong")
```

## Tool Context

The `ToolContext` provides shared state accessible across tools within an invocation:

```rust
let tool = FnTool::new("my_tool", "...", schema, |input, ctx: &ToolContext| async move {
    // Access shared state
    let user_id = ctx.state["user_id"].as_str();
    Ok(ToolOutput::success("done"))
});
```

## Tool Execution Modes

### Sequential (Default)

Tools execute one at a time. Hooks fire before/after each tool.

### Concurrent

Enable with `.concurrent_tools(true)` on the builder. All tool calls from a single model response execute in parallel via `tokio::join_all`.

```rust
let agent = Agent::builder()
    .model(model)
    .tool(tool_a)
    .tool(tool_b)
    .concurrent_tools(true)
    .build()?;
```
