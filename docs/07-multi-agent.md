# Multi-Agent Patterns

strands-rs supports three multi-agent orchestration patterns for complex tasks that benefit from specialized agents working together.

## 1. Agents as Tools

The simplest pattern. Wrap an agent as a tool and give it to another agent:

```rust
let researcher = Agent::builder()
    .model(model.clone())
    .system_prompt("You are a research specialist. Find and summarize information.")
    .build()?;

let writer = Agent::builder()
    .model(model.clone())
    .system_prompt("You are a writer. Create polished content from research.")
    .build()?;

let orchestrator = Agent::builder()
    .model(model.clone())
    .system_prompt("You coordinate research and writing tasks.")
    .tool(researcher.as_tool("research", "Research a topic and return findings"))
    .tool(writer.as_tool("write", "Write polished content given research notes"))
    .build()?;

let result = orchestrator.prompt("Write an article about Rust's ownership model").await?;
```

The sub-agent receives a `prompt` parameter and returns its text response.

## 2. Swarm

A collaborative system where agents hand off work to each other autonomously. Each agent sees the full task context, previous agents' work, and available peers.

```rust
use strands_core::multiagent::Swarm;
use std::time::Duration;

let researcher = Agent::builder()
    .model(model.clone())
    .system_prompt("You are a research specialist...")
    .build()?;

let coder = Agent::builder()
    .model(model.clone())
    .system_prompt("You are a coding specialist...")
    .build()?;

let reviewer = Agent::builder()
    .model(model.clone())
    .system_prompt("You are a code review specialist...")
    .build()?;

let swarm = Swarm::builder()
    .agent("researcher", "Researches topics and gathers information", researcher)
    .agent("coder", "Writes and implements code", coder)
    .agent("reviewer", "Reviews code for quality and correctness", reviewer)
    .entry_point("researcher")
    .max_handoffs(20)
    .max_iterations(20)
    .execution_timeout(Duration::from_secs(900))
    .node_timeout(Duration::from_secs(300))
    .repetitive_handoff_detection(8, 3)  // window=8, min_unique=3
    .build()?;

let result = swarm.run("Design and implement a REST API for a todo app").await?;

println!("Status: {:?}", result.status);
println!("Output: {}", result.output);
println!("Agents used: {:?}", result.execution_order);
println!("Total iterations: {}", result.execution_count);
```

### How Handoffs Work

Agents include `HANDOFF_TO: <agent_name> | <message>` in their response to transfer control. The swarm parses this and routes to the target agent with accumulated context including:
- Original task
- Previous agents' contributions
- Shared knowledge from all prior executions

### Safety Mechanisms

| Config | Purpose | Default |
|--------|---------|---------|
| `max_handoffs` | Cap total agent-to-agent transfers | 20 |
| `max_iterations` | Cap total agent executions | 20 |
| `execution_timeout` | Total swarm runtime limit | 15 min |
| `node_timeout` | Per-agent execution limit | 5 min |
| `repetitive_handoff_detection` | Detect ping-pong loops | Disabled |

## 3. Graph

A deterministic directed graph where agents execute according to edge dependencies. Output from one node propagates as input to the next.

```rust
use strands_core::multiagent::{Graph, GraphBuilder};
use std::time::Duration;

let researcher = Agent::builder()
    .model(model.clone())
    .system_prompt("Research the topic thoroughly.")
    .build()?;

let writer = Agent::builder()
    .model(model.clone())
    .system_prompt("Write an article based on the research.")
    .build()?;

let reviewer = Agent::builder()
    .model(model.clone())
    .system_prompt("Review the article for quality.")
    .build()?;

// Sequential pipeline: Research → Write → Review
let graph = Graph::builder()
    .node("researcher", researcher)
    .node("writer", writer)
    .node("reviewer", reviewer)
    .edge("researcher", "writer")
    .edge("writer", "reviewer")
    .entry_point("researcher")
    .build()?;

let result = graph.run("Write an article about async Rust").await?;
```

### Conditional Edges

Route execution based on results:

```rust
use strands_core::multiagent::graph::GraphState;

let graph = Graph::builder()
    .node("classifier", classifier_agent)
    .node("tech_writer", tech_agent)
    .node("business_writer", business_agent)
    .conditional_edge("classifier", "tech_writer", |state: &GraphState| {
        state.results.get("classifier")
            .and_then(|r| r.result.as_ref())
            .map(|r| r.text().contains("technical"))
            .unwrap_or(false)
    })
    .conditional_edge("classifier", "business_writer", |state: &GraphState| {
        state.results.get("classifier")
            .and_then(|r| r.result.as_ref())
            .map(|r| r.text().contains("business"))
            .unwrap_or(false)
    })
    .build()?;
```

### Topologies

**Sequential Pipeline:**
```
Research → Analysis → Review → Report
```

**Parallel with Aggregation:**
```
        → Worker1 →
Coordinator → Worker2 → Aggregator
        → Worker3 →
```

**Feedback Loop (cyclic):**
```rust
let graph = Graph::builder()
    .node("draft", drafter)
    .node("review", reviewer)
    .node("publish", publisher)
    .edge("draft", "review")
    .conditional_edge("review", "draft", |state| {
        // Loop back if review says needs revision
        state.results.get("review")
            .and_then(|r| r.result.as_ref())
            .map(|r| r.text().contains("needs revision"))
            .unwrap_or(false)
    })
    .conditional_edge("review", "publish", |state| {
        state.results.get("review")
            .and_then(|r| r.result.as_ref())
            .map(|r| r.text().contains("approved"))
            .unwrap_or(false)
    })
    .max_node_executions(10)  // Safety limit for cycles
    .reset_on_revisit(true)   // Clear agent state on revisit
    .build()?;
```

### Graph Configuration

| Config | Purpose | Default |
|--------|---------|---------|
| `max_node_executions` | Total execution limit (critical for cycles) | 50 |
| `execution_timeout` | Total graph runtime limit | 15 min |
| `node_timeout` | Per-node execution limit | 5 min |
| `reset_on_revisit` | Clear agent messages on revisit | false |

### Auto-Detection

If no entry points are specified, the graph automatically detects source nodes (nodes with no incoming edges).

## Results

All multi-agent patterns return `MultiAgentResult`:

```rust
let result = swarm.run("task").await?;

println!("Status: {:?}", result.status);       // Completed, Failed, etc.
println!("Output: {}", result.output);          // Final text from last node
println!("Time: {:?}", result.execution_time);  // Total duration
println!("Steps: {}", result.execution_count);  // Total node executions
println!("Order: {:?}", result.execution_order); // Node execution sequence

// Per-node results
for (node_id, node_result) in &result.results {
    println!("{}: {:?} ({:?})", node_id, node_result.status, node_result.execution_time);
}
```

Status values: `Completed`, `Failed`, `Cancelled`, `MaxStepsReached`, `TimedOut`.
