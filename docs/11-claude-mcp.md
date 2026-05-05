# Claude MCP Bridge

`strands-claude-mcp` exposes any set of `strands-core` tools as a Model Context
Protocol server that Claude Code (or any MCP host) can discover and call. It
splits the work into two pieces:

- **An in-process bridge** (`strands_claude_mcp::Bridge`) that holds your tools and
  serves them over a local TCP socket using newline-delimited JSON.
- **A standalone shim binary** (`strands-claude-mcp-shim`) that the MCP host spawns.
  The shim speaks JSON-RPC 2.0 over stdio (the MCP wire format) and forwards
  every `tools/list` and `tools/call` request to the bridge.

This indirection means the host app keeps its database connections, embeddings,
and other in-process state while still exposing tools to Claude Code via the
standard MCP protocol.

## Architecture

```
┌──────────────┐  stdio     ┌──────────────────┐  TCP/JSON   ┌──────────────┐
│  Claude Code │ ─────────> │ strands-claude-mcp-shim │ ──────────> │  Your app    │
│  (MCP host)  │ <───────── │  (small binary)  │ <────────── │   (Bridge)   │
└──────────────┘  JSON-RPC  └──────────────────┘  newlines   └──────────────┘
```

The shim is a thin proxy — no business logic, no state. It exists only because
MCP requires a host-spawnable executable, and your app is typically a
long-running daemon you don't want spawned per call.

## Quickstart

```rust
use strands_core::FnTool;
use strands_claude_mcp::Bridge;

let bridge = Bridge::builder("planner")
    .tool(create_node_tool)
    .tool(create_edge_tool)
    .build();

let port = bridge.port();
bridge.spawn();                       // listens on 127.0.0.1:<port>
strands_claude_mcp::install("planner", port)?; // claude mcp add ...
```

That's the whole setup. Calling `install` runs `claude mcp add planner -s user
-- <shim> --name planner --port <port>` for you, idempotently. The shim is
located by walking from the current executable, the cargo target dir, and
finally `PATH`.

## Bridge

`Bridge::builder(name)` namespaces every tool you add to it: a tool named
`create_node` registered with bridge `planner` becomes `planner__create_node`
in Claude. Multiple host apps can register their own bridges without colliding.

`bridge.spawn()` does the right thing whether you call it from inside an
existing Tokio runtime (uses the current handle) or from synchronous code like
a Tauri `setup` hook (stands up a dedicated multi-threaded runtime on a
worker thread).

The port is hashed from the bridge name (FNV-1a, dynamic range 49152–65535) so
the same name always gets the same port — re-launches don't churn registered
configs. Override with `BridgeBuilder::port(port)` if you ever need to.

## Wire protocol — bridge ↔ shim

Each request and response is a single line of JSON over the TCP connection.

| Request | Response shape |
|---|---|
| `{"method":"ping"}` | `{"result":"pong"}` |
| `{"method":"list_tools"}` | `{"result":[{"name":"...","description":"...","input_schema":{...}}]}` |
| `{"method":"call_tool","params":{"name":"...","arguments":{...}}}` | `{"result":{"content":<value>,"is_error":<bool>}}` or `{"error":"..."}` |

The shim translates these into MCP's `tools/list` and `tools/call` shapes
before writing to stdout. Tool results are wrapped as a single
`{"type":"text","text":"..."}` content block; non-string tool outputs are JSON
serialised before wrapping.

## Shim CLI

```
strands-claude-mcp-shim --name <bridge-name> --port <port>
```

Both arguments are optional. Defaults: name = `strands`, port = hashed from
name. The shim logs to stderr only; stdout is reserved for the JSON-RPC
channel.

## Install / uninstall

`install(name, port)` and `uninstall(name)` shell out to the `claude` CLI.
`install` removes any prior registration with the same name first, so it's
safe to call on every app launch. `uninstall` swallows errors — handy when
you're not sure whether the server was registered in the first place.

## Why not the official Rust MCP SDK?

[`rmcp`](https://crates.io/crates/rmcp) leans on `#[tool]`-style proc macros
that bake the tool list at compile time. `strands-claude-mcp` is intentionally
dynamic: the bridge's tool set comes from whatever `strands_core::Tool`s the
host app added at runtime, and the shim implements the MCP protocol subset by
hand (~150 lines) so the dependency surface stays small and the tool registry
stays mutable.
