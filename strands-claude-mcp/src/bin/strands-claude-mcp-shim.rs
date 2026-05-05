//! Standalone MCP server that Claude Code spawns. Forwards every `tools/list`
//! and `tools/call` request to the host application's in-process bridge over
//! TCP.
//!
//! Speaks JSON-RPC 2.0 over stdio. Logs to stderr only — stdout is the
//! protocol channel and any stray bytes break the host parser.

use std::io::{self, BufRead, BufReader, Write};

use serde_json::{json, Value};
use strands_claude_mcp::{BridgeClient, BridgeRequest, CallToolParams, MCP_PROTOCOL_VERSION};

fn main() {
    let args = parse_args();

    eprintln!(
        "strands-claude-mcp-shim starting: name={} port={}",
        args.name, args.port
    );

    let client = BridgeClient::new(args.port);

    // Best-effort handshake. If the bridge isn't up, we'll surface the failure
    // when the first real call comes in. Keep startup non-blocking.
    if let Err(e) = client.call(&BridgeRequest::Ping) {
        eprintln!("strands-claude-mcp-shim: bridge ping failed: {e}");
    }

    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut reader = BufReader::new(stdin.lock());
    let mut line = String::new();

    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break, // host closed stdin
            Ok(_) => {}
            Err(e) => {
                eprintln!("strands-claude-mcp-shim: stdin read error: {e}");
                break;
            }
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("strands-claude-mcp-shim: invalid JSON-RPC: {e}");
                continue;
            }
        };
        if let Some(response) = dispatch(&msg, &args, &client) {
            if let Err(e) = writeln!(stdout, "{response}") {
                eprintln!("strands-claude-mcp-shim: stdout write failed: {e}");
                break;
            }
            let _ = stdout.flush();
        }
    }
    eprintln!("strands-claude-mcp-shim: exiting");
}

#[derive(Debug, Clone)]
struct Args {
    name: String,
    port: u16,
}

fn parse_args() -> Args {
    let mut name = String::new();
    let mut port: u16 = 0;

    let mut iter = std::env::args().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--name" => {
                if let Some(v) = iter.next() {
                    name = v;
                }
            }
            "--port" => {
                if let Some(v) = iter.next() {
                    port = v.parse().unwrap_or(0);
                }
            }
            _ => {}
        }
    }
    if name.is_empty() {
        name = "strands".to_string();
    }
    if port == 0 {
        port = strands_claude_mcp::port_for(&name);
    }
    Args { name, port }
}

/// Returns Some(serialized response) for requests, None for notifications.
fn dispatch(msg: &Value, args: &Args, client: &BridgeClient) -> Option<String> {
    let id = msg.get("id").cloned();
    let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");

    // Notifications have no `id` and don't get a response.
    let is_notification = id.is_none();

    let result: Result<Value, (i64, String)> = match method {
        "initialize" => Ok(initialize_result(args)),
        "initialized" | "notifications/initialized" => {
            // No response.
            return None;
        }
        "ping" => Ok(json!({})),
        "tools/list" => list_tools(client),
        "tools/call" => call_tool(msg, client),
        _ => Err((-32601, format!("method not found: {method}"))),
    };

    if is_notification {
        return None;
    }

    let envelope = match result {
        Ok(value) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": value,
        }),
        Err((code, message)) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": code, "message": message },
        }),
    };
    Some(envelope.to_string())
}

fn initialize_result(args: &Args) -> Value {
    json!({
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": args.name,
            "version": env!("CARGO_PKG_VERSION"),
        }
    })
}

fn list_tools(client: &BridgeClient) -> Result<Value, (i64, String)> {
    let descriptors = client
        .call(&BridgeRequest::ListTools)
        .map_err(|e| (-32603, e))?;

    // descriptors is a Vec<ToolDescriptor>; remap into MCP shape.
    let arr = descriptors.as_array().cloned().unwrap_or_default();
    let mcp_tools: Vec<Value> = arr
        .into_iter()
        .map(|d| {
            json!({
                "name": d.get("name").cloned().unwrap_or(Value::Null),
                "description": d.get("description").cloned().unwrap_or(Value::Null),
                "inputSchema": d.get("input_schema").cloned().unwrap_or(json!({"type": "object"})),
            })
        })
        .collect();
    Ok(json!({ "tools": mcp_tools }))
}

fn call_tool(msg: &Value, client: &BridgeClient) -> Result<Value, (i64, String)> {
    let params = msg.get("params").ok_or((-32602, "missing params".into()))?;
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "missing tool name".into()))?
        .to_string();
    let arguments = params.get("arguments").cloned().unwrap_or(Value::Null);

    let result_value = client
        .call(&BridgeRequest::CallTool {
            params: CallToolParams { name, arguments },
        })
        .map_err(|e| (-32603, e))?;

    let is_error = result_value
        .get("is_error")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let content = result_value.get("content").cloned().unwrap_or(Value::Null);

    // Pack the JSON content into MCP's text content block. If the tool
    // returned a string we pass it through unquoted; otherwise we serialize.
    let text = match content {
        Value::String(s) => s,
        other => serde_json::to_string(&other).unwrap_or_else(|_| "<unserializable>".to_string()),
    };

    Ok(json!({
        "content": [{
            "type": "text",
            "text": text,
        }],
        "isError": is_error,
    }))
}
