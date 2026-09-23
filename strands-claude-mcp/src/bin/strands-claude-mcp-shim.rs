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
        "initialize" => Ok(initialize_result(args, client)),
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

fn initialize_result(args: &Args, client: &BridgeClient) -> Value {
    let mut result = json!({
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": args.name,
            "version": env!("CARGO_PKG_VERSION"),
        }
    });

    // Server instructions, when the host set any. Fetched rather than passed
    // as argv so the host can change the text without re-registering the
    // server — the registration only carries `--name` and `--port`.
    //
    // A bridge that is unreachable or too old to know the request simply
    // yields no instructions: `initialize` must still succeed, or the client
    // loses the whole server over an optional field.
    if let Ok(value) = client.call(&BridgeRequest::Instructions) {
        if let Some(text) = value.as_str().filter(|t| !t.trim().is_empty()) {
            result["instructions"] = json!(text);
        }
    }

    result
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
            let mut tool = json!({
                "name": d.get("name").cloned().unwrap_or(Value::Null),
                "description": d.get("description").cloned().unwrap_or(Value::Null),
                "inputSchema": d.get("input_schema").cloned().unwrap_or(json!({"type": "object"})),
            });

            // Only emit the optional fields when present: MCP treats a missing
            // annotation as "unknown", which is not the same as a false one.
            if let Some(map) = tool.as_object_mut() {
                if let Some(schema) = d.get("output_schema") {
                    if !schema.is_null() {
                        map.insert("outputSchema".to_string(), schema.clone());
                    }
                }
                if let Some(annotations) = d.get("annotations") {
                    if !annotations.is_null() {
                        map.insert("annotations".to_string(), annotations.clone());
                    }
                }
            }
            tool
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
            // Filled in by `BridgeClient::call` from this process's identity.
            caller: None,
        })
        .map_err(|e| (-32603, e))?;

    let is_error = result_value
        .get("is_error")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let content = result_value.get("content").cloned().unwrap_or(Value::Null);

    // A tool that produced an image can have it returned as one, by including
    // `image: {path, mime}` in its result. The client is multimodal, so an
    // attached image is looked at, where a path is only read if the model
    // decides to.
    //
    // The shim reads the file itself: it runs on the same host as the tool,
    // so the bytes never cross the bridge and a tool only has to name a file
    // it has already written.
    let mut content = content;
    let mut image_block = None;
    let mut image_problem = None;
    if let Some(spec) = content
        .as_object_mut()
        .and_then(|o| o.remove("image"))
        .filter(|v| !v.is_null())
    {
        match inline_image(&spec) {
            Ok(block) => image_block = Some(block),
            Err(why) => image_problem = Some(why),
        }
    }

    // Pack the JSON content into MCP's text content block. If the tool
    // returned a string we pass it through unquoted; otherwise we serialize.
    let mut text = match content {
        Value::String(s) => s,
        other => serde_json::to_string(&other).unwrap_or_else(|_| "<unserializable>".to_string()),
    };
    if let Some(why) = image_problem {
        text.push_str(&format!("\n(the image was not attached: {why})"));
    }

    let mut blocks = vec![json!({ "type": "text", "text": text })];
    blocks.extend(image_block);

    Ok(json!({
        "content": blocks,
        "isError": is_error,
    }))
}

/// Biggest image worth attaching.
///
/// Beyond this an attachment costs more context than it earns, and the path
/// stays in the text block for a caller that wants it.
const MAX_INLINE_IMAGE: u64 = 4 * 1024 * 1024;

/// Turn `{"path": "...", "mime": "image/png"}` into an MCP image block.
///
/// Errors are returned as a sentence rather than failing the call: the tool
/// did its work, and losing the result because the picture could not be
/// attached would be a worse trade than saying so.
fn inline_image(spec: &Value) -> Result<Value, String> {
    use base64::Engine as _;

    let path = spec
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| "no `path` in the image".to_string())?;
    let mime = spec
        .get("mime")
        .and_then(Value::as_str)
        .unwrap_or("image/png");

    let size = std::fs::metadata(path)
        .map_err(|e| format!("cannot stat {path}: {e}"))?
        .len();
    if size > MAX_INLINE_IMAGE {
        return Err(format!(
            "{size} bytes is over the {MAX_INLINE_IMAGE}-byte inline limit; read {path} instead"
        ));
    }

    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    Ok(json!({
        "type": "image",
        "data": base64::engine::general_purpose::STANDARD.encode(bytes),
        "mimeType": mime,
    }))
}

#[cfg(test)]
mod image_tests {
    use super::*;

    fn temp_file(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("strands-shim-{}-{name}", std::process::id()));
        std::fs::write(&path, bytes).expect("write temp file");
        path
    }

    #[test]
    fn a_file_becomes_a_base64_image_block() {
        let path = temp_file("ok.png", b"\x89PNG\r\n\x1a\n");
        let block = inline_image(&json!({ "path": path, "mime": "image/png" })).expect("block");
        assert_eq!(block["type"], "image");
        assert_eq!(block["mimeType"], "image/png");
        assert_eq!(block["data"], "iVBORw0KGgo=");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn the_mime_type_defaults_to_png() {
        let path = temp_file("default.png", b"x");
        let block = inline_image(&json!({ "path": path })).expect("block");
        assert_eq!(block["mimeType"], "image/png");
        std::fs::remove_file(path).ok();
    }

    /// Failures come back as a sentence so the caller can keep the tool's own
    /// result; none of them should panic or fail the call.
    #[test]
    fn problems_are_described_not_raised() {
        assert!(inline_image(&json!({})).unwrap_err().contains("no `path`"));
        let missing = inline_image(&json!({ "path": "/nope/not/here.png" })).unwrap_err();
        assert!(missing.contains("cannot stat"), "{missing}");
    }

    #[test]
    fn an_oversized_file_is_left_as_a_path() {
        let path = temp_file("big.png", &vec![0u8; (MAX_INLINE_IMAGE + 1) as usize]);
        let err = inline_image(&json!({ "path": path })).unwrap_err();
        assert!(err.contains("inline limit"), "{err}");
        std::fs::remove_file(path).ok();
    }
}
