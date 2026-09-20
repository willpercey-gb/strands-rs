//! Caller identity round-trip over the bridge protocol.
//!
//! The MCP server is registered at user scope, so one host app serves every
//! Claude session on the machine. Identity is what lets the host tell them
//! apart; without it a question like "what is the user looking at" is answered
//! for whichever project happens to be open, which is the wrong one as often
//! as not.
//!
//! The failure mode is silent in both directions — an unstamped call still
//! succeeds, and a stamped one still succeeds against a host that ignores it —
//! so it is pinned here rather than noticed in the app.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;

use serde_json::{json, Value};
use strands_claude_mcp::{Bridge, BridgeClient, BridgeRequest, CallToolParams, CallerInfo};
use strands_core::tool::{FnTool, ToolOutput};

/// A tool that reports the caller identity exactly as the host received it.
fn echo_caller_tool() -> FnTool {
    FnTool::new(
        "echo_caller",
        "Return the calling session's identity as the host sees it.",
        json!({ "type": "object", "properties": {} }),
        |_input, ctx| {
            // Read before building the future: the future must be 'static and
            // cannot borrow the context.
            let caller = CallerInfo::from_context(ctx);
            async move {
                Ok(ToolOutput::success(
                    serde_json::to_value(caller).unwrap_or(Value::Null),
                ))
            }
        },
    )
}

/// Ports are derived from the server name, so each test needs its own to
/// avoid colliding when the suite runs in parallel.
fn spawn(name: &str) -> (BridgeClient, u16) {
    let bridge = Bridge::builder(name).tool(echo_caller_tool()).build();
    let port = bridge.port();
    bridge.spawn();

    let client = BridgeClient::new(port);
    for _ in 0..50 {
        if client.call(&BridgeRequest::Ping).is_ok() {
            return (client, port);
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    panic!("bridge `{name}` never came up on port {port}");
}

fn call_echo(client: &BridgeClient) -> Value {
    let result = client
        .call(&BridgeRequest::CallTool {
            params: CallToolParams {
                name: "echo_caller".into(),
                arguments: json!({}),
            },
            caller: None, // stamped by BridgeClient::call
        })
        .expect("tool call should succeed");
    assert_eq!(result["is_error"], json!(false), "tool errored: {result}");
    result["content"].clone()
}

#[test]
fn identity_reaches_the_tool() {
    let (client, _) = spawn("test_caller_reaches_tool");
    let seen = call_echo(&client);

    assert_eq!(
        seen["pid"].as_u64(),
        Some(std::process::id() as u64),
        "tool should see this process as the caller, got: {seen}",
    );
    assert!(
        !seen["session"].as_str().unwrap_or("").is_empty(),
        "session id should be non-empty, got: {seen}",
    );
}

#[test]
fn identity_reports_the_callers_working_directory() {
    let (client, _) = spawn("test_caller_cwd");
    let seen = call_echo(&client);

    let expected = std::env::current_dir().unwrap();
    assert_eq!(
        seen["cwd"].as_str().map(std::path::PathBuf::from),
        Some(expected),
        "cwd is the only practical way to tell which project a session is in",
    );
}

#[test]
fn the_same_client_keeps_one_session_id_across_calls() {
    // Each call opens a fresh connection, so the id has to come from the
    // client rather than from connection state. If it were regenerated per
    // call, every tool invocation would look like a different session and
    // grouping by session would be meaningless.
    let (client, _) = spawn("test_caller_stable_session");
    let first = call_echo(&client);
    let second = call_echo(&client);

    assert_eq!(first["session"], second["session"]);
    assert_eq!(
        first["session"].as_str(),
        Some(client.caller().session.as_str()),
    );
}

#[test]
fn distinct_clients_get_distinct_session_ids() {
    let (a, port) = spawn("test_caller_distinct");
    let b = BridgeClient::new(port);
    assert_ne!(a.caller().session, b.caller().session);
}

/// A shim binary older than the host omits `caller` entirely. The shim is a
/// build artifact copied into the app's resources, so a stale copy is a real
/// possibility — and it must degrade to an anonymous call rather than take
/// every tool down with a deserialize error.
#[test]
fn a_request_without_identity_is_accepted_as_anonymous() {
    let (_, port) = spawn("test_caller_legacy_shim");

    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).unwrap();
    // Exactly what a pre-identity shim puts on the wire.
    let legacy = r#"{"method":"call_tool","params":{"name":"echo_caller","arguments":{}}}"#;
    writeln!(stream, "{legacy}").unwrap();
    stream.flush().unwrap();

    let mut line = String::new();
    BufReader::new(&stream).read_line(&mut line).unwrap();

    let response: Value = serde_json::from_str(&line).expect("valid response");
    assert!(
        response.get("error").is_none(),
        "legacy request should not error: {line}",
    );
    assert_eq!(
        response["result"]["is_error"],
        json!(false),
        "legacy request should still invoke the tool: {line}",
    );
    assert_eq!(
        response["result"]["content"],
        Value::Null,
        "an unstamped call should arrive as unknown, not as a fabricated identity",
    );
}

/// End-to-end through the real shim binary, run from a chosen directory.
///
/// The unit tests above prove the protocol carries identity. This proves the
/// shipped shim populates it from its own process, which is the whole basis
/// for telling one Claude session's project from another's.
#[test]
fn the_shim_reports_the_directory_it_was_launched_in() {
    use std::process::{Command, Stdio};

    let bridge = Bridge::builder("test_caller_shim_cwd")
        .tool(echo_caller_tool())
        .build();
    let port = bridge.port();
    bridge.spawn();

    let probe = BridgeClient::new(port);
    for _ in 0..50 {
        if probe.call(&BridgeRequest::Ping).is_ok() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    // Any real directory that is not the test's own cwd, so a shim that
    // reported the wrong thing would be caught rather than coincidentally
    // matching. Canonicalised because /tmp is a symlink on macOS.
    let launch_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .canonicalize()
        .unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_strands-claude-mcp-shim"))
        .args(["--name", "test_caller_shim_cwd", "--port", &port.to_string()])
        .current_dir(&launch_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("shim binary should run");

    let request = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo_caller","arguments":{}}}"#;
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "{request}").unwrap();
    stdin.flush().unwrap();

    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .expect("shim should answer tools/call");
    let _ = child.kill();

    let response: Value = serde_json::from_str(&line).expect("valid JSON-RPC");
    let text = response["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("expected a text content block, got: {line}"));
    let seen: Value = serde_json::from_str(text).expect("tool content should be JSON");

    assert_eq!(
        seen["cwd"].as_str().map(std::path::PathBuf::from),
        Some(launch_dir),
        "the shim must report where it was launched, not where the host runs",
    );
    assert_eq!(
        seen["pid"].as_u64(),
        Some(child.id() as u64),
        "identity should come from the shim process, not the host",
    );
}
