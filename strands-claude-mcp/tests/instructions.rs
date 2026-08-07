//! Server `instructions` round-trip over the bridge protocol.
//!
//! The shim asks the bridge for instructions during MCP `initialize`, so a
//! break here is silent: `initialize` still succeeds and the client simply
//! never sees the guidance. That failure mode is invisible in the host app,
//! which is why it is pinned here.

use strands_claude_mcp::{Bridge, BridgeClient, BridgeRequest};

/// Ports are derived from the server name, so each test needs its own to
/// avoid colliding when the suite runs in parallel.
fn spawn(name: &str, instructions: Option<&str>) -> BridgeClient {
    let mut builder = Bridge::builder(name);
    if let Some(text) = instructions {
        builder = builder.instructions(text);
    }
    let bridge = builder.build();
    let port = bridge.port();
    bridge.spawn();

    // The listener binds on a background runtime; retry briefly rather than
    // sleeping a fixed amount.
    let client = BridgeClient::new(port);
    for _ in 0..50 {
        if client.call(&BridgeRequest::Ping).is_ok() {
            return client;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    panic!("bridge `{name}` never came up on port {port}");
}

#[test]
fn instructions_round_trip() {
    let client = spawn("test_instructions_set", Some("Reach for these tools when X."));
    let value = client
        .call(&BridgeRequest::Instructions)
        .expect("instructions request should succeed");
    assert_eq!(value.as_str(), Some("Reach for these tools when X."));
}

#[test]
fn absent_instructions_yield_an_empty_string_not_an_error() {
    // The shim treats empty as "omit the field", so this has to be a
    // successful call rather than a failure — an error here would abort
    // `initialize` over an optional field.
    let client = spawn("test_instructions_unset", None);
    let value = client
        .call(&BridgeRequest::Instructions)
        .expect("request should succeed even with no instructions set");
    assert_eq!(value.as_str(), Some(""));
}

/// End-to-end through the real shim binary: spawn a bridge, run the shim as
/// Claude Code would, and check the `initialize` result actually carries the
/// instructions.
///
/// The unit tests above prove the bridge answers the request; this proves the
/// shim asks it and puts the answer where an MCP client will look. Those are
/// separate failures and only this one is what the user experiences.
#[test]
fn shim_initialize_carries_instructions() {
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Command, Stdio};

    const TEXT: &str = "Use these tools for diagrams.";
    let bridge = Bridge::builder("test_shim_initialize")
        .instructions(TEXT)
        .build();
    let port = bridge.port();
    bridge.spawn();

    let client = BridgeClient::new(port);
    for _ in 0..50 {
        if client.call(&BridgeRequest::Ping).is_ok() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    let mut child = Command::new(env!("CARGO_BIN_EXE_strands-claude-mcp-shim"))
        .args(["--name", "test_shim_initialize", "--port", &port.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("shim binary should run");

    let request = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "{request}").unwrap();
    stdin.flush().unwrap();

    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .expect("shim should answer initialize");
    let _ = child.kill();

    let response: serde_json::Value = serde_json::from_str(&line).expect("valid JSON-RPC");
    assert_eq!(
        response["result"]["instructions"].as_str(),
        Some(TEXT),
        "initialize result was: {line}"
    );
}
