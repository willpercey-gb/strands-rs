//! Tools are advertised under their own names, not namespaced with the
//! server name.
//!
//! The bridge used to key tools as `{server}__{tool}`. Because MCP clients
//! already scope by server, Claude Code then surfaced them as
//! `mcp__puml_studio__puml_studio__validate_puml` — a double prefix that
//! wasted ~8 tokens per tool and read like a bug. This pins the fix, and
//! pins the invariant that matters more: what `tools/list` advertises is
//! exactly what `tools/call` dispatches on.

use async_trait::async_trait;
use serde_json::{json, Value};
use strands_claude_mcp::{Bridge, BridgeClient, BridgeRequest};
use strands_core::tool::{Tool, ToolContext, ToolOutput};
use strands_core::types::tools::ToolSpec;
use strands_core::StrandsError;

struct Echo;

#[async_trait]
impl Tool for Echo {
    fn name(&self) -> &str {
        "echo"
    }
    fn spec(&self) -> ToolSpec {
        ToolSpec::new("echo", "echo the input", json!({"type": "object"}))
    }
    async fn invoke(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, StrandsError> {
        Ok(ToolOutput::success(input))
    }
}

fn client_for(name: &str) -> BridgeClient {
    let bridge = Bridge::builder(name).tool(Echo).build();
    let port = bridge.port();
    bridge.spawn();
    let client = BridgeClient::new(port);
    for _ in 0..50 {
        if client.call(&BridgeRequest::Ping).is_ok() {
            return client;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    panic!("bridge `{name}` never came up");
}

#[test]
fn tools_are_listed_under_their_own_name() {
    let client = client_for("test_names_list");
    let listed = client.call(&BridgeRequest::ListTools).unwrap();
    let names: Vec<&str> = listed
        .as_array()
        .expect("descriptors should be an array")
        .iter()
        .filter_map(|d| d["name"].as_str())
        .collect();

    assert_eq!(names, vec!["echo"]);
    assert!(
        !names.iter().any(|n| n.contains("test_names_list__")),
        "tool names must not be namespaced with the server name: {names:?}"
    );
}

#[test]
fn the_advertised_name_is_the_callable_name() {
    // The real regression risk is `descriptors()` and `invoke()` drifting
    // apart — a client can only call what it was told exists.
    let client = client_for("test_names_call");
    let listed = client.call(&BridgeRequest::ListTools).unwrap();
    let advertised = listed[0]["name"].as_str().unwrap().to_string();

    let result = client
        .call(&BridgeRequest::CallTool {
            params: strands_claude_mcp::CallToolParams {
                name: advertised.clone(),
                arguments: json!({"hello": "world"}),
            },
        })
        .unwrap();

    assert_eq!(
        result["is_error"],
        json!(false),
        "calling `{advertised}` failed"
    );
    assert_eq!(result["content"]["hello"], json!("world"));
}

/// A tool that declares MCP annotations and an output schema.
struct Annotated;

#[async_trait]
impl Tool for Annotated {
    fn name(&self) -> &str {
        "annotated"
    }
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "annotated",
            "does something risky",
            json!({"type": "object"}),
        )
        .with_output_schema(json!({"type": "string"}))
        .with_annotations(strands_core::types::tools::ToolAnnotations {
            destructive_hint: Some(true),
            read_only_hint: Some(false),
            ..Default::default()
        })
    }
    async fn invoke(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, StrandsError> {
        Ok(ToolOutput::success(input))
    }
}

#[test]
fn annotations_and_output_schema_reach_the_descriptor() {
    // MCP clients surface annotations when asking the user to approve a call;
    // dropping them makes a destructive tool look unremarkable.
    let bridge = Bridge::builder("annotations-test").tool(Annotated).build();
    let descriptors = bridge.registry().descriptors();

    let tool = descriptors
        .iter()
        .find(|d| d.name == "annotated")
        .expect("tool present");

    let annotations = tool.annotations.as_ref().expect("annotations forwarded");
    assert!(annotations.is_destructive());
    assert!(!annotations.is_read_only());
    assert!(tool.output_schema.is_some());
}
