//! Expose [`strands_core`] tools as a dynamic MCP server that Claude Code can
//! discover and call. The host process runs an in-memory bridge over TCP; a
//! separate `strands-claude-mcp-shim` binary (registered with `claude mcp add`)
//! forwards JSON-RPC over stdio into the bridge.
//!
//! Usage:
//!
//! ```ignore
//! use strands_claude_mcp::Bridge;
//!
//! let bridge = Bridge::builder("planner")
//!     .tool(create_node_tool)
//!     .tool(create_edge_tool)
//!     .build();
//!
//! bridge.spawn();
//! strands_claude_mcp::install("planner", bridge.port())?;
//! ```
//!
//! Tools added to the bridge are exposed to Claude Code as MCP tools whose
//! names are prefixed with the bridge's `name` (so the planner's
//! `create_node` becomes `planner__create_node` to avoid collisions when
//! multiple host apps register).

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use strands_core::tool::{Tool, ToolContext};

pub mod bridge;
pub mod install;

pub use bridge::Bridge;
pub use install::{find_shim_binary, install, uninstall};

/// MCP protocol version we speak. Stable subset — initialize, tools/list,
/// tools/call.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Hash a server name to a deterministic port in the dynamic range.
/// FNV-1a over the bytes; mod into 49152..65535 (IANA dynamic).
pub fn port_for(name: &str) -> u16 {
    let mut h: u64 = 14695981039346656037;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    49152 + (h % (65535 - 49152)) as u16
}

// ---------------------------------------------------------------------------
// Wire protocol — used by both the bridge and the shim
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum BridgeRequest {
    Ping,
    ListTools,
    CallTool {
        params: CallToolParams,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallToolParams {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    /// Whatever the tool returned. JSON; the shim turns it into a text content
    /// block for MCP.
    pub content: Value,
    pub is_error: bool,
}

/// One-line client used by the shim to talk to the bridge over TCP.
/// Synchronous: we read a single line per request.
pub struct BridgeClient {
    addr: String,
}

impl BridgeClient {
    pub fn new(port: u16) -> Self {
        Self {
            addr: format!("127.0.0.1:{port}"),
        }
    }

    pub fn call(&self, req: &BridgeRequest) -> Result<Value, String> {
        let mut stream = TcpStream::connect(&self.addr).map_err(|e| {
            format!("connect to bridge {}: {e} (is the host app running?)", self.addr)
        })?;
        let mut line = serde_json::to_string(req)
            .map_err(|e| format!("encode request: {e}"))?;
        line.push('\n');
        stream
            .write_all(line.as_bytes())
            .map_err(|e| format!("write to bridge: {e}"))?;
        // Make sure we don't block forever on a hanging server.
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(60)));

        let mut reader = BufReader::new(stream);
        let mut response = String::new();
        reader
            .read_line(&mut response)
            .map_err(|e| format!("read from bridge: {e}"))?;
        let parsed: Value = serde_json::from_str(&response)
            .map_err(|e| format!("parse bridge response: {e}"))?;
        if let Some(err) = parsed.get("error").and_then(|e| e.as_str()) {
            return Err(err.to_string());
        }
        Ok(parsed.get("result").cloned().unwrap_or(Value::Null))
    }
}

// ---------------------------------------------------------------------------
// Internal: a sharable tool registry
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct ToolRegistry {
    /// `prefixed_name` → strands tool. Tool names are namespaced with the
    /// bridge name to avoid collisions when more than one host app exposes
    /// tools with overlapping names.
    pub tools: Arc<HashMap<String, Arc<dyn Tool>>>,
}

impl ToolRegistry {
    pub fn descriptors(&self) -> Vec<ToolDescriptor> {
        self.tools
            .iter()
            .map(|(prefixed, t)| {
                let spec = t.spec();
                ToolDescriptor {
                    name: prefixed.clone(),
                    description: spec.description,
                    input_schema: spec.input_schema,
                }
            })
            .collect()
    }

    pub async fn invoke(&self, name: &str, arguments: Value) -> ToolCallResult {
        match self.tools.get(name) {
            None => ToolCallResult {
                content: Value::String(format!("unknown tool: {name}")),
                is_error: true,
            },
            Some(tool) => {
                let ctx = ToolContext::default();
                match tool.invoke(arguments, &ctx).await {
                    Ok(out) => ToolCallResult {
                        content: out.content,
                        is_error: out.is_error,
                    },
                    Err(e) => ToolCallResult {
                        content: Value::String(e.to_string()),
                        is_error: true,
                    },
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience for callers — find the binary on disk
// ---------------------------------------------------------------------------

pub fn cargo_target_candidates() -> Vec<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut out = vec![];
    if let Some(workspace) = manifest.parent() {
        for profile in ["debug", "release"] {
            out.push(workspace.join("target").join(profile).join("strands-claude-mcp-shim"));
        }
    }
    out
}

/// Locate the `claude` CLI on the user's PATH. Returned as the absolute path
/// so callers can log it; failures here are usually a missing install.
pub fn find_claude_cli() -> Option<PathBuf> {
    let out = Command::new("sh").args(["-lc", "command -v claude"]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if path.is_empty() {
        None
    } else {
        Some(PathBuf::from(path))
    }
}
