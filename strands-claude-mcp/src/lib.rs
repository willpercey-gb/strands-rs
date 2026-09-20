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
//! Tools added to the bridge are exposed to Claude Code as MCP tools under
//! their own names. The client scopes them by server — Claude Code surfaces
//! the planner's `create_node` as `mcp__planner__create_node` — so the bridge
//! does not add a second `planner__` namespace of its own.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
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
    /// JSON Schema for the tool's output, when it declares one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
    /// Behavioural hints (readOnlyHint, destructiveHint, ...).
    ///
    /// Forwarded because MCP clients surface these to the user when deciding
    /// whether to approve a call; dropping them makes every tool look equally
    /// unremarkable, which is exactly wrong for a destructive one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<strands_core::types::tools::ToolAnnotations>,
}

/// Identity of the session making a call.
///
/// One host app runs one bridge, but the MCP server is registered at user
/// scope, so every Claude session on the machine shares it. Without this the
/// host cannot tell ten concurrent sessions apart, and any question of the
/// form "what is the user working on" is answered for the wrong project as
/// often as the right one.
///
/// It rides on the request rather than the connection because
/// [`BridgeClient::call`] opens a fresh TCP connection per call: there is no
/// connection state to hang it on.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallerInfo {
    /// Stable for the lifetime of one shim process, which is one Claude
    /// session. Not meaningful across restarts and not meant to be.
    pub session: String,
    /// Directory the session was launched in, which in practice is the
    /// project the agent is working on. `None` if it could not be read.
    pub cwd: Option<String>,
    pub pid: u32,
}

/// Key under which [`CallerInfo`] is placed in [`ToolContext::state`].
pub const CALLER_STATE_KEY: &str = "caller";

impl CallerInfo {
    /// Read the calling session's identity back out of a tool's context.
    ///
    /// `None` means the call is unattributed, which happens two ways: an older
    /// shim that does not send identity, or a tool invoked by the host's own
    /// in-process agent rather than over the bridge. Both are legitimate, so
    /// treat it as "unknown session" rather than as an error.
    pub fn from_context(ctx: &ToolContext) -> Option<Self> {
        serde_json::from_value(ctx.state.get(CALLER_STATE_KEY)?.clone()).ok()
    }

    /// Capture this process's identity. Called once, when the shim starts.
    pub fn current() -> Self {
        let pid = std::process::id();
        // Good enough for a local, per-process handle: pid alone is reused by
        // the OS, so mix in the start time to keep ids distinct within a run.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        Self {
            session: format!("{pid:x}-{nanos:x}"),
            cwd: std::env::current_dir()
                .ok()
                .map(|p| p.display().to_string()),
            pid,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum BridgeRequest {
    Ping,
    ListTools,
    /// Server-level guidance returned in the MCP `initialize` result.
    /// Reaches every connecting session before any tool is loaded, which
    /// makes it the only place to say *when* to reach for these tools.
    Instructions,
    CallTool {
        params: CallToolParams,
        /// Stamped by [`BridgeClient::call`], not by the caller.
        ///
        /// Optional so that a shim binary older than the host still works: the
        /// shim is a build artifact copied into the app's resources, and a
        /// stale copy must degrade to an anonymous call rather than take every
        /// tool down. Hosts should treat `None` as "unknown session".
        #[serde(default)]
        caller: Option<CallerInfo>,
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
    caller: CallerInfo,
}

impl BridgeClient {
    pub fn new(port: u16) -> Self {
        Self {
            addr: format!("127.0.0.1:{port}"),
            caller: CallerInfo::current(),
        }
    }

    /// Identity stamped onto every `CallTool` this client sends.
    pub fn caller(&self) -> &CallerInfo {
        &self.caller
    }

    pub fn call(&self, req: &BridgeRequest) -> Result<Value, String> {
        // Stamped here rather than at each call site so that a future request
        // kind cannot quietly ship without identity.
        let stamped;
        let req = match req {
            BridgeRequest::CallTool { params, .. } => {
                stamped = BridgeRequest::CallTool {
                    params: params.clone(),
                    caller: Some(self.caller.clone()),
                };
                &stamped
            }
            other => other,
        };

        let mut stream = TcpStream::connect(&self.addr).map_err(|e| {
            format!(
                "connect to bridge {}: {e} (is the host app running?)",
                self.addr
            )
        })?;
        let mut line = serde_json::to_string(req).map_err(|e| format!("encode request: {e}"))?;
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
        let parsed: Value =
            serde_json::from_str(&response).map_err(|e| format!("parse bridge response: {e}"))?;
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
/// The set of tools a bridge exposes.
pub struct ToolRegistry {
    /// Tool name → strands tool. Keys are the tools' own names; the MCP
    /// client namespaces by server, so collisions between host apps are
    /// already handled a layer up.
    pub tools: Arc<HashMap<String, Arc<dyn Tool>>>,
}

impl ToolRegistry {
    pub fn descriptors(&self) -> Vec<ToolDescriptor> {
        self.tools
            .iter()
            .map(|(name, t)| {
                let spec = t.spec();
                ToolDescriptor {
                    name: name.clone(),
                    description: spec.description,
                    input_schema: spec.input_schema,
                    output_schema: spec.output_schema,
                    annotations: spec.annotations,
                }
            })
            .collect()
    }

    pub async fn invoke(
        &self,
        name: &str,
        arguments: Value,
        caller: Option<CallerInfo>,
    ) -> ToolCallResult {
        match self.tools.get(name) {
            None => ToolCallResult {
                content: Value::String(format!("unknown tool: {name}")),
                is_error: true,
            },
            Some(tool) => {
                let mut ctx = ToolContext::default();
                if let Some(caller) = caller {
                    if let Ok(value) = serde_json::to_value(caller) {
                        ctx.state = json!({ CALLER_STATE_KEY: value });
                    }
                }
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
            out.push(
                workspace
                    .join("target")
                    .join(profile)
                    .join("strands-claude-mcp-shim"),
            );
        }
    }
    out
}

/// Locate the `claude` CLI on the user's PATH. Returned as the absolute path
/// so callers can log it; failures here are usually a missing install.
pub fn find_claude_cli() -> Option<PathBuf> {
    let out = Command::new("sh")
        .args(["-lc", "command -v claude"])
        .output()
        .ok()?;
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
