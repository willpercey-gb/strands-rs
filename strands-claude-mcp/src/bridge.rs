use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;
use strands_core::tool::Tool;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

use crate::{port_for, BridgeRequest, ToolRegistry};

/// In-process TCP bridge that exposes a set of strands `Tool`s to the
/// [`crate::install`]-registered shim binary.
///
/// One bridge per host app. Cheap to clone.
#[derive(Clone)]
pub struct Bridge {
    name: String,
    port: u16,
    registry: ToolRegistry,
}

impl Bridge {
    pub fn builder(name: impl Into<String>) -> BridgeBuilder {
        BridgeBuilder::new(name)
    }

    /// Server name. Becomes the MCP server identifier registered with Claude.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// TCP port the bridge listens on. Deterministic from the name unless
    /// overridden via [`BridgeBuilder::port`].
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Spawn the listener and return immediately.
    ///
    /// If a Tokio runtime is already active on the current thread (e.g. you're
    /// inside an `async fn`), the listener is scheduled on that runtime.
    /// Otherwise — the common case when called from a host's synchronous
    /// startup hook — a dedicated multi-threaded runtime is built on a fresh
    /// thread and the listener runs there forever. Idempotent: re-spawning
    /// the same bridge is safe but only the first listener wins the port.
    pub fn spawn(&self) {
        let bridge = self.clone();
        if tokio::runtime::Handle::try_current().is_ok() {
            tokio::spawn(async move {
                if let Err(e) = bridge.serve().await {
                    tracing::warn!("strands-claude-mcp bridge `{}` exited: {e}", bridge.name);
                }
            });
            return;
        }

        std::thread::Builder::new()
            .name(format!("strands-claude-mcp-bridge-{}", self.name))
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .worker_threads(2)
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        tracing::warn!(
                            "strands-claude-mcp bridge `{}`: failed to build runtime: {e}",
                            bridge.name
                        );
                        return;
                    }
                };
                rt.block_on(async {
                    if let Err(e) = bridge.serve().await {
                        tracing::warn!("strands-claude-mcp bridge `{}` exited: {e}", bridge.name);
                    }
                });
            })
            .expect("strands-claude-mcp: failed to spawn bridge thread");
    }

    /// Run the bridge in the current task. Useful for tests.
    pub async fn serve(&self) -> std::io::Result<()> {
        let addr = format!("127.0.0.1:{}", self.port);
        let listener = TcpListener::bind(&addr).await?;
        tracing::info!("strands-claude-mcp bridge `{}` listening on {addr}", self.name);

        loop {
            let (stream, peer) = listener.accept().await?;
            tracing::debug!("strands-claude-mcp bridge: connection from {peer}");
            let registry = self.registry.clone();
            tokio::spawn(async move {
                if let Err(e) = handle_connection(stream, registry).await {
                    tracing::debug!("strands-claude-mcp bridge: connection error: {e}");
                }
            });
        }
    }
}

async fn handle_connection(
    stream: tokio::net::TcpStream,
    registry: ToolRegistry,
) -> std::io::Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    while let Some(line) = lines.next_line().await? {
        let response = match serde_json::from_str::<BridgeRequest>(&line) {
            Err(e) => json!({ "error": format!("invalid request: {e}") }),
            Ok(BridgeRequest::Ping) => json!({ "result": "pong" }),
            Ok(BridgeRequest::ListTools) => {
                json!({ "result": registry.descriptors() })
            }
            Ok(BridgeRequest::CallTool { params }) => {
                let result = registry.invoke(&params.name, params.arguments).await;
                json!({ "result": result })
            }
        };
        let mut out = serde_json::to_string(&response)
            .unwrap_or_else(|_| r#"{"error":"serialize"}"#.into());
        out.push('\n');
        if writer.write_all(out.as_bytes()).await.is_err() {
            break;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

pub struct BridgeBuilder {
    name: String,
    port: Option<u16>,
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl BridgeBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            port: None,
            tools: HashMap::new(),
        }
    }

    /// Override the deterministic port. Most callers should leave this alone.
    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Add a strands tool. The tool's name is namespaced with the server
    /// name (`<server>__<tool>`) so multiple bridges don't collide in
    /// Claude's tool registry.
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        let prefixed = format!("{}__{}", self.name, tool.name());
        self.tools.insert(prefixed, Arc::new(tool));
        self
    }

    /// Add an already-Arc'd tool — useful when the same tool is shared with
    /// a strands `Agent` via `.tool(...)` and the bridge.
    pub fn tool_arc(mut self, tool: Arc<dyn Tool>) -> Self {
        let prefixed = format!("{}__{}", self.name, tool.name());
        self.tools.insert(prefixed, tool);
        self
    }

    pub fn build(self) -> Bridge {
        let port = self.port.unwrap_or_else(|| port_for(&self.name));
        Bridge {
            name: self.name,
            port,
            registry: ToolRegistry {
                tools: Arc::new(self.tools),
            },
        }
    }
}
