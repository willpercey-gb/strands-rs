//! A tool the model can call to end its own turn.
//!
//! Gives the model an explicit way to say "I am done" instead of the loop
//! having to infer completion from the absence of a tool call.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use strands_core::error::StrandsError;
use strands_core::tool::{Tool, ToolContext, ToolOutput};
use strands_core::types::tools::{ToolAnnotations, ToolSpec};

/// Signals that the model has finished.
///
/// Pair with [`StopTool::signal`] to observe it from outside the agent.
#[derive(Debug, Clone, Default)]
pub struct StopSignal {
    stopped: Arc<AtomicBool>,
    reason: Arc<std::sync::Mutex<Option<String>>>,
}

impl StopSignal {
    /// Create with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the model has called the stop tool.
    /// Whether the model has called the stop tool.
    pub fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::Relaxed)
    }

    /// The reason the model gave, if any.
    /// The reason the model gave, if any.
    pub fn reason(&self) -> Option<String> {
        self.reason.lock().ok()?.clone()
    }

    /// Clear the signal, so the same agent can be reused.
    /// Clear the signal, so the same agent can be reused.
    pub fn reset(&self) {
        self.stopped.store(false, Ordering::Relaxed);
        if let Ok(mut reason) = self.reason.lock() {
            *reason = None;
        }
    }
}

/// Lets the model end its turn deliberately.
#[derive(Debug, Clone, Default)]
pub struct StopTool {
    signal: StopSignal,
}

impl StopTool {
    /// Create with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// A handle to observe whether the model has stopped.
    /// A handle for observing whether the model has stopped.
    pub fn signal(&self) -> StopSignal {
        self.signal.clone()
    }
}

#[async_trait]
impl Tool for StopTool {
    fn name(&self) -> &str {
        "stop"
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "stop",
            "Signal that the task is complete and no further work is needed. \
             Call this when you have finished, rather than continuing to loop.",
            json!({
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the task is complete"
                    }
                }
            }),
        )
        .with_annotations(ToolAnnotations {
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
            ..Default::default()
        })
    }

    async fn invoke(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, StrandsError> {
        let reason = input
            .get("reason")
            .and_then(Value::as_str)
            .unwrap_or("task complete")
            .to_string();

        self.signal.stopped.store(true, Ordering::Relaxed);
        if let Ok(mut slot) = self.signal.reason.lock() {
            *slot = Some(reason.clone());
        }

        Ok(ToolOutput::success(format!("Stopping: {reason}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn calling_stop_raises_the_signal() {
        let tool = StopTool::new();
        let signal = tool.signal();
        assert!(!signal.is_stopped());

        tool.invoke(json!({"reason": "all done"}), &ToolContext::default())
            .await
            .unwrap();

        assert!(signal.is_stopped());
        assert_eq!(signal.reason().as_deref(), Some("all done"));
    }

    #[tokio::test]
    async fn a_missing_reason_gets_a_default() {
        let tool = StopTool::new();
        tool.invoke(json!({}), &ToolContext::default())
            .await
            .unwrap();
        assert_eq!(tool.signal().reason().as_deref(), Some("task complete"));
    }

    #[tokio::test]
    async fn reset_allows_reuse() {
        let tool = StopTool::new();
        tool.invoke(json!({}), &ToolContext::default())
            .await
            .unwrap();

        tool.signal().reset();
        assert!(!tool.signal().is_stopped());
        assert!(tool.signal().reason().is_none());
    }

    #[tokio::test]
    async fn signal_handles_share_state() {
        let tool = StopTool::new();
        let a = tool.signal();
        let b = tool.signal();

        tool.invoke(json!({}), &ToolContext::default())
            .await
            .unwrap();

        assert!(a.is_stopped() && b.is_stopped());
    }
}
