//! A tool that pauses for a given number of seconds.
//!
//! Useful for polling loops and rate-limit backoff that the model drives itself.

use async_trait::async_trait;
use serde_json::{json, Value};
use strands_core::error::StrandsError;
use strands_core::tool::{Tool, ToolContext, ToolOutput};
use strands_core::types::tools::{ToolAnnotations, ToolSpec};

/// Longest a single sleep may last.
///
/// Without a cap, a model that misreads a unit ("sleep 3600") silently parks
/// the agent for an hour while looking like it is working.
pub const MAX_SLEEP_SECONDS: f64 = 300.0;

/// Pauses execution for a bounded interval.
#[derive(Debug, Clone, Copy)]
pub struct SleepTool {
    max_seconds: f64,
}

impl Default for SleepTool {
    fn default() -> Self {
        Self {
            max_seconds: MAX_SLEEP_SECONDS,
        }
    }
}

impl SleepTool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Lower the per-call cap.
    pub fn with_max_seconds(mut self, seconds: f64) -> Self {
        self.max_seconds = seconds.max(0.0);
        self
    }
}

#[async_trait]
impl Tool for SleepTool {
    fn name(&self) -> &str {
        "sleep"
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "sleep",
            format!(
                "Pause for a number of seconds (maximum {}). Use between polls \
                 of a slow operation rather than busy-waiting.",
                self.max_seconds
            ),
            json!({
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "number",
                        "description": "How long to pause, in seconds",
                        "minimum": 0
                    }
                },
                "required": ["seconds"]
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
        let Some(seconds) = input.get("seconds").and_then(Value::as_f64) else {
            return Ok(ToolOutput::error("'seconds' must be a number"));
        };

        if seconds < 0.0 {
            return Ok(ToolOutput::error("'seconds' must not be negative"));
        }

        if seconds > self.max_seconds {
            return Ok(ToolOutput::error(format!(
                "'seconds' must be at most {}; got {seconds}",
                self.max_seconds
            )));
        }

        tokio::time::sleep(std::time::Duration::from_secs_f64(seconds)).await;
        Ok(ToolOutput::success(format!("Slept for {seconds} seconds")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sleeps_for_the_requested_time() {
        let started = std::time::Instant::now();
        let out = SleepTool::new()
            .invoke(json!({"seconds": 0.05}), &ToolContext::default())
            .await
            .unwrap();

        assert!(!out.is_error);
        assert!(started.elapsed() >= std::time::Duration::from_millis(40));
    }

    #[tokio::test]
    async fn rejects_a_sleep_beyond_the_cap() {
        // A model that misreads a unit should get an error, not park the agent.
        let out = SleepTool::new()
            .invoke(json!({"seconds": 3600}), &ToolContext::default())
            .await
            .unwrap();
        assert!(out.is_error);
    }

    #[tokio::test]
    async fn rejects_negative_and_non_numeric_input() {
        for bad in [json!({"seconds": -1}), json!({"seconds": "soon"}), json!({})] {
            let out = SleepTool::new()
                .invoke(bad.clone(), &ToolContext::default())
                .await
                .unwrap();
            assert!(out.is_error, "expected {bad} to be rejected");
        }
    }

    #[tokio::test]
    async fn the_cap_is_configurable() {
        let tool = SleepTool::new().with_max_seconds(0.01);
        let out = tool
            .invoke(json!({"seconds": 1}), &ToolContext::default())
            .await
            .unwrap();
        assert!(out.is_error);
    }

    #[test]
    fn annotated_as_read_only_and_non_destructive() {
        let spec = SleepTool::new().spec();
        let annotations = spec.annotations.expect("annotations present");
        assert!(annotations.is_read_only());
        assert!(!annotations.is_destructive());
        assert!(!annotations.is_open_world());
    }
}
