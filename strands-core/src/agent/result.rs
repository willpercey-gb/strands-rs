use crate::types::message::Message;
use crate::types::streaming::{Metrics, StopReason, Usage};

/// The result of a complete agent invocation.
#[derive(Debug, Clone)]
pub struct AgentResult {
    /// Why the agent stopped.
    pub stop_reason: StopReason,
    /// The final assistant message.
    pub message: Message,
    /// Token usage across all cycles.
    pub usage: Usage,
    /// Performance metrics across all cycles.
    pub metrics: Metrics,
    /// How many model call cycles were executed.
    pub cycle_count: usize,
    /// Per-cycle and per-tool telemetry for this invocation.
    ///
    /// Cycle metrics are collected by the loop. Tool metrics come from
    /// `AfterToolCall`, which carries each call's measured duration — register a
    /// hook feeding [`MetricsCollector::record_tool`](crate::MetricsCollector::record_tool)
    /// to populate them.
    pub telemetry: crate::telemetry::AgentMetrics,
    /// Interrupts awaiting a human answer.
    ///
    /// Non-empty exactly when `stop_reason` is
    /// [`StopReason::Interrupt`]. Answer them via
    /// [`Agent::respond`](crate::Agent::respond) and re-invoke to continue.
    pub interrupts: Vec<crate::interrupt::Interrupt>,
}

impl AgentResult {
    /// Whether the run paused for human input.
    pub fn is_interrupted(&self) -> bool {
        self.stop_reason == StopReason::Interrupt
    }

    /// Extract the text content from the final message.
    pub fn text(&self) -> String {
        self.message.text()
    }
}
