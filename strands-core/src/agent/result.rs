use crate::types::message::Message;
use crate::types::streaming::{StopReason, Usage};

/// The result of a complete agent invocation.
#[derive(Debug, Clone)]
pub struct AgentResult {
    /// Why the agent stopped.
    pub stop_reason: StopReason,
    /// The final assistant message.
    pub message: Message,
    /// Token usage across all cycles.
    pub usage: Usage,
    /// How many model call cycles were executed.
    pub cycle_count: usize,
}

impl AgentResult {
    /// Extract the text content from the final message.
    pub fn text(&self) -> String {
        self.message.text()
    }
}
