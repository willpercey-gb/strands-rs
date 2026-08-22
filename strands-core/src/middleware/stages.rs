//! The built-in middleware stages.
//!
//! Each stage names one interception point and fixes the context and result
//! types that flow through it.
//!
//! Ported from upstream `_middleware/stages.py`.

use std::sync::Arc;

use crate::error::StrandsError;
use crate::model::Model;
use crate::types::content::{ContentBlock, SystemPrompt};
use crate::types::message::Message;
use crate::types::streaming::{Metrics, StopReason, Usage};
use crate::types::tools::ToolSpec;

/// Context for the model-invocation stage.
///
/// The collections are owned copies, so middleware can rewrite what this call
/// sends without disturbing the agent's own history. `model` is shared and may
/// be swapped per call, which is how model routing works.
pub struct InvokeModelContext {
    /// Messages this call will send.
    pub messages: Vec<Message>,
    /// System prompt for this call.
    pub system_prompt: Option<SystemPrompt>,
    /// Tools advertised on this call.
    pub tool_specs: Vec<ToolSpec>,
    /// The model to invoke. Replace it to route this call elsewhere.
    pub model: Arc<dyn Model>,
    /// Which cycle of the agent loop this is.
    pub cycle: usize,
}

impl std::fmt::Debug for InvokeModelContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvokeModelContext")
            .field("messages", &self.messages.len())
            .field("tool_specs", &self.tool_specs.len())
            .field("cycle", &self.cycle)
            .finish()
    }
}

/// What one model call produced.
#[derive(Debug, Clone)]
pub struct ModelCallOutcome {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: Usage,
    pub metrics: Metrics,
}

impl ModelCallOutcome {
    /// The concatenated text of the response.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| b.as_text())
            .collect::<Vec<_>>()
            .join("")
    }
}

/// Result type for the model stage.
pub type InvokeModelResult = Result<ModelCallOutcome, StrandsError>;

/// Context for the tool-execution stage.
///
/// Covers one tool call. `input` is owned, so middleware can rewrite arguments
/// — validate and coerce them, redact a secret, substitute a default — without
/// mutating what the model actually asked for in the history.
#[derive(Debug, Clone)]
pub struct ExecuteToolContext {
    pub tool_use_id: String,
    pub tool_name: String,
    pub input: serde_json::Value,
}

/// Result type for the tool stage.
pub type ExecuteToolResult = Result<crate::tool::ToolOutput, StrandsError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcome_text_concatenates_text_blocks_only() {
        let outcome = ModelCallOutcome {
            content: vec![
                ContentBlock::Text {
                    text: "hello ".into(),
                },
                ContentBlock::ToolUse {
                    tool_use_id: "1".into(),
                    name: "t".into(),
                    input: serde_json::json!({}),
                },
                ContentBlock::Text {
                    text: "world".into(),
                },
            ],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
            metrics: Metrics::default(),
        };
        assert_eq!(outcome.text(), "hello world");
    }

    #[test]
    fn outcome_text_is_empty_without_text_blocks() {
        let outcome = ModelCallOutcome {
            content: vec![],
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
            metrics: Metrics::default(),
        };
        assert_eq!(outcome.text(), "");
    }
}
