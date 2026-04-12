use super::message::Role;

/// Events emitted during streaming model responses.
/// Unified protocol between model adapters and the agent loop.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    MessageStart {
        role: Role,
    },
    ContentBlockStart {
        index: usize,
        content_type: ContentBlockType,
    },
    ContentBlockDelta {
        index: usize,
        delta: DeltaContent,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageStop {
        stop_reason: StopReason,
    },
    Metadata {
        usage: Usage,
    },
}

#[derive(Debug, Clone)]
pub enum ContentBlockType {
    Text,
    ToolUse { tool_use_id: String, name: String },
}

#[derive(Debug, Clone)]
pub enum DeltaContent {
    TextDelta(String),
    ToolInputDelta(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Cancelled,
    ContentFiltered,
    GuardrailIntervention,
}

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_duration_ns: Option<u64>,
}
