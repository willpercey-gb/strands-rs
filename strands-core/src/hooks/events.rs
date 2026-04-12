use serde_json::Value;

use crate::types::message::Message;
use crate::types::streaming::StopReason;

/// Lifecycle events dispatched during agent execution.
///
/// Some events carry mutable fields that hooks can modify to influence
/// agent behavior (e.g., cancel a tool call, retry a model call,
/// override messages).
#[derive(Debug)]
pub enum HookEvent {
    /// Agent has been initialized.
    AgentInitialized,

    /// Before processing a user prompt.
    /// Hooks can override the messages that will be processed.
    BeforeInvocation(BeforeInvocationEvent),

    /// After completing an invocation.
    /// Hooks can set `resume` to re-invoke the agent automatically.
    AfterInvocation(AfterInvocationEvent),

    /// A message was added to conversation history.
    MessageAdded {
        message: Message,
    },

    /// Before calling the model.
    BeforeModelCall {
        cycle: usize,
    },

    /// After receiving a model response.
    /// Hooks can set `retry` to re-invoke the model.
    AfterModelCall(AfterModelCallEvent),

    /// Before executing a tool.
    /// Hooks can set `cancel` to skip tool execution.
    BeforeToolCall(BeforeToolCallEvent),

    /// After executing a tool.
    /// Hooks can set `retry` to re-execute the tool.
    AfterToolCall(AfterToolCallEvent),
}

#[derive(Debug)]
pub struct BeforeInvocationEvent {
    pub messages: Vec<Message>,
    /// Set to override the messages the agent will process.
    pub override_messages: Option<Vec<Message>>,
}

#[derive(Debug)]
pub struct AfterInvocationEvent {
    pub stop_reason: StopReason,
    pub cycle_count: usize,
    /// Set to `true` to re-invoke the agent with the same messages.
    pub resume: bool,
}

#[derive(Debug)]
pub struct AfterModelCallEvent {
    pub stop_reason: StopReason,
    pub cycle: usize,
    /// Set to `true` to retry the model call (e.g., on throttling).
    pub retry: bool,
}

#[derive(Debug)]
pub struct BeforeToolCallEvent {
    pub tool_name: String,
    pub input: Value,
    /// Set to `true` to cancel this tool execution.
    pub cancel: bool,
}

#[derive(Debug)]
pub struct AfterToolCallEvent {
    pub tool_name: String,
    pub is_error: bool,
    /// Set to `true` to retry the tool execution.
    pub retry: bool,
}
