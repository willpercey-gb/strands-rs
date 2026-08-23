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
        /// The message that was added.
        message: Message,
    },

    /// Before calling the model.
    BeforeModelCall {
        /// Which iteration of the loop this is.
        cycle: usize,
    },

    /// After receiving a model response.
    /// Hooks can set `retry` to re-invoke the model.
    AfterModelCall(AfterModelCallEvent),

    /// Before executing a batch of tools requested by one assistant message.
    ///
    /// Fires once per cycle, ahead of any per-tool event. Hooks can set
    /// `cancel` to skip the whole batch — cheaper and more predictable than
    /// cancelling each tool individually.
    BeforeTools(BeforeToolsEvent),

    /// Before executing a tool.
    /// Hooks can set `cancel` to skip tool execution.
    BeforeToolCall(BeforeToolCallEvent),

    /// After executing a tool.
    /// Hooks can set `retry` to re-execute the tool.
    AfterToolCall(AfterToolCallEvent),

    /// After a batch of tools has finished and results are ready.
    ///
    /// Dispatched in reverse hook order, so cleanup unwinds opposite to setup.
    /// Hooks can set `end_turn` to halt the loop without another model call.
    AfterTools(AfterToolsEvent),
}

impl HookEvent {
    /// Whether this event unwinds in reverse hook order.
    ///
    /// Teardown events run last-registered-first so a hook tears down before
    /// whatever it was layered on top of.
    pub fn is_teardown(&self) -> bool {
        matches!(
            self,
            HookEvent::AfterTools(_) | HookEvent::AfterInvocation(_)
        )
    }
}

/// Payload for [`HookEvent::BeforeInvocation`].
#[derive(Debug)]
pub struct BeforeInvocationEvent {
    /// The conversation as it stands.
    pub messages: Vec<Message>,
    /// Set to override the messages the agent will process.
    pub override_messages: Option<Vec<Message>>,
}

/// Payload for [`HookEvent::AfterInvocation`].
#[derive(Debug)]
pub struct AfterInvocationEvent {
    /// Why the model stopped.
    pub stop_reason: StopReason,
    /// How many cycles ran.
    pub cycle_count: usize,
    /// Set to `true` to re-invoke the agent with the same messages.
    pub resume: bool,
}

/// Payload for [`HookEvent::AfterModelCall`].
#[derive(Debug)]
pub struct AfterModelCallEvent {
    /// Why the model stopped.
    pub stop_reason: StopReason,
    /// Which iteration of the loop this is.
    pub cycle: usize,
    /// Set to `true` to retry the model call (e.g., on throttling).
    pub retry: bool,
}

/// Payload for [`HookEvent::BeforeTools`].
#[derive(Debug)]
pub struct BeforeToolsEvent {
    /// The tool calls the model requested, as `(tool_use_id, name)`.
    pub tool_calls: Vec<(String, String)>,
    /// Set to `true` to cancel every tool in this batch.
    pub cancel: bool,
}

/// Payload for [`HookEvent::AfterTools`].
#[derive(Debug)]
pub struct AfterToolsEvent {
    /// Outcome of each tool in the batch, as `(tool_name, is_error)`.
    pub results: Vec<(String, bool)>,
    /// Set to `true` to end the turn without calling the model again.
    pub end_turn: bool,
}

/// Payload for [`HookEvent::BeforeToolCall`].
#[derive(Debug)]
pub struct BeforeToolCallEvent {
    /// Name of the tool involved.
    pub tool_name: String,
    /// Arguments the model supplied.
    pub input: Value,
    /// Set to `true` to cancel this tool execution.
    pub cancel: bool,
    /// Interrupts raised and answered during this invocation.
    ///
    /// Call [`interrupt`](Self::interrupt) to pause for human input.
    pub interrupts: crate::interrupt::InterruptState,
}

impl BeforeToolCallEvent {
    /// Pause for human input, or collect the answer if one was supplied.
    ///
    /// Returns `None` on the first pass — the hook must then decline to
    /// approve whatever it was asking about, typically by setting
    /// [`cancel`](Self::cancel). It returns the answer once the caller has
    /// responded and re-invoked.
    pub fn interrupt(&mut self, name: &str, reason: Option<Value>) -> Option<Value> {
        self.interrupts.interrupt(name, reason)
    }
}

/// Payload for [`HookEvent::AfterToolCall`].
#[derive(Debug)]
pub struct AfterToolCallEvent {
    /// Name of the tool involved.
    pub tool_name: String,
    /// Whether the call returned an error.
    pub is_error: bool,
    /// How long the tool took to execute.
    pub duration: std::time::Duration,
    /// Set to `true` to retry the tool execution.
    pub retry: bool,
}
