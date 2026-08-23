//! The agent and its loop.
//!
//! [`Agent`] owns the model, the tool set, the conversation, and the policy
//! objects that shape a run — a [`ConversationManager`],
//! hooks, middleware, [`Limits`] and an optional session manager. Build one with
//! [`Agent::builder`].

use crate::types::content::SystemPrompt;
/// Fluent construction of an [`Agent`].
mod builder;
/// Real-time streaming callbacks.
pub mod callback;
/// Durable snapshots at turn boundaries.
pub mod checkpoint;
/// The ReAct loop itself.
mod event_loop;
/// Per-invocation budget caps.
pub mod limits;
/// What an invocation produced.
mod result;
/// Durable per-agent key/value state.
pub mod state;

pub use builder::AgentBuilder;
pub use callback::CallbackHandler;
pub use checkpoint::{CheckpointPolicy, Checkpointer};
pub use event_loop::RetryConfig;
pub use limits::Limits;
pub use result::AgentResult;
pub use state::AgentState;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::conversation::ConversationManager;
use crate::error::StrandsError;
use crate::hooks::HookRegistry;
use crate::model::Model;
use crate::session::SessionManager;
use crate::tool::{Tool, ToolContext, ToolOutput};
use crate::types::message::Message;
use crate::types::tools::ToolSpec;
use tracing::debug;

/// The core agent. Orchestrates model calls, tool execution,
/// and conversation management in a ReAct loop.
pub struct Agent {
    pub(crate) model: std::sync::Arc<dyn Model>,
    pub(crate) tools: HashMap<String, Box<dyn Tool>>,
    pub(crate) system_prompt: Option<SystemPrompt>,
    pub(crate) messages: Vec<Message>,
    pub(crate) conversation_manager: Box<dyn ConversationManager>,
    pub(crate) session_manager: Option<Box<dyn SessionManager>>,
    pub(crate) hooks: HookRegistry,
    pub(crate) callback_handler: Option<Box<dyn CallbackHandler>>,
    pub(crate) cancel: Arc<AtomicBool>,
    pub(crate) max_cycles: usize,
    pub(crate) retry_config: RetryConfig,
    /// Strategy for running a batch of tool calls.
    pub(crate) tool_executor: Box<dyn crate::tool::ToolExecutor>,
    /// Per-invocation budget caps.
    pub(crate) limits: Limits,
    /// Interrupts raised and answered, retained across invocations so a
    /// resumed run can find its answers.
    pub(crate) interrupts: crate::interrupt::InterruptState,
    /// Middleware wrapping each model invocation.
    pub(crate) model_middleware: std::sync::Arc<
        crate::middleware::MiddlewareChain<
            crate::middleware::InvokeModelContext,
            crate::middleware::stages::InvokeModelResult,
        >,
    >,
    /// Per-invocation state, persisted across cycles within a single prompt() call.
    pub(crate) invocation_state: serde_json::Value,
    /// User-defined persistent state, preserved across invocations and
    /// persisted with the session.
    pub state: AgentState,
    /// Agent name (used for identification in multi-agent patterns).
    pub name: Option<String>,
    /// Agent description (used for auto-conversion to tool).
    pub description: Option<String>,
}

impl Agent {
    /// Create a new agent via the builder pattern.
    pub fn builder() -> AgentBuilder {
        AgentBuilder::new()
    }

    /// Process a user prompt through the full ReAct loop.
    pub async fn prompt(&mut self, input: &str) -> crate::Result<AgentResult> {
        self.cancel.store(false, Ordering::Relaxed);
        self.invocation_state = serde_json::Value::Object(serde_json::Map::new());

        // Add the user message, stamped with a durable id.
        let mut user_msg = Message::user(input);
        user_msg.ensure_tracking_id();
        self.messages.push(user_msg);

        let result = event_loop::run_loop(
            self.model.clone(),
            &self.tools,
            &mut self.messages,
            self.system_prompt.as_ref(),
            self.conversation_manager.as_ref(),
            &self.hooks,
            self.callback_handler.as_deref(),
            &self.cancel,
            self.max_cycles,
            &self.retry_config,
            &mut self.invocation_state,
            self.tool_executor.as_ref(),
            &self.limits,
            &mut self.interrupts,
            self.model_middleware.as_ref(),
        )
        .await?;

        // A run that finished without pausing must not carry a stale request
        // into the next one; answers are kept for the rest of the cycle.
        if result.stop_reason != crate::types::streaming::StopReason::Interrupt {
            self.interrupts.clear_unanswered();
        }

        // Persist if session manager is configured
        if let Some(ref sm) = self.session_manager {
            let session_id = "default";
            sm.save(session_id, &self.messages).await?;
        }

        Ok(result)
    }

    /// Run the loop and return a schema-constrained final answer.
    ///
    /// A synthetic tool carrying `spec` is registered for the duration of the
    /// call, so the model produces the answer through the provider's own
    /// constrained-decoding path rather than by emitting JSON into prose.
    ///
    /// If the model finishes without calling it, one follow-up prompt asks it
    /// to format what it just said. That single retry is deliberate: models
    /// commonly forget the final call once, and almost never twice, so
    /// retrying further mostly burns tokens.
    pub async fn prompt_structured<T>(
        &mut self,
        input: &str,
        spec: crate::tool::StructuredOutputSpec,
    ) -> crate::Result<T>
    where
        T: serde::de::DeserializeOwned + Send + Sync + 'static,
    {
        use crate::tool::{StructuredOutputSlot, StructuredOutputTool};

        let name = spec.name.clone();
        let slot = StructuredOutputSlot::<T>::new();

        // Register the synthetic tool for the duration of this call only.
        let displaced = self.tools.insert(
            name.clone(),
            Box::new(StructuredOutputTool::new(spec, slot.clone())),
        );

        let outcome = async {
            self.prompt(input).await?;

            if !slot.is_filled() {
                debug!(
                    tool_name = %name,
                    "Model finished without structured output; prompting once to format"
                );
                self.prompt(crate::tool::DEFAULT_STRUCTURED_OUTPUT_PROMPT)
                    .await?;
            }

            slot.take().ok_or_else(|| {
                StrandsError::Other(format!(
                    "Model did not produce structured output for {name}"
                ))
            })
        }
        .await;

        // Restore the tool table whatever happened, so a failed structured
        // call does not leave the synthetic tool advertised on later turns.
        match displaced {
            Some(tool) => {
                self.tools.insert(name, tool);
            }
            None => {
                self.tools.remove(&name);
            }
        }

        outcome
    }

    /// Supply answers to pending interrupts.
    ///
    /// Returns how many were applied. Re-invoke with
    /// [`prompt`](Self::prompt) afterwards to continue the paused run; the
    /// hook that paused will see its answer on the next pass.
    pub fn respond(&mut self, responses: &[crate::interrupt::InterruptResponse]) -> usize {
        self.interrupts.respond(responses)
    }

    /// Interrupts currently awaiting an answer.
    pub fn pending_interrupts(&self) -> Vec<crate::interrupt::Interrupt> {
        self.interrupts.pending()
    }

    /// Cancel an in-progress invocation.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Borrow the shared cancel flag. Useful when the agent has been
    /// moved into a worker task and the caller still needs a handle to
    /// flip cancellation from outside (e.g. a UI cancel button).
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    /// Names of the tools currently registered.
    pub fn tool_names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(String::as_str)
    }

    /// Get the current conversation history.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Clear conversation history.
    pub fn clear_messages(&mut self) {
        self.messages.clear();
    }

    /// Replace the entire conversation history. Useful when the agent
    /// is owned by a worker task and the caller wants to seed prior
    /// turns from an external store before calling `prompt`. The next
    /// `prompt(input)` call will still append `input` as a user
    /// message on top of whatever is set here, so do not include the
    /// new prompt in `msgs`.
    pub fn set_messages(&mut self, msgs: Vec<Message>) {
        self.messages = msgs;
    }

    /// Access the invocation state from the last prompt() call.
    pub fn invocation_state(&self) -> &serde_json::Value {
        &self.invocation_state
    }

    /// Wrap this agent as a tool for use by another agent.
    pub fn as_tool(self, name: impl Into<String>, description: impl Into<String>) -> AgentTool {
        AgentTool {
            name: name.into(),
            description: description.into(),
            agent: Arc::new(tokio::sync::Mutex::new(self)),
            mode: DelegationMode::FreshPrompt,
        }
    }
}

/// How a sub-agent receives work from its caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelegationMode {
    /// The sub-agent sees only the prompt it is given.
    ///
    /// Cheapest and most predictable: the sub-agent cannot be confused or
    /// steered by the caller's history.
    FreshPrompt,
    /// The sub-agent is seeded with the caller's conversation, then prompted.
    ///
    /// For work that genuinely depends on what came before — a reviewer that
    /// needs to see what was written, not a summary of it. Costs the full
    /// history in tokens on every call, which is why it is not the default.
    ///
    /// Upstream v1.53 `feat(py): add agent-as-tool delegation`.
    SharedContext,
}

// ---------------------------------------------------------------------------
// AgentTool — wraps an Agent as a Tool for multi-agent delegation
// ---------------------------------------------------------------------------

/// An agent wrapped as a tool, enabling hierarchical multi-agent patterns.
pub struct AgentTool {
    name: String,
    description: String,
    agent: Arc<tokio::sync::Mutex<Agent>>,
    mode: DelegationMode,
}

impl AgentTool {
    /// Hand the sub-agent the caller's conversation before prompting it.
    /// Set the shared context.
    pub fn with_shared_context(mut self) -> Self {
        self.mode = DelegationMode::SharedContext;
        self
    }

    /// The configured delegation mode.
    pub fn mode(&self) -> DelegationMode {
        self.mode
    }

    /// Seed the sub-agent with `messages` before its next invocation.
    ///
    /// Only meaningful in [`DelegationMode::SharedContext`].
    pub async fn share_context(&self, messages: Vec<Message>) {
        if self.mode == DelegationMode::SharedContext {
            self.agent.lock().await.set_messages(messages);
        }
    }
}

#[async_trait::async_trait]
impl Tool for AgentTool {
    /// Name, for logs and diagnostics.
    fn name(&self) -> &str {
        &self.name
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            self.name.clone(),
            self.description.clone(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to send to the sub-agent"
                    }
                },
                "required": ["prompt"]
            }),
        )
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> crate::Result<ToolOutput> {
        let prompt = input["prompt"].as_str().ok_or_else(|| StrandsError::Tool {
            tool_name: self.name.clone(),
            message: "Missing 'prompt' field".into(),
        })?;

        let mut agent = self.agent.lock().await;
        let result = agent.prompt(prompt).await?;
        let text = result.text();

        Ok(ToolOutput::success(serde_json::Value::String(text)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    use crate::types::message::Message;
    use crate::types::streaming::StreamEvent;
    use crate::types::tools::ToolSpec;
    use async_trait::async_trait;
    use futures::stream;

    /// A model that yields no events and exits with EndTurn — enough to
    /// build a real Agent for the cancel-handle test.
    struct EmptyModel;

    #[async_trait]
    impl Model for EmptyModel {
        async fn stream(
            &self,
            _messages: &[Message],
            _system_prompt: Option<&SystemPrompt>,
            _tool_specs: &[ToolSpec],
        ) -> Result<crate::model::ModelStream, StrandsError> {
            let s = stream::iter(vec![Ok::<StreamEvent, StrandsError>(
                StreamEvent::MessageStop {
                    stop_reason: crate::types::streaming::StopReason::EndTurn,
                },
            )]);
            Ok(Box::pin(s))
        }
    }

    #[test]
    fn cancel_handle_is_shared_with_internal_flag() {
        let agent = Agent::builder()
            .model(EmptyModel)
            .build()
            .expect("build agent");
        let handle = agent.cancel_handle();
        assert!(!handle.load(Ordering::Relaxed));
        handle.store(true, Ordering::Relaxed);
        // The agent's own cancel field reads the same value because they
        // are clones of one Arc<AtomicBool>.
        assert!(agent.cancel.load(Ordering::Relaxed));
    }
}
