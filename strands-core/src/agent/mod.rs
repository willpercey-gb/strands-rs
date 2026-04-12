mod builder;
pub mod callback;
mod event_loop;
mod result;

pub use builder::AgentBuilder;
pub use callback::CallbackHandler;
pub use event_loop::RetryConfig;
pub use result::AgentResult;

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

/// The core agent. Orchestrates model calls, tool execution,
/// and conversation management in a ReAct loop.
pub struct Agent {
    pub(crate) model: Box<dyn Model>,
    pub(crate) tools: HashMap<String, Box<dyn Tool>>,
    pub(crate) system_prompt: Option<String>,
    pub(crate) messages: Vec<Message>,
    pub(crate) conversation_manager: Box<dyn ConversationManager>,
    pub(crate) session_manager: Option<Box<dyn SessionManager>>,
    pub(crate) hooks: HookRegistry,
    pub(crate) callback_handler: Option<Box<dyn CallbackHandler>>,
    pub(crate) cancel: Arc<AtomicBool>,
    pub(crate) max_cycles: usize,
    pub(crate) retry_config: RetryConfig,
    /// Whether to execute tools concurrently (default: false = sequential).
    pub(crate) concurrent_tools: bool,
    /// Per-invocation state, persisted across cycles within a single prompt() call.
    pub(crate) invocation_state: serde_json::Value,
    /// User-defined persistent state, preserved across invocations.
    pub state: HashMap<String, serde_json::Value>,
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

        // Add the user message
        self.messages.push(Message::user(input));

        let result = event_loop::run_loop(
            self.model.as_ref(),
            &self.tools,
            &mut self.messages,
            self.system_prompt.as_deref(),
            self.conversation_manager.as_ref(),
            &self.hooks,
            self.callback_handler.as_deref(),
            &self.cancel,
            self.max_cycles,
            &self.retry_config,
            &mut self.invocation_state,
            self.concurrent_tools,
        )
        .await?;

        // Persist if session manager is configured
        if let Some(ref sm) = self.session_manager {
            let session_id = "default";
            sm.save(session_id, &self.messages).await?;
        }

        Ok(result)
    }

    /// Cancel an in-progress invocation.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Get the current conversation history.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Clear conversation history.
    pub fn clear_messages(&mut self) {
        self.messages.clear();
    }

    /// Access the invocation state from the last prompt() call.
    pub fn invocation_state(&self) -> &serde_json::Value {
        &self.invocation_state
    }

    /// Wrap this agent as a tool for use by another agent.
    pub fn as_tool(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> AgentTool {
        AgentTool {
            name: name.into(),
            description: description.into(),
            agent: Arc::new(tokio::sync::Mutex::new(self)),
        }
    }
}

// ---------------------------------------------------------------------------
// AgentTool — wraps an Agent as a Tool for multi-agent delegation
// ---------------------------------------------------------------------------

/// An agent wrapped as a tool, enabling hierarchical multi-agent patterns.
pub struct AgentTool {
    name: String,
    description: String,
    agent: Arc<tokio::sync::Mutex<Agent>>,
}

#[async_trait::async_trait]
impl Tool for AgentTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to send to the sub-agent"
                    }
                },
                "required": ["prompt"]
            }),
        }
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> crate::Result<ToolOutput> {
        let prompt = input["prompt"]
            .as_str()
            .ok_or_else(|| StrandsError::Tool {
                tool_name: self.name.clone(),
                message: "Missing 'prompt' field".into(),
            })?;

        let mut agent = self.agent.lock().await;
        let result = agent.prompt(prompt).await?;
        let text = result.text();

        Ok(ToolOutput::success(serde_json::Value::String(text)))
    }
}
