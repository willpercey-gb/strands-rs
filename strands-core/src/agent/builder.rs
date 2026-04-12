use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::conversation::sliding_window::SlidingWindowConversationManager;
use crate::conversation::ConversationManager;
use crate::error::StrandsError;
use crate::hooks::{Hook, HookRegistry};
use crate::model::Model;
use crate::plugin::Plugin;
use crate::session::SessionManager;
use crate::tool::Tool;

use super::callback::CallbackHandler;
use super::event_loop::RetryConfig;
use super::Agent;

/// Fluent builder for constructing an [`Agent`].
pub struct AgentBuilder {
    model: Option<Box<dyn Model>>,
    tools: HashMap<String, Box<dyn Tool>>,
    system_prompt: Option<String>,
    conversation_manager: Option<Box<dyn ConversationManager>>,
    session_manager: Option<Box<dyn SessionManager>>,
    hooks: HookRegistry,
    callback_handler: Option<Box<dyn CallbackHandler>>,
    max_cycles: usize,
    retry_config: RetryConfig,
    concurrent_tools: bool,
    name: Option<String>,
    description: Option<String>,
}

impl AgentBuilder {
    pub fn new() -> Self {
        Self {
            model: None,
            tools: HashMap::new(),
            system_prompt: None,
            conversation_manager: None,
            session_manager: None,
            hooks: HookRegistry::new(),
            callback_handler: None,
            max_cycles: 20,
            retry_config: RetryConfig::default(),
            concurrent_tools: false,
            name: None,
            description: None,
        }
    }

    /// Set the model provider.
    pub fn model(mut self, model: impl Model + 'static) -> Self {
        self.model = Some(Box::new(model));
        self
    }

    /// Add a tool the agent can use.
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        let name = tool.name().to_string();
        self.tools.insert(name, Box::new(tool));
        self
    }

    /// Add multiple tools at once.
    pub fn tools(mut self, tools: Vec<Box<dyn Tool>>) -> Self {
        for tool in tools {
            let name = tool.name().to_string();
            self.tools.insert(name, tool);
        }
        self
    }

    /// Set the system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the conversation manager for context window management.
    pub fn conversation_manager(mut self, cm: impl ConversationManager + 'static) -> Self {
        self.conversation_manager = Some(Box::new(cm));
        self
    }

    /// Set the session manager for persistence.
    pub fn session_manager(mut self, sm: impl SessionManager + 'static) -> Self {
        self.session_manager = Some(Box::new(sm));
        self
    }

    /// Register a lifecycle hook.
    pub fn hook(mut self, hook: impl Hook + 'static) -> Self {
        self.hooks.register(hook);
        self
    }

    /// Set the maximum number of ReAct cycles (default: 20).
    pub fn max_cycles(mut self, n: usize) -> Self {
        self.max_cycles = n;
        self
    }

    /// Configure model call retry behavior.
    pub fn retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    /// Set a callback handler for real-time stream events.
    pub fn callback_handler(mut self, handler: impl CallbackHandler + 'static) -> Self {
        self.callback_handler = Some(Box::new(handler));
        self
    }

    /// Enable concurrent tool execution (default: sequential).
    pub fn concurrent_tools(mut self, enabled: bool) -> Self {
        self.concurrent_tools = enabled;
        self
    }

    /// Set the agent name (used for identification in multi-agent patterns).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the agent description (used when auto-converting to tool).
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Register a plugin that contributes hooks and/or tools.
    pub fn plugin(mut self, plugin: impl Plugin + 'static) -> Self {
        // Register the plugin's hooks
        plugin.register_hooks(&mut self.hooks);
        // Register the plugin's tools
        for tool in plugin.tools() {
            let name = tool.name().to_string();
            self.tools.insert(name, tool);
        }
        self
    }

    /// Build the agent. Requires at least a model to be set.
    pub fn build(self) -> Result<Agent, StrandsError> {
        let model = self
            .model
            .ok_or_else(|| StrandsError::Other("Model is required".into()))?;

        let conversation_manager: Box<dyn ConversationManager> = self
            .conversation_manager
            .unwrap_or_else(|| Box::new(SlidingWindowConversationManager::default()));

        Ok(Agent {
            model,
            tools: self.tools,
            system_prompt: self.system_prompt,
            messages: Vec::new(),
            conversation_manager,
            session_manager: self.session_manager,
            hooks: self.hooks,
            callback_handler: self.callback_handler,
            cancel: Arc::new(AtomicBool::new(false)),
            max_cycles: self.max_cycles,
            retry_config: self.retry_config,
            concurrent_tools: self.concurrent_tools,
            invocation_state: serde_json::Value::Object(serde_json::Map::new()),
            state: HashMap::new(),
            name: self.name,
            description: self.description,
        })
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}
