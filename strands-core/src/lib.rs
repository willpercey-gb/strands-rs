pub mod agent;
pub mod conversation;
pub mod error;
pub mod hooks;
pub mod model;
pub mod multiagent;
pub mod plugin;
pub mod session;
pub mod tool;
pub mod types;

// Re-exports for convenience
pub use agent::{Agent, AgentBuilder, AgentResult, CallbackHandler, RetryConfig};
pub use conversation::ConversationManager;
pub use error::{classify_cli_failure, Result, StrandsError};
pub use hooks::{Hook, HookEvent, HookRegistry};
pub use model::Model;
pub use plugin::Plugin;
pub use session::SessionManager;
pub use tool::{FnTool, Tool, ToolContext, ToolOutput};
pub use types::content::ContentBlock;
pub use types::message::{Message, Role};
pub use types::streaming::{StopReason, StreamEvent, Usage};
pub use types::tools::ToolSpec;

#[cfg(feature = "macros")]
pub use strands_macros::tool;
