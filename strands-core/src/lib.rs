pub mod agent;
pub mod conversation;
pub mod error;
pub mod hooks;
pub mod interrupt;
pub mod interventions;
pub mod memory;
pub mod middleware;
pub mod model;
pub mod multiagent;
pub mod plugin;
pub mod session;
pub mod storage;
pub mod telemetry;
pub mod tool;
pub mod types;

// Re-exports for convenience
pub use agent::{Agent, AgentBuilder, AgentResult, CallbackHandler, RetryConfig};
pub use conversation::ConversationManager;
pub use error::{classify_cli_failure, classify_provider_failure, Result, StrandsError};
pub use hooks::{Hook, HookEvent, HookRegistry};
pub use interrupt::{Interrupt, InterruptResponse, InterruptState};
pub use interventions::{
    InterventionAction, InterventionContext, InterventionHandler, InterventionRegistry,
};
pub use memory::{MemoryManager, MemoryRecord, MemoryStore};
pub use middleware::{Middleware, MiddlewareChain, Next};
pub use model::Model;
pub use plugin::Plugin;
pub use session::SessionManager;
pub use storage::{InMemoryStorage, LocalFileStorage, Storage, StorageExt};
pub use telemetry::{AgentMetrics, MetricsCollector};
pub use tool::{FnTool, Tool, ToolContext, ToolOutput};
pub use types::content::ContentBlock;
pub use types::message::{Message, Role};
pub use types::streaming::{StopReason, StreamEvent, Usage};
pub use types::tools::ToolSpec;

#[cfg(feature = "macros")]
pub use strands_macros::tool;
