//! A Rust port of the [AWS Strands Agents SDK](https://github.com/strands-agents/sdk-python).
//!
//! An agent is a model, a set of tools, and a loop between them. [`Agent`]
//! drives that loop: it calls the model, executes whatever tools the model
//! asks for, feeds the results back, and repeats until the model stops asking.
//!
//! ```no_run
//! use strands_core::{Agent, FnTool, ToolContext, ToolOutput};
//! # async fn example(model: impl strands_core::Model + 'static) -> strands_core::Result<()> {
//! let mut agent = Agent::builder()
//!     .model(model)
//!     .system_prompt("You are a helpful assistant.")
//!     .build()?;
//!
//! let result = agent.prompt("What is 2 + 2?").await?;
//! println!("{}", result.text());
//! # Ok(())
//! # }
//! ```
//!
//! # Where things live
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`agent`] | The loop, its builder, limits, state and checkpointing |
//! | [`model`] | The [`Model`] trait, token counting, routing and fallback |
//! | [`mod@tool`] | Tool definitions, execution strategies, structured output |
//! | [`types`] | Messages, content blocks, streaming events, tool specs |
//! | [`conversation`] | Context-window management: trimming, pinning, summarizing |
//! | [`hooks`] | Lifecycle callbacks that can observe and steer the loop |
//! | [`middleware`] | Wrapping model calls to cache, route, rate-limit or transform |
//! | [`interrupt`] | Pausing for human input and resuming |
//! | [`interventions`] | Allow/deny/escalate policy over tool calls |
//! | [`memory`] | Searchable facts that outlive a conversation |
//! | [`storage`] | A byte-oriented backend shared by memory, sessions and offloading |
//! | [`session`] | Persisting conversations and snapshots |
//! | [`multiagent`] | Swarm and graph orchestration |
//! | [`plugin`] | Bundling hooks and tools; the ready-made plugins |
//! | [`telemetry`] | Per-cycle and per-tool metrics |
//!
//! Prose guides live in [`docs/`](https://github.com/willpercey-gb/strands-rs/tree/main/docs).

// Public API must be documented: this crate is published, so anything
// undocumented here shows up as a gap on docs.rs.
#![warn(missing_docs)]

pub mod agent;
pub mod conversation;
/// Error types and provider-failure classification.
pub mod error;
pub mod hooks;
pub mod interrupt;
pub mod interventions;
pub mod memory;
pub mod middleware;
/// The [`Model`] trait, token counting, routing and fallback.
pub mod model;
pub mod multiagent;
/// Bundling hooks and tools; the ready-made plugins.
pub mod plugin;
pub mod session;
pub mod storage;
pub mod telemetry;
/// Tool definitions, execution strategies and structured output.
///
/// Note this shares its name with the [`tool`](macro@crate::tool) attribute
/// macro. They live in different namespaces, so `use strands_core::tool;`
/// imports both and `#[tool]` still resolves.
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

/// Re-exports the `#[tool]` macro expands to.
///
/// Not part of the public API. It exists so a user of the macro does not have
/// to add `async_trait` to their own `Cargo.toml` just to satisfy code they
/// never wrote.
#[doc(hidden)]
#[cfg(feature = "macros")]
pub mod __macro_support {
    pub use async_trait::async_trait;
}
