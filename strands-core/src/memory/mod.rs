//! Long-term memory — facts an agent keeps across sessions, not just within
//! one conversation.
//!
//! Distinct from the conversation history (which is bounded and gets trimmed)
//! and from [`AgentState`](crate::agent::AgentState) (which is a key/value
//! scratchpad). Memory is searchable, accumulates over time, and is what lets
//! an agent recall something a user told it last week.
//!
//! Ported from upstream `memory/`.

/// Deciding what is worth remembering.
pub mod extraction;
/// Coordinating memory across a run.
pub mod manager;
/// Where remembered facts live.
pub mod store;

pub use extraction::{ExtractionConfig, ExtractionTrigger, MemoryExtractor};
pub use manager::MemoryManager;
pub use store::{
    InMemoryMemoryStore, MemoryRecord, MemoryStore, MemoryStoreConfig, SearchQuery, SearchResult,
};
