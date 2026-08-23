//! Persisting conversations.
//!
//! A [`SessionManager`] saves and restores message history. [`Snapshot`] goes
//! further, capturing agent state and manager bookkeeping too, which is what
//! [`Checkpointer`](crate::agent::Checkpointer) builds on.

/// One JSON file per session.
pub mod file;
/// Adapting an arbitrary backend.
pub mod repository;
/// Complete, restorable captures.
pub mod snapshot;

use async_trait::async_trait;

use crate::error::StrandsError;
use crate::types::message::Message;

/// Persistence layer for agent conversation state.
#[async_trait]
pub trait SessionManager: Send + Sync {
    /// Persist a conversation, replacing any previous one.
    async fn save(&self, session_id: &str, messages: &[Message]) -> Result<(), StrandsError>;
    /// Restore a conversation, or `None` if the session is unknown.
    async fn load(&self, session_id: &str) -> Result<Option<Vec<Message>>, StrandsError>;
    /// Delete a session. A no-op if it does not exist.
    async fn delete(&self, session_id: &str) -> Result<(), StrandsError>;
}

pub use file::FileSessionManager;
pub use repository::{RepositorySessionManager, SessionRepository};
pub use snapshot::{InMemorySnapshotStore, Snapshot, SnapshotStore, SNAPSHOT_VERSION};
