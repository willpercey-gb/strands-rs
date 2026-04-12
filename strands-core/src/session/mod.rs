pub mod file;
pub mod repository;

use async_trait::async_trait;

use crate::error::StrandsError;
use crate::types::message::Message;

/// Persistence layer for agent conversation state.
#[async_trait]
pub trait SessionManager: Send + Sync {
    async fn save(&self, session_id: &str, messages: &[Message]) -> Result<(), StrandsError>;
    async fn load(&self, session_id: &str) -> Result<Option<Vec<Message>>, StrandsError>;
    async fn delete(&self, session_id: &str) -> Result<(), StrandsError>;
}

pub use file::FileSessionManager;
pub use repository::{RepositorySessionManager, SessionRepository};
