use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::StrandsError;
use crate::types::message::Message;

use super::SessionManager;

/// Stored session metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub session_id: String,
    pub created_at: String,
    pub updated_at: String,
}

/// Stored agent state within a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRecord {
    pub agent_id: String,
    pub session_id: String,
    pub conversation_manager_state: Option<serde_json::Value>,
}

/// Abstract backend for session storage.
///
/// Implement this trait to persist sessions to any backend
/// (database, S3, Redis, Elasticsearch, etc.). The
/// [`RepositorySessionManager`] adapts this into the
/// [`SessionManager`] trait.
#[async_trait]
pub trait SessionRepository: Send + Sync {
    /// Create or update session metadata.
    async fn save_session(&self, record: &SessionRecord) -> Result<(), StrandsError>;

    /// Load session metadata.
    async fn load_session(
        &self,
        session_id: &str,
    ) -> Result<Option<SessionRecord>, StrandsError>;

    /// Delete a session and all its data.
    async fn delete_session(&self, session_id: &str) -> Result<(), StrandsError>;

    /// Save an agent's state within a session.
    async fn save_agent(&self, record: &AgentRecord) -> Result<(), StrandsError>;

    /// Load an agent's state.
    async fn load_agent(
        &self,
        session_id: &str,
        agent_id: &str,
    ) -> Result<Option<AgentRecord>, StrandsError>;

    /// Append a message to the agent's conversation history.
    async fn append_message(
        &self,
        session_id: &str,
        agent_id: &str,
        index: usize,
        message: &Message,
    ) -> Result<(), StrandsError>;

    /// Load all messages for an agent within a session.
    async fn load_messages(
        &self,
        session_id: &str,
        agent_id: &str,
    ) -> Result<Vec<Message>, StrandsError>;

    /// Delete all messages for an agent within a session.
    async fn delete_messages(
        &self,
        session_id: &str,
        agent_id: &str,
    ) -> Result<(), StrandsError>;
}

/// Session manager backed by a [`SessionRepository`].
///
/// Adapts any repository implementation into the [`SessionManager`] trait.
/// Handles the mapping between the flat `SessionManager` API and the
/// richer repository operations.
pub struct RepositorySessionManager {
    repository: Box<dyn SessionRepository>,
    agent_id: String,
}

impl RepositorySessionManager {
    pub fn new(repository: impl SessionRepository + 'static, agent_id: impl Into<String>) -> Self {
        Self {
            repository: Box::new(repository),
            agent_id: agent_id.into(),
        }
    }
}

#[async_trait]
impl SessionManager for RepositorySessionManager {
    async fn save(
        &self,
        session_id: &str,
        messages: &[Message],
    ) -> Result<(), StrandsError> {
        let now = chrono::Utc::now().to_rfc3339();

        // Upsert session metadata
        let session_record = SessionRecord {
            session_id: session_id.to_string(),
            created_at: now.clone(),
            updated_at: now,
        };
        self.repository.save_session(&session_record).await?;

        // Save agent state
        let agent_record = AgentRecord {
            agent_id: self.agent_id.clone(),
            session_id: session_id.to_string(),
            conversation_manager_state: None,
        };
        self.repository.save_agent(&agent_record).await?;

        // Clear and re-save all messages
        self.repository
            .delete_messages(session_id, &self.agent_id)
            .await?;
        for (i, msg) in messages.iter().enumerate() {
            self.repository
                .append_message(session_id, &self.agent_id, i, msg)
                .await?;
        }

        Ok(())
    }

    async fn load(
        &self,
        session_id: &str,
    ) -> Result<Option<Vec<Message>>, StrandsError> {
        let session = self.repository.load_session(session_id).await?;
        if session.is_none() {
            return Ok(None);
        }

        let messages = self
            .repository
            .load_messages(session_id, &self.agent_id)
            .await?;

        if messages.is_empty() {
            Ok(None)
        } else {
            Ok(Some(messages))
        }
    }

    async fn delete(&self, session_id: &str) -> Result<(), StrandsError> {
        self.repository.delete_session(session_id).await
    }
}
