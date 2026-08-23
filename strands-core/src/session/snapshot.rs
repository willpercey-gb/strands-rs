//! Point-in-time snapshots of an agent's full state.
//!
//! A [`SessionManager`] persists conversation history. A snapshot captures
//! everything else too — agent state, conversation-manager bookkeeping, model
//! state — so a run can be resumed on a different process, or rewound to an
//! earlier point.
//!
//! That completeness is the whole point: restoring messages alone silently
//! loses whatever the agent had accumulated in [`AgentState`](crate::agent::AgentState), which is exactly
//! the context a resumed run needs.
//!
//! Ported from upstream `session/snapshot_session_manager.py` and
//! `types/_snapshot.py`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::AgentState;
use crate::error::StrandsError;
use crate::types::message::Message;

/// Version of the snapshot format, so a future change can be detected rather
/// than silently misread.
pub const SNAPSHOT_VERSION: u32 = 1;

/// A complete, restorable capture of an agent at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Format version. See [`SNAPSHOT_VERSION`].
    #[serde(default = "default_version")]
    pub version: u32,
    /// Session this snapshot belongs to.
    pub session_id: String,
    /// Agent within the session.
    pub agent_id: String,
    /// RFC 3339 capture time.
    pub created_at: String,
    /// Full conversation history, tracking ids and metadata included.
    pub messages: Vec<Message>,
    /// The agent's durable key/value state.
    #[serde(default)]
    pub agent_state: AgentState,
    /// Conversation-manager bookkeeping (e.g. how much has been summarized).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation_manager_state: Option<Value>,
    /// Provider-specific state a model adapter needs to resume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_state: Option<Value>,
}

fn default_version() -> u32 {
    SNAPSHOT_VERSION
}

impl Snapshot {
    /// Capture a snapshot of the given history and state.
    pub fn new(
        session_id: impl Into<String>,
        agent_id: impl Into<String>,
        messages: Vec<Message>,
        agent_state: AgentState,
    ) -> Self {
        Self {
            version: SNAPSHOT_VERSION,
            session_id: session_id.into(),
            agent_id: agent_id.into(),
            created_at: chrono::Utc::now().to_rfc3339(),
            messages,
            agent_state,
            conversation_manager_state: None,
            model_state: None,
        }
    }

    /// Set the conversation manager state.
    pub fn with_conversation_manager_state(mut self, state: Value) -> Self {
        self.conversation_manager_state = Some(state);
        self
    }

    /// Set the model state.
    pub fn with_model_state(mut self, state: Value) -> Self {
        self.model_state = Some(state);
        self
    }

    /// Whether this snapshot's format is one this build understands.
    pub fn is_supported(&self) -> bool {
        self.version <= SNAPSHOT_VERSION
    }
}

/// Storage for snapshots.
///
/// Separate from [`SessionManager`](super::SessionManager) because the access
/// pattern differs: snapshots are listed and selected by id, not overwritten
/// in place.
#[async_trait]
pub trait SnapshotStore: Send + Sync {
    /// Persist a snapshot, returning its id.
    async fn save(&self, snapshot: &Snapshot) -> Result<String, StrandsError>;

    /// Load a snapshot by id.
    async fn load(&self, snapshot_id: &str) -> Result<Option<Snapshot>, StrandsError>;

    /// Snapshot ids for a session, newest first.
    async fn list(&self, session_id: &str) -> Result<Vec<String>, StrandsError>;

    /// Delete a snapshot.
    async fn delete(&self, snapshot_id: &str) -> Result<(), StrandsError>;
}

/// In-memory snapshot store.
///
/// Useful for tests and for rewind-within-a-run, where durability is not
/// wanted.
#[derive(Default)]
pub struct InMemorySnapshotStore {
    snapshots: tokio::sync::Mutex<Vec<(String, Snapshot)>>,
}

impl InMemorySnapshotStore {
    /// Create with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of entries.
    pub async fn len(&self) -> usize {
        self.snapshots.lock().await.len()
    }

    /// Whether there are no entries.
    pub async fn is_empty(&self) -> bool {
        self.snapshots.lock().await.is_empty()
    }
}

#[async_trait]
impl SnapshotStore for InMemorySnapshotStore {
    async fn save(&self, snapshot: &Snapshot) -> Result<String, StrandsError> {
        let mut snapshots = self.snapshots.lock().await;
        let id = format!("{}-{}", snapshot.session_id, snapshots.len());
        snapshots.push((id.clone(), snapshot.clone()));
        Ok(id)
    }

    async fn load(&self, snapshot_id: &str) -> Result<Option<Snapshot>, StrandsError> {
        Ok(self
            .snapshots
            .lock()
            .await
            .iter()
            .find(|(id, _)| id == snapshot_id)
            .map(|(_, s)| s.clone()))
    }

    async fn list(&self, session_id: &str) -> Result<Vec<String>, StrandsError> {
        Ok(self
            .snapshots
            .lock()
            .await
            .iter()
            .rev()
            .filter(|(_, s)| s.session_id == session_id)
            .map(|(id, _)| id.clone())
            .collect())
    }

    async fn delete(&self, snapshot_id: &str) -> Result<(), StrandsError> {
        self.snapshots
            .lock()
            .await
            .retain(|(id, _)| id != snapshot_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn snapshot(session: &str) -> Snapshot {
        let mut state = AgentState::new();
        state.set("visits", 3);
        Snapshot::new(session, "agent-1", vec![Message::user("hello")], state)
    }

    #[tokio::test]
    async fn save_and_load_round_trip() {
        let store = InMemorySnapshotStore::new();
        let id = store.save(&snapshot("s1")).await.unwrap();

        let loaded = store.load(&id).await.unwrap().expect("snapshot present");
        assert_eq!(loaded.session_id, "s1");
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.agent_state.get("visits"), Some(&json!(3)));
    }

    #[tokio::test]
    async fn agent_state_survives_serialization() {
        // Restoring messages alone would silently drop everything the agent
        // had accumulated, which is the context a resumed run needs most.
        let snap = snapshot("s1").with_model_state(json!({"cursor": "abc"}));

        let text = serde_json::to_string(&snap).unwrap();
        let back: Snapshot = serde_json::from_str(&text).unwrap();

        assert_eq!(back.agent_state.get("visits"), Some(&json!(3)));
        assert_eq!(back.model_state, Some(json!({"cursor": "abc"})));
    }

    #[tokio::test]
    async fn message_tracking_ids_survive_a_snapshot() {
        let mut message = Message::user("hi");
        let id = message.ensure_tracking_id().to_string();

        let snap = Snapshot::new("s1", "a", vec![message], AgentState::new());
        let back: Snapshot = serde_json::from_str(&serde_json::to_string(&snap).unwrap()).unwrap();

        assert_eq!(back.messages[0].tracking_id.as_deref(), Some(id.as_str()));
    }

    #[tokio::test]
    async fn list_is_newest_first_and_scoped_to_the_session() {
        let store = InMemorySnapshotStore::new();
        store.save(&snapshot("s1")).await.unwrap();
        let second = store.save(&snapshot("s1")).await.unwrap();
        store.save(&snapshot("other")).await.unwrap();

        let ids = store.list("s1").await.unwrap();
        assert_eq!(ids.len(), 2, "other sessions must not leak in");
        assert_eq!(ids[0], second, "newest first");
    }

    #[tokio::test]
    async fn delete_removes_only_the_named_snapshot() {
        let store = InMemorySnapshotStore::new();
        let first = store.save(&snapshot("s1")).await.unwrap();
        let second = store.save(&snapshot("s1")).await.unwrap();

        store.delete(&first).await.unwrap();

        assert!(store.load(&first).await.unwrap().is_none());
        assert!(store.load(&second).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn loading_an_unknown_id_is_not_an_error() {
        let store = InMemorySnapshotStore::new();
        assert!(store.load("nope").await.unwrap().is_none());
    }

    #[test]
    fn a_newer_format_version_is_detected_rather_than_misread() {
        let mut snap = snapshot("s1");
        assert!(snap.is_supported());

        snap.version = SNAPSHOT_VERSION + 1;
        assert!(!snap.is_supported());
    }

    #[test]
    fn snapshots_without_a_version_default_to_the_current_one() {
        let text = r#"{"session_id":"s","agent_id":"a","created_at":"now","messages":[]}"#;
        let snap: Snapshot = serde_json::from_str(text).unwrap();
        assert_eq!(snap.version, SNAPSHOT_VERSION);
        assert!(snap.agent_state.is_empty());
    }
}
