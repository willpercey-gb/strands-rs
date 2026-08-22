//! Checkpointing — durably persisting an agent mid-run so it can be resumed
//! elsewhere.
//!
//! A checkpoint is a [`Snapshot`] taken at a turn boundary. Turn boundaries are
//! the only safe point: mid-batch, the history holds a `ToolUse` whose
//! `ToolResult` does not exist yet, and restoring from there would resume into a
//! conversation the provider rejects.
//!
//! Ported from upstream `experimental/checkpoint/`.

use std::sync::Arc;

use crate::agent::AgentState;
use crate::error::StrandsError;
use crate::session::{Snapshot, SnapshotStore};
use crate::types::message::Message;

/// When to take a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointPolicy {
    /// After every turn.
    EveryTurn,
    /// After every `n`th turn.
    EveryNTurns(usize),
    /// Only when explicitly asked.
    Manual,
}

impl CheckpointPolicy {
    /// Whether a checkpoint is due after completing `turn` (1-based).
    pub fn should_checkpoint(&self, turn: usize) -> bool {
        match self {
            CheckpointPolicy::EveryTurn => true,
            // A zero interval would mean "never", which `Manual` already says
            // more clearly; treat it as every turn rather than dividing by zero.
            CheckpointPolicy::EveryNTurns(0) => true,
            CheckpointPolicy::EveryNTurns(n) => turn % n == 0,
            CheckpointPolicy::Manual => false,
        }
    }
}

/// Captures checkpoints into a [`SnapshotStore`].
pub struct Checkpointer {
    store: Arc<dyn SnapshotStore>,
    policy: CheckpointPolicy,
    session_id: String,
    agent_id: String,
}

impl Checkpointer {
    pub fn new(
        store: Arc<dyn SnapshotStore>,
        session_id: impl Into<String>,
        agent_id: impl Into<String>,
    ) -> Self {
        Self {
            store,
            policy: CheckpointPolicy::EveryTurn,
            session_id: session_id.into(),
            agent_id: agent_id.into(),
        }
    }

    pub fn with_policy(mut self, policy: CheckpointPolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn policy(&self) -> CheckpointPolicy {
        self.policy
    }

    /// Take a checkpoint if the policy says one is due after `turn`.
    ///
    /// Returns the new checkpoint's id, or `None` if none was due.
    pub async fn maybe_checkpoint(
        &self,
        turn: usize,
        messages: &[Message],
        state: &AgentState,
    ) -> Result<Option<String>, StrandsError> {
        if !self.policy.should_checkpoint(turn) {
            return Ok(None);
        }
        self.checkpoint(messages, state).await.map(Some)
    }

    /// Take a checkpoint unconditionally.
    pub async fn checkpoint(
        &self,
        messages: &[Message],
        state: &AgentState,
    ) -> Result<String, StrandsError> {
        let snapshot = Snapshot::new(
            self.session_id.clone(),
            self.agent_id.clone(),
            messages.to_vec(),
            state.clone(),
        );
        self.store.save(&snapshot).await
    }

    /// Load a checkpoint by id.
    pub async fn restore(&self, id: &str) -> Result<Option<Snapshot>, StrandsError> {
        let Some(snapshot) = self.store.load(id).await? else {
            return Ok(None);
        };

        if !snapshot.is_supported() {
            return Err(StrandsError::Session(format!(
                "checkpoint {id} was written by a newer version (format {}); refusing to restore",
                snapshot.version
            )));
        }

        Ok(Some(snapshot))
    }

    /// The most recent checkpoint for this session, if any.
    pub async fn latest(&self) -> Result<Option<Snapshot>, StrandsError> {
        let ids = self.store.list(&self.session_id).await?;
        match ids.first() {
            Some(id) => self.restore(id).await,
            None => Ok(None),
        }
    }

    /// Checkpoint ids for this session, newest first.
    pub async fn history(&self) -> Result<Vec<String>, StrandsError> {
        self.store.list(&self.session_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::InMemorySnapshotStore;
    use serde_json::json;

    fn checkpointer() -> (Checkpointer, Arc<InMemorySnapshotStore>) {
        let store = Arc::new(InMemorySnapshotStore::new());
        (
            Checkpointer::new(store.clone(), "session-1", "agent-1"),
            store,
        )
    }

    fn state() -> AgentState {
        let mut s = AgentState::new();
        s.set("progress", 7);
        s
    }

    #[test]
    fn every_turn_policy_always_fires() {
        for turn in 1..5 {
            assert!(CheckpointPolicy::EveryTurn.should_checkpoint(turn));
        }
    }

    #[test]
    fn interval_policy_fires_on_multiples() {
        let p = CheckpointPolicy::EveryNTurns(3);
        assert!(!p.should_checkpoint(1));
        assert!(!p.should_checkpoint(2));
        assert!(p.should_checkpoint(3));
        assert!(p.should_checkpoint(6));
    }

    #[test]
    fn a_zero_interval_does_not_divide_by_zero() {
        assert!(CheckpointPolicy::EveryNTurns(0).should_checkpoint(1));
    }

    #[test]
    fn manual_policy_never_fires_automatically() {
        for turn in 1..5 {
            assert!(!CheckpointPolicy::Manual.should_checkpoint(turn));
        }
    }

    #[tokio::test]
    async fn checkpoint_captures_messages_and_state() {
        let (cp, _) = checkpointer();
        let id = cp
            .checkpoint(&[Message::user("hello")], &state())
            .await
            .unwrap();

        let restored = cp.restore(&id).await.unwrap().expect("checkpoint present");
        assert_eq!(restored.messages.len(), 1);
        assert_eq!(restored.agent_state.get("progress"), Some(&json!(7)));
    }

    #[tokio::test]
    async fn maybe_checkpoint_honours_the_policy() {
        let store = Arc::new(InMemorySnapshotStore::new());
        let cp = Checkpointer::new(store.clone(), "s", "a")
            .with_policy(CheckpointPolicy::EveryNTurns(2));

        assert!(cp.maybe_checkpoint(1, &[], &state()).await.unwrap().is_none());
        assert_eq!(store.len().await, 0);

        assert!(cp.maybe_checkpoint(2, &[], &state()).await.unwrap().is_some());
        assert_eq!(store.len().await, 1);
    }

    #[tokio::test]
    async fn latest_returns_the_newest_checkpoint() {
        let (cp, _) = checkpointer();
        cp.checkpoint(&[Message::user("first")], &AgentState::new())
            .await
            .unwrap();
        cp.checkpoint(&[Message::user("second")], &AgentState::new())
            .await
            .unwrap();

        let latest = cp.latest().await.unwrap().expect("a checkpoint");
        assert_eq!(latest.messages[0].text(), "second");
    }

    #[tokio::test]
    async fn latest_is_none_before_any_checkpoint() {
        let (cp, _) = checkpointer();
        assert!(cp.latest().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn history_is_newest_first() {
        let (cp, _) = checkpointer();
        let first = cp.checkpoint(&[], &AgentState::new()).await.unwrap();
        let second = cp.checkpoint(&[], &AgentState::new()).await.unwrap();

        let history = cp.history().await.unwrap();
        assert_eq!(history, vec![second, first]);
    }

    #[tokio::test]
    async fn restoring_an_unknown_id_is_not_an_error() {
        let (cp, _) = checkpointer();
        assert!(cp.restore("nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn a_newer_format_is_refused_rather_than_misread() {
        // Silently loading a future format would resume an agent from state
        // this build does not actually understand.
        let store = Arc::new(InMemorySnapshotStore::new());
        let mut snapshot = Snapshot::new("s", "a", vec![], AgentState::new());
        snapshot.version = crate::session::SNAPSHOT_VERSION + 1;
        let id = store.save(&snapshot).await.unwrap();

        let cp = Checkpointer::new(store, "s", "a");
        let err = cp.restore(&id).await.unwrap_err();
        assert!(err.to_string().contains("newer version"), "got {err}");
    }
}
