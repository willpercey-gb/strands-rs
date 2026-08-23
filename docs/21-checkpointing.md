# Checkpointing

A checkpoint is a complete, restorable capture of an agent mid-run, so a long
task can survive a restart or be resumed on another machine.

## Snapshots

A snapshot captures more than the conversation:

| Field | Why it matters |
|-------|----------------|
| `messages` | Full history, tracking ids and metadata included |
| `agent_state` | What the agent accumulated — restoring messages alone loses it |
| `conversation_manager_state` | How much has been summarized so far |
| `model_state` | Whatever the adapter needs to resume |

## Taking checkpoints

```rust,ignore
use strands_core::agent::{CheckpointPolicy, Checkpointer};
use strands_core::session::InMemorySnapshotStore;

let store = Arc::new(InMemorySnapshotStore::new());
let checkpointer = Checkpointer::new(store, "session-1", "agent-1")
    .with_policy(CheckpointPolicy::EveryNTurns(5));

// After each turn:
checkpointer
    .maybe_checkpoint(turn, agent.messages(), &agent.state)
    .await?;
```

Policies are `EveryTurn`, `EveryNTurns(n)`, or `Manual`.

## Restoring

```rust,ignore
if let Some(snapshot) = checkpointer.latest().await? {
    agent.set_messages(snapshot.messages);
    agent.state = snapshot.agent_state;
}
```

`restore` **refuses a snapshot written by a newer format version** rather than
silently resuming from state this build does not understand.

## Turn boundaries only

Checkpoints are taken between turns, never mid-batch. Mid-batch the history
holds a `ToolUse` whose `ToolResult` does not exist yet, and restoring from
there resumes into a conversation the provider rejects.

## Custom stores

```rust,ignore
#[async_trait]
impl SnapshotStore for S3Snapshots {
    async fn save(&self, snapshot: &Snapshot) -> Result<String, StrandsError> { /* ... */ }
    async fn load(&self, id: &str) -> Result<Option<Snapshot>, StrandsError> { /* ... */ }
    async fn list(&self, session_id: &str) -> Result<Vec<String>, StrandsError> { /* ... */ }
    async fn delete(&self, id: &str) -> Result<(), StrandsError> { /* ... */ }
}
```

`list` returns newest first.
