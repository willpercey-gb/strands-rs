# Session Management

Session managers persist conversation history across agent invocations, allowing conversations to survive process restarts or be shared across services.

## Built-In Managers

### FileSessionManager

Stores sessions as JSON files:

```rust
use strands_core::session::FileSessionManager;

let agent = Agent::builder()
    .model(model)
    .session_manager(FileSessionManager::new("./sessions"))
    .build()?;
```

Each session is saved as `{base_dir}/{session_id}.json`.

### RepositorySessionManager

Adapts any storage backend via the `SessionRepository` trait:

```rust
use strands_core::session::{RepositorySessionManager, SessionRepository};

struct MyDatabaseRepo { /* ... */ }

#[async_trait]
impl SessionRepository for MyDatabaseRepo {
    async fn save_session(&self, record: &SessionRecord) -> Result<(), StrandsError> { /* ... */ }
    async fn load_session(&self, session_id: &str) -> Result<Option<SessionRecord>, StrandsError> { /* ... */ }
    async fn delete_session(&self, session_id: &str) -> Result<(), StrandsError> { /* ... */ }
    async fn save_agent(&self, record: &AgentRecord) -> Result<(), StrandsError> { /* ... */ }
    async fn load_agent(&self, session_id: &str, agent_id: &str) -> Result<Option<AgentRecord>, StrandsError> { /* ... */ }
    async fn append_message(&self, session_id: &str, agent_id: &str, index: usize, message: &Message) -> Result<(), StrandsError> { /* ... */ }
    async fn load_messages(&self, session_id: &str, agent_id: &str) -> Result<Vec<Message>, StrandsError> { /* ... */ }
    async fn delete_messages(&self, session_id: &str, agent_id: &str) -> Result<(), StrandsError> { /* ... */ }
}

let repo = MyDatabaseRepo::new(/* ... */);
let session_mgr = RepositorySessionManager::new(repo, "my-agent");

let agent = Agent::builder()
    .model(model)
    .session_manager(session_mgr)
    .build()?;
```

The `SessionRepository` trait separates storage concerns:
- `SessionRecord` — session metadata (ID, timestamps)
- `AgentRecord` — per-agent state within a session
- Messages stored individually by index

## Custom Session Managers

Implement `SessionManager` directly for simpler cases:

```rust
#[async_trait]
pub trait SessionManager: Send + Sync {
    async fn save(&self, session_id: &str, messages: &[Message]) -> Result<(), StrandsError>;
    async fn load(&self, session_id: &str) -> Result<Option<Vec<Message>>, StrandsError>;
    async fn delete(&self, session_id: &str) -> Result<(), StrandsError>;
}
```

## Agent State

Agents have user-defined state that persists across invocations (separate from conversation messages):

```rust
let mut agent = Agent::builder().model(model).build()?;

// Set state
agent.state.insert("user_id".into(), json!("user_123"));
agent.state.insert("preferences".into(), json!({"theme": "dark"}));

// Read state (e.g., in tools via ToolContext, or directly)
let user_id = agent.state.get("user_id");
```
