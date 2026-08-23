# Memory & Storage

Two related but distinct things:

| | Scope | Purpose |
|---|-------|---------|
| Conversation history | One run | What the model sees; bounded and trimmed |
| `AgentState` | Across runs | Key/value scratchpad |
| **Memory** | Across sessions | Searchable facts that accumulate |
| **Storage** | — | Where bytes go; backs the others |

## Storage

One byte-oriented trait, so memory, the context offloader and session
persistence share a backend instead of each configuring the same bucket
differently.

```rust,ignore
use strands_core::storage::{InMemoryStorage, LocalFileStorage, Storage, StorageExt};

let storage = LocalFileStorage::new("/var/lib/my-agent");

storage.write("notes/today", b"content".to_vec()).await?;
let bytes = storage.read("notes/today").await?;        // Option<Vec<u8>>
let keys  = storage.list("notes").await?;

// JSON helpers come from StorageExt.
storage.write_json("config", &settings).await?;
let settings: Option<Settings> = storage.read_json("config").await?;
```

A missing key reads as `Ok(None)`, not an error — forcing callers to
pattern-match an error for "not there" makes real failures easy to swallow.

**Keys are validated.** They routinely come from model output, and
`LocalFileStorage` maps them onto real paths, so `.`/`..` segments are rejected
and separators collapsed. `LocalFileStorage` also writes via temp-plus-rename
and refuses to read through symlinks.

## Memory

```rust,ignore
use strands_core::memory::{InMemoryMemoryStore, MemoryManager};

let store = Arc::new(InMemoryMemoryStore::default());
let memory = MemoryManager::new(store);

// After a turn, extract anything worth keeping.
memory.maybe_extract(agent.messages()).await?;

// Later, recall it.
for hit in memory.recall("deployment key").await? {
    println!("{} ({:.2})", hit.record.content, hit.score);
}
```

### Extraction

Storing every message would make memory a second, worse copy of the transcript.
Extraction selects what is durable.

```rust,ignore
use strands_core::memory::{ExtractionConfig, ExtractionTrigger};

let memory = MemoryManager::new(store).with_config(
    ExtractionConfig::default().with_trigger(ExtractionTrigger::MessageCount(10)),
);
```

It defaults to **user messages only**. The assistant's own output is the least
reliable source of durable fact — remembering it turns a guess into something
the agent later treats as established.

### The watermark

`maybe_extract` only considers messages past what it has already seen, so
calling it every turn does not re-store the whole conversation. Messages that
were considered and rejected still advance the watermark, so they are not
re-examined forever.

A store that rejects a write is logged, not raised: memory is an enhancement,
and losing a write should not take the conversation down with it.

### Custom stores

The bundled `InMemoryMemoryStore` scores by substring overlap — deliberately
simple rather than pretending to be semantic. Real deployments want a vector
store:

```rust,ignore
#[async_trait]
impl MemoryStore for MyVectorStore {
    fn config(&self) -> &MemoryStoreConfig { &self.config }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, StrandsError> {
        // embed + nearest-neighbour lookup
    }

    async fn add(&self, record: MemoryRecord) -> Result<(), StrandsError> { /* ... */ }
}
```

`add` defaults to refusing, so a read-only store need not implement a write path
that would be a lie.
