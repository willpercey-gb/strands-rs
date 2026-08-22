//! Memory stores — where remembered facts live.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::StrandsError;
use crate::types::message::Message;

/// A single remembered fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    /// Stable id within the store.
    pub id: String,
    /// The remembered content.
    pub content: String,
    /// Arbitrary caller metadata — source, topic, confidence.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub metadata: serde_json::Map<String, Value>,
    /// RFC 3339 creation time.
    pub created_at: String,
}

impl MemoryRecord {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.into(),
            metadata: serde_json::Map::new(),
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A search against a store.
#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub text: String,
    /// Cap on results. `None` uses the store's configured default.
    pub max_results: Option<usize>,
    /// Only return records scoring at or above this.
    pub min_score: Option<f32>,
}

impl SearchQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            max_results: None,
            min_score: None,
        }
    }

    pub fn with_max_results(mut self, n: usize) -> Self {
        self.max_results = Some(n);
        self
    }

    pub fn with_min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }
}

/// A record matched by a search, with its relevance.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub record: MemoryRecord,
    /// Relevance in `[0, 1]`. Higher is more relevant.
    pub score: f32,
}

/// Identity and behaviour a store is configured with.
#[derive(Debug, Clone)]
pub struct MemoryStoreConfig {
    /// Unique name, used to target this store from tools.
    pub name: String,
    /// Human-readable description, shown to the model in tool descriptions.
    pub description: String,
    /// Default cap on search results.
    pub max_search_results: usize,
    /// Whether this store accepts writes.
    pub writable: bool,
}

impl Default for MemoryStoreConfig {
    fn default() -> Self {
        Self {
            name: "memory".to_string(),
            description: "Long-term memory".to_string(),
            max_search_results: 10,
            writable: true,
        }
    }
}

/// A backend that stores and retrieves remembered facts.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// This store's identity and defaults.
    fn config(&self) -> &MemoryStoreConfig;

    /// Find records relevant to `query`.
    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, StrandsError>;

    /// Store a record.
    ///
    /// The default rejects the call, so a read-only store does not have to
    /// implement a write path that would be a lie.
    async fn add(&self, _record: MemoryRecord) -> Result<(), StrandsError> {
        Err(StrandsError::Other(format!(
            "memory store '{}' is not writable",
            self.config().name
        )))
    }

    /// Ingest a batch of conversation messages.
    ///
    /// The default derives one record per message with text, which is enough
    /// for stores without their own ingestion pipeline.
    async fn add_messages(&self, messages: &[Message]) -> Result<usize, StrandsError> {
        let mut stored = 0;
        for message in messages {
            let text = message.text();
            if text.is_empty() {
                continue;
            }
            self.add(
                MemoryRecord::new(text).with_metadata("role", format!("{:?}", message.role)),
            )
            .await?;
            stored += 1;
        }
        Ok(stored)
    }

    /// Async setup that must succeed before the agent runs.
    async fn initialize(&self) -> Result<(), StrandsError> {
        Ok(())
    }
}

/// An in-memory store with substring-overlap scoring.
///
/// Real deployments want a vector store. This exists so memory can be used and
/// tested without one, and its scoring is deliberately simple rather than
/// pretending to be semantic.
pub struct InMemoryMemoryStore {
    config: MemoryStoreConfig,
    records: tokio::sync::RwLock<Vec<MemoryRecord>>,
}

impl Default for InMemoryMemoryStore {
    fn default() -> Self {
        Self::new(MemoryStoreConfig::default())
    }
}

impl InMemoryMemoryStore {
    pub fn new(config: MemoryStoreConfig) -> Self {
        Self {
            config,
            records: tokio::sync::RwLock::new(Vec::new()),
        }
    }

    pub async fn len(&self) -> usize {
        self.records.read().await.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.records.read().await.is_empty()
    }

    /// Fraction of the query's words that appear in `content`.
    fn score(query: &str, content: &str) -> f32 {
        let content = content.to_lowercase();
        let words: Vec<&str> = query.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        let hits = words
            .iter()
            .filter(|w| content.contains(&w.to_lowercase()))
            .count();
        hits as f32 / words.len() as f32
    }
}

#[async_trait]
impl MemoryStore for InMemoryMemoryStore {
    fn config(&self) -> &MemoryStoreConfig {
        &self.config
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, StrandsError> {
        let limit = query
            .max_results
            .unwrap_or(self.config.max_search_results);
        let floor = query.min_score.unwrap_or(f32::EPSILON);

        let mut results: Vec<SearchResult> = self
            .records
            .read()
            .await
            .iter()
            .map(|record| SearchResult {
                score: Self::score(&query.text, &record.content),
                record: record.clone(),
            })
            .filter(|r| r.score >= floor)
            .collect();

        // Highest score first; ties broken by recency so a fresh fact wins over
        // an equally-matching stale one.
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.record.created_at.cmp(&a.record.created_at))
        });
        results.truncate(limit);
        Ok(results)
    }

    async fn add(&self, record: MemoryRecord) -> Result<(), StrandsError> {
        if !self.config.writable {
            return Err(StrandsError::Other(format!(
                "memory store '{}' is not writable",
                self.config.name
            )));
        }
        self.records.write().await.push(record);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn store_with(contents: &[&str]) -> InMemoryMemoryStore {
        let store = InMemoryMemoryStore::default();
        for content in contents {
            store.add(MemoryRecord::new(*content)).await.unwrap();
        }
        store
    }

    #[tokio::test]
    async fn search_finds_matching_records() {
        let store = store_with(&["the user prefers dark mode", "unrelated note"]).await;

        let results = store.search(&SearchQuery::new("dark mode")).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].record.content.contains("dark mode"));
    }

    #[tokio::test]
    async fn results_are_ordered_by_score() {
        let store = store_with(&["alpha beta gamma", "alpha only"]).await;

        let results = store
            .search(&SearchQuery::new("alpha beta gamma"))
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(
            results[0].score > results[1].score,
            "the fuller match should rank first"
        );
    }

    #[tokio::test]
    async fn max_results_caps_the_response() {
        let store = store_with(&["x one", "x two", "x three"]).await;
        let results = store
            .search(&SearchQuery::new("x").with_max_results(2))
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn min_score_filters_weak_matches() {
        let store = store_with(&["alpha beta gamma delta"]).await;

        // One word of four matches, so the score is 0.25.
        let loose = store
            .search(&SearchQuery::new("alpha nope nope nope"))
            .await
            .unwrap();
        assert_eq!(loose.len(), 1);

        let strict = store
            .search(&SearchQuery::new("alpha nope nope nope").with_min_score(0.5))
            .await
            .unwrap();
        assert!(strict.is_empty(), "the weak match should be filtered out");
    }

    #[tokio::test]
    async fn non_matching_records_are_excluded() {
        let store = store_with(&["completely unrelated"]).await;
        let results = store.search(&SearchQuery::new("zebra")).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_is_case_insensitive() {
        let store = store_with(&["The User Prefers Dark Mode"]).await;
        let results = store.search(&SearchQuery::new("dark MODE")).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn a_read_only_store_refuses_writes() {
        let store = InMemoryMemoryStore::new(MemoryStoreConfig {
            writable: false,
            ..Default::default()
        });

        let err = store.add(MemoryRecord::new("x")).await;
        assert!(err.is_err(), "a read-only store must refuse writes");
    }

    #[tokio::test]
    async fn add_messages_stores_text_and_skips_empty() {
        let store = InMemoryMemoryStore::default();
        let messages = vec![
            Message::user("remember this"),
            Message::assistant(vec![]),
        ];

        let stored = store.add_messages(&messages).await.unwrap();
        assert_eq!(stored, 1, "an empty message contributes nothing to memory");
        assert_eq!(store.len().await, 1);
    }

    #[tokio::test]
    async fn records_carry_metadata() {
        let store = InMemoryMemoryStore::default();
        store
            .add(MemoryRecord::new("fact").with_metadata("topic", "prefs"))
            .await
            .unwrap();

        let results = store.search(&SearchQuery::new("fact")).await.unwrap();
        assert_eq!(
            results[0].record.metadata.get("topic"),
            Some(&Value::String("prefs".into()))
        );
    }

    #[test]
    fn scoring_is_a_fraction_of_query_words_matched() {
        assert_eq!(InMemoryMemoryStore::score("a b", "a b c"), 1.0);
        assert_eq!(InMemoryMemoryStore::score("a b", "a"), 0.5);
        assert_eq!(InMemoryMemoryStore::score("a b", "z"), 0.0);
        assert_eq!(InMemoryMemoryStore::score("", "anything"), 0.0);
    }
}
