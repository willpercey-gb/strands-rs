//! Coordinating memory across an agent's lifetime.
//!
//! Ported from upstream `memory/memory_manager.py`.

use std::sync::Arc;

use tracing::{debug, warn};

use crate::error::StrandsError;
use crate::types::message::Message;

use super::extraction::{ExtractionConfig, MemoryExtractor, VerbatimExtractor};
use super::store::{MemoryStore, SearchQuery, SearchResult};

/// Ties a store, an extractor and a trigger policy together.
pub struct MemoryManager {
    store: Arc<dyn MemoryStore>,
    extractor: Arc<dyn MemoryExtractor>,
    config: ExtractionConfig,
    /// How many messages had already been seen at the last extraction.
    watermark: std::sync::atomic::AtomicUsize,
}

impl MemoryManager {
    pub fn new(store: Arc<dyn MemoryStore>) -> Self {
        Self {
            store,
            extractor: Arc::new(VerbatimExtractor),
            config: ExtractionConfig::default(),
            watermark: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn with_extractor(mut self, extractor: Arc<dyn MemoryExtractor>) -> Self {
        self.extractor = extractor;
        self
    }

    pub fn with_config(mut self, config: ExtractionConfig) -> Self {
        self.config = config;
        self
    }

    pub fn store(&self) -> &Arc<dyn MemoryStore> {
        &self.store
    }

    /// Prepare the store before the agent runs.
    pub async fn initialize(&self) -> Result<(), StrandsError> {
        self.store.initialize().await
    }

    /// Recall records relevant to `query`.
    pub async fn recall(&self, query: &str) -> Result<Vec<SearchResult>, StrandsError> {
        self.store.search(&SearchQuery::new(query)).await
    }

    /// Extract and store anything new in `messages`, if the trigger says so.
    ///
    /// Only messages past the watermark are considered, so repeated calls over
    /// a growing history do not re-store what was already remembered.
    ///
    /// Returns how many records were stored.
    pub async fn maybe_extract(&self, messages: &[Message]) -> Result<usize, StrandsError> {
        use std::sync::atomic::Ordering;

        let seen = self.watermark.load(Ordering::Relaxed);
        let fresh = messages.get(seen..).unwrap_or(&[]);

        if !self.config.trigger.should_extract(fresh.len()) {
            return Ok(0);
        }

        let eligible = self.config.filter(fresh);
        if eligible.is_empty() {
            // Still advance: these messages were considered and rejected, and
            // re-examining them on every turn would be pure waste.
            self.watermark.store(messages.len(), Ordering::Relaxed);
            return Ok(0);
        }

        let records = self.extractor.extract(&eligible).await?;
        let mut stored = 0;
        for record in records {
            match self.store.add(record).await {
                Ok(()) => stored += 1,
                Err(e) => {
                    // A store that rejects a write should not fail the agent's
                    // turn — the conversation is still valid without the memory.
                    warn!(error = %e, "Failed to store a memory record");
                }
            }
        }

        self.watermark.store(messages.len(), Ordering::Relaxed);
        debug!(stored, considered = eligible.len(), "Extracted memories");
        Ok(stored)
    }

    /// Force extraction regardless of the trigger.
    pub async fn extract_now(&self, messages: &[Message]) -> Result<usize, StrandsError> {
        use std::sync::atomic::Ordering;

        let eligible = self.config.filter(messages);
        if eligible.is_empty() {
            return Ok(0);
        }

        let records = self.extractor.extract(&eligible).await?;
        let mut stored = 0;
        for record in records {
            if self.store.add(record).await.is_ok() {
                stored += 1;
            }
        }
        self.watermark.store(messages.len(), Ordering::Relaxed);
        Ok(stored)
    }

    /// Reset the watermark, so the next extraction reconsiders everything.
    pub fn reset(&self) {
        self.watermark
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::extraction::ExtractionTrigger;
    use crate::memory::store::{InMemoryMemoryStore, MemoryStoreConfig};

    fn manager() -> (MemoryManager, Arc<InMemoryMemoryStore>) {
        let store = Arc::new(InMemoryMemoryStore::default());
        (MemoryManager::new(store.clone()), store)
    }

    #[tokio::test]
    async fn extraction_stores_user_messages() {
        let (manager, store) = manager();
        let messages = vec![Message::user("I prefer dark mode")];

        assert_eq!(manager.maybe_extract(&messages).await.unwrap(), 1);
        assert_eq!(store.len().await, 1);
    }

    #[tokio::test]
    async fn the_watermark_prevents_re_storing_old_messages() {
        // Without this, every turn would re-remember the whole conversation.
        let (manager, store) = manager();

        let mut messages = vec![Message::user("first")];
        manager.maybe_extract(&messages).await.unwrap();

        messages.push(Message::user("second"));
        assert_eq!(
            manager.maybe_extract(&messages).await.unwrap(),
            1,
            "only the new message should be extracted"
        );
        assert_eq!(store.len().await, 2);
    }

    #[tokio::test]
    async fn re_extracting_an_unchanged_history_stores_nothing() {
        let (manager, store) = manager();
        let messages = vec![Message::user("only one")];

        manager.maybe_extract(&messages).await.unwrap();
        assert_eq!(manager.maybe_extract(&messages).await.unwrap(), 0);
        assert_eq!(store.len().await, 1);
    }

    #[tokio::test]
    async fn the_trigger_is_honoured() {
        let store = Arc::new(InMemoryMemoryStore::default());
        let manager = MemoryManager::new(store.clone())
            .with_config(ExtractionConfig::default().with_trigger(ExtractionTrigger::MessageCount(3)));

        let messages = vec![Message::user("a"), Message::user("b")];
        assert_eq!(manager.maybe_extract(&messages).await.unwrap(), 0);

        let messages = vec![Message::user("a"), Message::user("b"), Message::user("c")];
        assert_eq!(manager.maybe_extract(&messages).await.unwrap(), 3);
    }

    #[tokio::test]
    async fn extract_now_ignores_the_trigger() {
        let store = Arc::new(InMemoryMemoryStore::default());
        let manager = MemoryManager::new(store.clone())
            .with_config(ExtractionConfig::default().with_trigger(ExtractionTrigger::Manual));

        let messages = vec![Message::user("remember")];
        assert_eq!(manager.maybe_extract(&messages).await.unwrap(), 0);
        assert_eq!(manager.extract_now(&messages).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn a_rejecting_store_does_not_fail_the_turn() {
        // Memory is an enhancement; losing a write should not take the
        // conversation down with it.
        let store = Arc::new(InMemoryMemoryStore::new(MemoryStoreConfig {
            writable: false,
            ..Default::default()
        }));
        let manager = MemoryManager::new(store);

        let stored = manager
            .maybe_extract(&[Message::user("something")])
            .await
            .expect("a rejected write must not surface as an error");
        assert_eq!(stored, 0);
    }

    #[tokio::test]
    async fn ineligible_messages_still_advance_the_watermark() {
        // Otherwise every turn would re-examine the same rejected messages.
        let (manager, _) = manager();
        let assistant_only = vec![Message::assistant(vec![
            crate::types::content::ContentBlock::Text { text: "hi".into() },
        ])];

        assert_eq!(manager.maybe_extract(&assistant_only).await.unwrap(), 0);

        let mut messages = assistant_only;
        messages.push(Message::user("now something worth keeping"));
        assert_eq!(manager.maybe_extract(&messages).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn recall_finds_stored_memories() {
        let (manager, _) = manager();
        manager
            .maybe_extract(&[Message::user("the deploy key is in vault")])
            .await
            .unwrap();

        let results = manager.recall("deploy key").await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].record.content.contains("deploy key"));
    }

    #[tokio::test]
    async fn reset_reconsiders_everything() {
        let (manager, store) = manager();
        let messages = vec![Message::user("a")];

        manager.maybe_extract(&messages).await.unwrap();
        manager.reset();
        manager.maybe_extract(&messages).await.unwrap();

        assert_eq!(store.len().await, 2, "after a reset the message is seen again");
    }
}
