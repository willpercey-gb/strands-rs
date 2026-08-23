//! In-memory storage backend.
//!
//! For tests, and for offloading that only needs to outlive one run.

use std::collections::BTreeMap;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::error::StrandsError;

use super::{normalize_key, Storage};

/// Keeps values in a `BTreeMap`, so `list` is naturally ordered.
#[derive(Default)]
pub struct InMemoryStorage {
    values: RwLock<BTreeMap<String, Vec<u8>>>,
}

impl InMemoryStorage {
    /// Create with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of entries.
    pub async fn len(&self) -> usize {
        self.values.read().await.len()
    }

    /// Whether there are no entries.
    pub async fn is_empty(&self) -> bool {
        self.values.read().await.is_empty()
    }

    /// Remove every entry.
    pub async fn clear(&self) {
        self.values.write().await.clear();
    }
}

#[async_trait]
impl Storage for InMemoryStorage {
    async fn write(&self, key: &str, data: Vec<u8>) -> Result<(), StrandsError> {
        let key = normalize_key(key)?;
        self.values.write().await.insert(key, data);
        Ok(())
    }

    async fn read(&self, key: &str) -> Result<Option<Vec<u8>>, StrandsError> {
        let key = normalize_key(key)?;
        Ok(self.values.read().await.get(&key).cloned())
    }

    async fn delete(&self, key: &str) -> Result<(), StrandsError> {
        let key = normalize_key(key)?;
        self.values.write().await.remove(&key);
        Ok(())
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>, StrandsError> {
        // An empty prefix means "everything", so it bypasses key validation —
        // which would otherwise reject it as having no usable segments.
        let normalized = if prefix.is_empty() {
            String::new()
        } else {
            normalize_key(prefix)?
        };

        Ok(self
            .values
            .read()
            .await
            .keys()
            .filter(|k| k.starts_with(&normalized))
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::StorageExt;

    #[tokio::test]
    async fn write_read_delete_round_trip() {
        let s = InMemoryStorage::new();
        assert!(s.read("k").await.unwrap().is_none());

        s.write("k", b"value".to_vec()).await.unwrap();
        assert_eq!(s.read("k").await.unwrap(), Some(b"value".to_vec()));

        s.delete("k").await.unwrap();
        assert!(s.read("k").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn a_missing_key_reads_as_none_not_an_error() {
        let s = InMemoryStorage::new();
        assert!(s.read("absent").await.is_ok());
    }

    #[tokio::test]
    async fn deleting_a_missing_key_is_a_no_op() {
        let s = InMemoryStorage::new();
        assert!(s.delete("absent").await.is_ok());
    }

    #[tokio::test]
    async fn write_replaces_an_existing_value() {
        let s = InMemoryStorage::new();
        s.write("k", b"first".to_vec()).await.unwrap();
        s.write("k", b"second".to_vec()).await.unwrap();
        assert_eq!(s.read("k").await.unwrap(), Some(b"second".to_vec()));
        assert_eq!(s.len().await, 1);
    }

    #[tokio::test]
    async fn list_filters_by_prefix_and_is_ordered() {
        let s = InMemoryStorage::new();
        for key in ["b/2", "a/1", "b/1", "c/1"] {
            s.write(key, b"x".to_vec()).await.unwrap();
        }

        assert_eq!(s.list("b").await.unwrap(), vec!["b/1", "b/2"]);
        assert_eq!(
            s.list("").await.unwrap(),
            vec!["a/1", "b/1", "b/2", "c/1"],
            "an empty prefix lists everything"
        );
    }

    #[tokio::test]
    async fn keys_are_normalized_consistently_across_operations() {
        let s = InMemoryStorage::new();
        s.write("/a//b/", b"x".to_vec()).await.unwrap();
        assert_eq!(s.read("a/b").await.unwrap(), Some(b"x".to_vec()));
        assert_eq!(s.list("a").await.unwrap(), vec!["a/b"]);
    }

    #[tokio::test]
    async fn traversal_keys_are_refused() {
        let s = InMemoryStorage::new();
        assert!(s.write("../escape", b"x".to_vec()).await.is_err());
        assert!(s.read("../escape").await.is_err());
    }

    #[tokio::test]
    async fn json_helpers_round_trip() {
        let s = InMemoryStorage::new();
        s.write_json("cfg", &serde_json::json!({"n": 1}))
            .await
            .unwrap();

        let back: Option<serde_json::Value> = s.read_json("cfg").await.unwrap();
        assert_eq!(back, Some(serde_json::json!({"n": 1})));
        assert!(s.exists("cfg").await.unwrap());
        assert!(!s.exists("other").await.unwrap());
    }
}
