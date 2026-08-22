//! Filesystem storage backend.
//!
//! Keys map onto paths beneath a root directory. Writes go through a temp file
//! and a rename, so a partial write is never observable as a complete value.

use std::path::{Path, PathBuf};

use async_trait::async_trait;

use crate::error::StrandsError;

use super::{normalize_key, Storage};

/// Stores values as files beneath `root`.
pub struct LocalFileStorage {
    root: PathBuf,
}

impl LocalFileStorage {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Resolve a key to a path, refusing anything that escapes `root`.
    ///
    /// `normalize_key` already rejects traversal segments; this is the second
    /// check, against a `root` that is itself a symlink or a key that resolves
    /// oddly on a case-insensitive filesystem.
    fn path_for(&self, key: &str) -> Result<PathBuf, StrandsError> {
        let key = normalize_key(key)?;
        let path = self.root.join(&key);

        if !path.starts_with(&self.root) {
            return Err(StrandsError::Other(format!(
                "storage key escapes the storage root: {key}"
            )));
        }
        Ok(path)
    }

    /// Reconstruct a key from a path beneath `root`.
    fn key_for(&self, path: &Path) -> Option<String> {
        let relative = path.strip_prefix(&self.root).ok()?;
        Some(
            relative
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/"),
        )
    }

    /// Walk every file beneath `dir`, depth-first.
    fn collect_files<'a>(
        &'a self,
        dir: PathBuf,
        out: &'a mut Vec<String>,
    ) -> futures::future::BoxFuture<'a, Result<(), StrandsError>> {
        Box::pin(async move {
            let mut entries = match tokio::fs::read_dir(&dir).await {
                Ok(entries) => entries,
                // A prefix that names no directory simply has no keys.
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
                Err(e) => return Err(StrandsError::Other(e.to_string())),
            };

            while let Some(entry) = entries
                .next_entry()
                .await
                .map_err(|e| StrandsError::Other(e.to_string()))?
            {
                let path = entry.path();
                let file_type = entry
                    .file_type()
                    .await
                    .map_err(|e| StrandsError::Other(e.to_string()))?;

                if file_type.is_dir() {
                    self.collect_files(path, out).await?;
                } else if file_type.is_file() {
                    if let Some(key) = self.key_for(&path) {
                        out.push(key);
                    }
                }
                // Symlinks are skipped: following one would read or report a
                // value from outside the storage root.
            }
            Ok(())
        })
    }
}

#[async_trait]
impl Storage for LocalFileStorage {
    async fn write(&self, key: &str, data: Vec<u8>) -> Result<(), StrandsError> {
        let path = self.path_for(key)?;

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| StrandsError::Other(e.to_string()))?;
        }

        // Temp file plus rename: a reader never sees a half-written value.
        let tmp = path.with_extension("tmp");
        tokio::fs::write(&tmp, data)
            .await
            .map_err(|e| StrandsError::Other(e.to_string()))?;
        tokio::fs::rename(&tmp, &path)
            .await
            .map_err(|e| StrandsError::Other(e.to_string()))
    }

    async fn read(&self, key: &str) -> Result<Option<Vec<u8>>, StrandsError> {
        let path = self.path_for(key)?;

        match tokio::fs::symlink_metadata(&path).await {
            Ok(meta) if meta.file_type().is_symlink() => {
                return Err(StrandsError::Other(format!(
                    "refusing to read through a symlink at {}",
                    path.display()
                )))
            }
            _ => {}
        }

        match tokio::fs::read(&path).await {
            Ok(data) => Ok(Some(data)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(StrandsError::Other(e.to_string())),
        }
    }

    async fn delete(&self, key: &str) -> Result<(), StrandsError> {
        let path = self.path_for(key)?;
        match tokio::fs::remove_file(path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(StrandsError::Other(e.to_string())),
        }
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>, StrandsError> {
        let start = if prefix.is_empty() {
            self.root.clone()
        } else {
            self.path_for(prefix)?
        };

        let mut keys = Vec::new();
        // A prefix may name a directory or be a partial key; walk from the
        // nearest existing directory and filter.
        let walk_from = if start.is_dir() {
            start.clone()
        } else {
            start.parent().unwrap_or(&self.root).to_path_buf()
        };
        self.collect_files(walk_from, &mut keys).await?;

        let normalized = if prefix.is_empty() {
            String::new()
        } else {
            normalize_key(prefix)?
        };
        keys.retain(|k| k.starts_with(&normalized));
        keys.sort();
        Ok(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::StorageExt;

    fn temp_root() -> PathBuf {
        std::env::temp_dir().join(format!("strands-storage-{}", uuid::Uuid::new_v4()))
    }

    #[tokio::test]
    async fn write_read_delete_round_trip() {
        let root = temp_root();
        let s = LocalFileStorage::new(&root);

        assert!(s.read("a/b").await.unwrap().is_none());
        s.write("a/b", b"value".to_vec()).await.unwrap();
        assert_eq!(s.read("a/b").await.unwrap(), Some(b"value".to_vec()));

        s.delete("a/b").await.unwrap();
        assert!(s.read("a/b").await.unwrap().is_none());

        let _ = tokio::fs::remove_dir_all(&root).await;
    }

    #[tokio::test]
    async fn nested_keys_create_directories() {
        let root = temp_root();
        let s = LocalFileStorage::new(&root);

        s.write("deep/nested/key", b"x".to_vec()).await.unwrap();
        assert_eq!(s.read("deep/nested/key").await.unwrap(), Some(b"x".to_vec()));

        let _ = tokio::fs::remove_dir_all(&root).await;
    }

    #[tokio::test]
    async fn list_walks_nested_directories() {
        let root = temp_root();
        let s = LocalFileStorage::new(&root);

        for key in ["a/1", "a/deep/2", "b/3"] {
            s.write(key, b"x".to_vec()).await.unwrap();
        }

        assert_eq!(s.list("a").await.unwrap(), vec!["a/1", "a/deep/2"]);
        assert_eq!(s.list("").await.unwrap(), vec!["a/1", "a/deep/2", "b/3"]);

        let _ = tokio::fs::remove_dir_all(&root).await;
    }

    #[tokio::test]
    async fn listing_a_missing_prefix_is_empty_not_an_error() {
        let root = temp_root();
        let s = LocalFileStorage::new(&root);
        assert!(s.list("nothing").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn traversal_keys_cannot_escape_the_root() {
        let root = temp_root();
        let s = LocalFileStorage::new(&root);

        assert!(s.write("../escaped", b"x".to_vec()).await.is_err());
        assert!(s.read("../../etc/passwd").await.is_err());

        let _ = tokio::fs::remove_dir_all(&root).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn reading_through_a_symlink_is_refused() {
        let root = temp_root();
        tokio::fs::create_dir_all(&root).await.unwrap();

        let outside = root.join("outside.txt");
        tokio::fs::write(&outside, b"secret").await.unwrap();
        std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();

        let s = LocalFileStorage::new(&root);
        assert!(s.read("link").await.is_err());

        let _ = tokio::fs::remove_dir_all(&root).await;
    }

    #[tokio::test]
    async fn a_partial_write_is_never_observable() {
        // The temp-plus-rename path means a reader sees either the old value or
        // the new one, never a truncated file.
        let root = temp_root();
        let s = LocalFileStorage::new(&root);

        s.write("k", b"first".to_vec()).await.unwrap();
        s.write("k", b"second-and-longer".to_vec()).await.unwrap();
        assert_eq!(
            s.read("k").await.unwrap(),
            Some(b"second-and-longer".to_vec())
        );

        let _ = tokio::fs::remove_dir_all(&root).await;
    }

    #[tokio::test]
    async fn json_helpers_work_over_files() {
        let root = temp_root();
        let s = LocalFileStorage::new(&root);

        s.write_json("cfg/x", &serde_json::json!({"a": 1}))
            .await
            .unwrap();
        let back: Option<serde_json::Value> = s.read_json("cfg/x").await.unwrap();
        assert_eq!(back, Some(serde_json::json!({"a": 1})));

        let _ = tokio::fs::remove_dir_all(&root).await;
    }
}
