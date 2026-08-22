//! Unified storage — a minimal byte-oriented backend behind one trait.
//!
//! Four operations over opaque bytes. Deliberately small: the memory store, the
//! context offloader and session persistence all need somewhere to put bytes,
//! and giving each its own backend abstraction would mean configuring the same
//! S3 bucket three different ways.
//!
//! Keys are opaque strings. The shipped backends treat `/` as a logical
//! separator, collapse repeated separators, and reject traversal — see
//! [`normalize_key`].
//!
//! Ported from upstream `storage/`.

pub mod local_file;
pub mod memory;

use async_trait::async_trait;

use crate::error::StrandsError;

pub use local_file::LocalFileStorage;
pub use memory::InMemoryStorage;

/// A backend for storing and retrieving raw bytes under string keys.
#[async_trait]
pub trait Storage: Send + Sync {
    /// Store `data` under `key`, replacing any existing value.
    async fn write(&self, key: &str, data: Vec<u8>) -> Result<(), StrandsError>;

    /// Retrieve the bytes stored under `key`, or `None` if absent.
    ///
    /// A missing key is `Ok(None)`, not an error — "not there" is a normal
    /// answer, and forcing callers to pattern-match an error for it makes real
    /// failures easy to swallow.
    async fn read(&self, key: &str) -> Result<Option<Vec<u8>>, StrandsError>;

    /// Delete the value under `key`. A no-op if it does not exist.
    async fn delete(&self, key: &str) -> Result<(), StrandsError>;

    /// Keys beginning with `prefix`, in lexicographic order.
    async fn list(&self, prefix: &str) -> Result<Vec<String>, StrandsError>;
}

/// Validate and canonicalize a storage key.
///
/// Collapses repeated `/`, strips leading and trailing separators, and rejects
/// `.` / `..` segments. The traversal check matters because keys routinely come
/// from model output — a tool result id, a memory topic — and
/// [`LocalFileStorage`] maps them onto real paths.
pub fn normalize_key(key: &str) -> Result<String, StrandsError> {
    if key.is_empty() {
        return Err(StrandsError::Other("storage key cannot be empty".into()));
    }
    if key.contains('\0') {
        return Err(StrandsError::Other(
            "storage key cannot contain a null byte".into(),
        ));
    }

    let segments: Vec<&str> = key.split('/').filter(|s| !s.is_empty()).collect();

    if segments.iter().any(|s| *s == "." || *s == "..") {
        return Err(StrandsError::Other(format!(
            "storage key must not contain '.' or '..' segments: {key}"
        )));
    }
    if segments.is_empty() {
        return Err(StrandsError::Other(format!(
            "storage key has no usable segments: {key}"
        )));
    }

    Ok(segments.join("/"))
}

/// Convenience helpers layered over any [`Storage`].
#[async_trait]
pub trait StorageExt: Storage {
    /// Write a value as JSON.
    async fn write_json<T: serde::Serialize + Sync>(
        &self,
        key: &str,
        value: &T,
    ) -> Result<(), StrandsError> {
        let bytes = serde_json::to_vec(value)?;
        self.write(key, bytes).await
    }

    /// Read a JSON value, or `None` if the key is absent.
    async fn read_json<T: serde::de::DeserializeOwned>(
        &self,
        key: &str,
    ) -> Result<Option<T>, StrandsError> {
        match self.read(key).await? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Whether a value exists under `key`.
    async fn exists(&self, key: &str) -> Result<bool, StrandsError> {
        Ok(self.read(key).await?.is_some())
    }
}

impl<T: Storage + ?Sized> StorageExt for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_keys_pass_through() {
        assert_eq!(normalize_key("a/b/c").unwrap(), "a/b/c");
        assert_eq!(normalize_key("single").unwrap(), "single");
    }

    #[test]
    fn separators_are_collapsed_and_trimmed() {
        assert_eq!(normalize_key("/a//b/").unwrap(), "a/b");
        assert_eq!(normalize_key("///a///").unwrap(), "a");
    }

    #[test]
    fn traversal_segments_are_rejected() {
        // Keys routinely come from model output and LocalFileStorage maps them
        // onto real paths, so this is a boundary not a nicety.
        for bad in ["../escape", "a/../b", "./a", "a/."] {
            assert!(
                normalize_key(bad).is_err(),
                "expected {bad:?} to be rejected"
            );
        }
    }

    #[test]
    fn empty_and_separator_only_keys_are_rejected() {
        assert!(normalize_key("").is_err());
        assert!(normalize_key("/").is_err());
        assert!(normalize_key("///").is_err());
    }

    #[test]
    fn null_bytes_are_rejected() {
        assert!(normalize_key("a\0b").is_err());
    }
}
