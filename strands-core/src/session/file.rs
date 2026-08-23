use std::path::{Path, PathBuf};

use async_trait::async_trait;

use crate::error::StrandsError;
use crate::types::message::Message;

use super::SessionManager;

/// Validate a session id before it is used to build a filesystem path.
///
/// A session id is frequently caller- or model-supplied, so it is untrusted
/// input. Without this check an id like `../../etc/cron.d/x` escapes the
/// storage directory entirely.
fn validate_session_id(session_id: &str) -> Result<(), StrandsError> {
    if session_id.is_empty() {
        return Err(StrandsError::Session(
            "session_id cannot be empty".to_string(),
        ));
    }

    // Must be a bare filename component: no separators, no traversal, no root.
    let is_bare = Path::new(session_id).components().count() == 1
        && !session_id.contains('/')
        && !session_id.contains('\\')
        && session_id != ".."
        && session_id != ".";

    if !is_bare {
        return Err(StrandsError::Session(format!(
            "session_id={session_id} | id cannot contain path separators"
        )));
    }

    Ok(())
}

/// File-based session persistence. Each session is stored as a JSON file.
///
/// Writes are atomic (temp file + rename) and refuse to follow symlinks, so a
/// pre-planted link at the target path cannot redirect session data to
/// somewhere else on the filesystem.
pub struct FileSessionManager {
    base_dir: PathBuf,
}

impl FileSessionManager {
    /// Create a new instance.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    /// Use a user-private default directory, `~/.strands/sessions`.
    ///
    /// Preferred over a shared temp directory, which on multi-user systems is
    /// world-writable and lets another user pre-create session paths.
    /// Set the default dir.
    pub fn with_default_dir() -> Result<Self, StrandsError> {
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .ok_or_else(|| {
                StrandsError::Session(
                    "cannot determine home directory; pass an explicit base_dir".to_string(),
                )
            })?;
        Ok(Self::new(
            PathBuf::from(home).join(".strands").join("sessions"),
        ))
    }

    fn session_path(&self, session_id: &str) -> Result<PathBuf, StrandsError> {
        validate_session_id(session_id)?;
        Ok(self.base_dir.join(format!("{session_id}.json")))
    }

    /// Create the storage directory, restricted to the current user on unix.
    async fn ensure_dir(dir: &Path) -> Result<(), StrandsError> {
        tokio::fs::create_dir_all(dir)
            .await
            .map_err(|e| StrandsError::Session(e.to_string()))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o700);
            // Best-effort: a pre-existing directory the caller deliberately
            // shared is their choice, so a failure here is not fatal.
            let _ = tokio::fs::set_permissions(dir, perms).await;
        }

        Ok(())
    }

    /// Reject a path that is a symlink.
    ///
    /// Reading through one lets an attacker feed the agent arbitrary content;
    /// writing through one lets them redirect session data to a file of their
    /// choosing.
    async fn reject_symlink(path: &Path) -> Result<(), StrandsError> {
        match tokio::fs::symlink_metadata(path).await {
            Ok(meta) if meta.file_type().is_symlink() => Err(StrandsError::Session(format!(
                "Refusing to follow symlink at {}. This may indicate session tampering.",
                path.display()
            ))),
            _ => Ok(()),
        }
    }
}

#[async_trait]
impl SessionManager for FileSessionManager {
    async fn save(&self, session_id: &str, messages: &[Message]) -> Result<(), StrandsError> {
        let path = self.session_path(session_id)?;
        if let Some(parent) = path.parent() {
            Self::ensure_dir(parent).await?;
        }
        Self::reject_symlink(&path).await?;

        let data = serde_json::to_string_pretty(messages)?;

        // Write to a temp file in the same directory, then rename. A partial
        // write can never be observed as a valid session file.
        let tmp = path.with_extension("json.tmp");
        Self::reject_symlink(&tmp).await?;
        tokio::fs::write(&tmp, data)
            .await
            .map_err(|e| StrandsError::Session(e.to_string()))?;
        tokio::fs::rename(&tmp, &path)
            .await
            .map_err(|e| StrandsError::Session(e.to_string()))
    }

    async fn load(&self, session_id: &str) -> Result<Option<Vec<Message>>, StrandsError> {
        let path = self.session_path(session_id)?;
        Self::reject_symlink(&path).await?;

        match tokio::fs::read_to_string(&path).await {
            Ok(data) => Ok(Some(serde_json::from_str(&data)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(StrandsError::Session(e.to_string())),
        }
    }

    async fn delete(&self, session_id: &str) -> Result<(), StrandsError> {
        let path = self.session_path(session_id)?;
        match tokio::fs::remove_file(path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(StrandsError::Session(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_ids_are_accepted() {
        assert!(validate_session_id("abc-123").is_ok());
        assert!(validate_session_id("default").is_ok());
    }

    #[test]
    fn traversal_ids_are_rejected() {
        for bad in ["../escape", "a/b", "..", ".", "", "/abs", "a\\b"] {
            assert!(
                validate_session_id(bad).is_err(),
                "expected {bad:?} to be rejected"
            );
        }
    }

    #[tokio::test]
    async fn save_load_delete_round_trip() {
        let dir = std::env::temp_dir().join(format!("strands-sess-{}", uuid::Uuid::new_v4()));
        let sm = FileSessionManager::new(&dir);

        assert!(sm.load("s1").await.unwrap().is_none());

        sm.save("s1", &[Message::user("hello")]).await.unwrap();
        let loaded = sm.load("s1").await.unwrap().expect("session present");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].text(), "hello");

        sm.delete("s1").await.unwrap();
        assert!(sm.load("s1").await.unwrap().is_none());

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn traversal_id_cannot_escape_the_storage_dir() {
        let dir = std::env::temp_dir().join(format!("strands-sess-{}", uuid::Uuid::new_v4()));
        let sm = FileSessionManager::new(&dir);

        let err = sm.save("../escaped", &[Message::user("x")]).await;
        assert!(err.is_err(), "traversal id must not be written");
        assert!(sm.load("../escaped").await.is_err());
        assert!(sm.delete("../escaped").await.is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn save_refuses_to_follow_a_symlink() {
        let dir = std::env::temp_dir().join(format!("strands-sess-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();

        let target = dir.join("target.txt");
        tokio::fs::write(&target, "original").await.unwrap();
        std::os::unix::fs::symlink(&target, dir.join("s1.json")).unwrap();

        let sm = FileSessionManager::new(&dir);
        let result = sm.save("s1", &[Message::user("payload")]).await;

        assert!(result.is_err(), "expected symlink write to be refused");
        assert_eq!(
            tokio::fs::read_to_string(&target).await.unwrap(),
            "original",
            "symlink target must not have been overwritten"
        );

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }
}
