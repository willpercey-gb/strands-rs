use std::path::PathBuf;

use async_trait::async_trait;

use crate::error::StrandsError;
use crate::types::message::Message;

use super::SessionManager;

/// File-based session persistence. Each session is stored as a JSON file.
pub struct FileSessionManager {
    base_dir: PathBuf,
}

impl FileSessionManager {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }
}

#[async_trait]
impl SessionManager for FileSessionManager {
    async fn save(&self, session_id: &str, messages: &[Message]) -> Result<(), StrandsError> {
        let path = self.base_dir.join(format!("{session_id}.json"));
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| StrandsError::Session(e.to_string()))?;
        }
        let data = serde_json::to_string_pretty(messages)?;
        tokio::fs::write(path, data)
            .await
            .map_err(|e| StrandsError::Session(e.to_string()))
    }

    async fn load(&self, session_id: &str) -> Result<Option<Vec<Message>>, StrandsError> {
        let path = self.base_dir.join(format!("{session_id}.json"));
        match tokio::fs::read_to_string(&path).await {
            Ok(data) => Ok(Some(serde_json::from_str(&data)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(StrandsError::Session(e.to_string())),
        }
    }

    async fn delete(&self, session_id: &str) -> Result<(), StrandsError> {
        let path = self.base_dir.join(format!("{session_id}.json"));
        match tokio::fs::remove_file(path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(StrandsError::Session(e.to_string())),
        }
    }
}
