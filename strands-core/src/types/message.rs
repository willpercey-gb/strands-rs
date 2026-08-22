use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::content::ContentBlock;
use super::streaming::{Metrics, Usage};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    System,
}

/// Bookkeeping attached to a message.
///
/// Never sent to providers — stripped before model calls — but persisted
/// alongside the message in session storage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageMetadata {
    /// Token usage from the model response that produced this message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Performance metrics from the model response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Metrics>,
    /// Arbitrary caller/framework metadata, e.g. compression provenance.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub custom: HashMap<String, serde_json::Value>,
}

impl MessageMetadata {
    pub fn is_empty(&self) -> bool {
        self.usage.is_none() && self.metrics.is_none() && self.custom.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
    /// Durable identifier for this message.
    ///
    /// Survives session save/restore and is stripped before model calls. The
    /// agent assigns one automatically, so callers do not normally set it; one
    /// supplying its own should use a UUID v4. The same message carries the
    /// same id everywhere it is observed, which is what makes it usable as a
    /// join key against externally stored per-message data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tracking_id: Option<String>,
    /// Optional metadata; stripped before model calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<MessageMetadata>,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self::new(
            Role::User,
            vec![ContentBlock::Text { text: text.into() }],
        )
    }

    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// Build a message with no tracking id or metadata assigned yet.
    pub fn new(role: Role, content: Vec<ContentBlock>) -> Self {
        Self {
            role,
            content,
            tracking_id: None,
            metadata: None,
        }
    }

    /// Assign a durable tracking id if the message does not already carry one.
    ///
    /// A message restored from a session, or given an id by the caller, keeps
    /// it — re-keying would break every reference held against the old id.
    /// Returns the id in effect afterwards.
    pub fn ensure_tracking_id(&mut self) -> &str {
        if self
            .tracking_id
            .as_deref()
            .map(str::is_empty)
            .unwrap_or(true)
        {
            self.tracking_id = Some(uuid::Uuid::new_v4().to_string());
        }
        self.tracking_id.as_deref().unwrap()
    }

    /// Builder-style tracking id assignment.
    pub fn with_tracking_id(mut self, id: impl Into<String>) -> Self {
        self.tracking_id = Some(id.into());
        self
    }

    /// Attach metadata.
    pub fn with_metadata(mut self, metadata: MessageMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// A copy stripped of everything providers must not receive.
    ///
    /// Tracking ids and metadata are SDK bookkeeping; sending them upstream at
    /// best wastes tokens and at worst is rejected as an unknown field.
    pub fn for_model(&self) -> Message {
        Message {
            role: self.role,
            content: self.content.clone(),
            tracking_id: None,
            metadata: None,
        }
    }

    /// Extract all text content from this message, concatenated.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| b.as_text())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Whether this message carries at least one `ToolUse` block.
    pub fn has_tool_use(&self) -> bool {
        self.content.iter().any(|b| b.is_tool_use())
    }

    /// Whether this message carries at least one `ToolResult` block.
    pub fn has_tool_result(&self) -> bool {
        self.content.iter().any(|b| b.is_tool_result())
    }

    /// Return all tool use blocks in this message.
    pub fn tool_uses(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse {
                    tool_use_id,
                    name,
                    input,
                } => Some((tool_use_id.as_str(), name.as_str(), input)),
                _ => None,
            })
            .collect()
    }
}

/// Strip SDK bookkeeping from a slice of messages before a model call.
pub fn messages_for_model(messages: &[Message]) -> Vec<Message> {
    messages.iter().map(Message::for_model).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracking_id_is_assigned_once_and_then_stable() {
        let mut m = Message::user("hi");
        assert!(m.tracking_id.is_none());

        let first = m.ensure_tracking_id().to_string();
        let second = m.ensure_tracking_id().to_string();
        assert_eq!(first, second, "id must not be re-keyed");
        assert_eq!(first.len(), 36, "expected a UUID v4");
    }

    #[test]
    fn caller_supplied_tracking_id_is_preserved() {
        let mut m = Message::user("hi").with_tracking_id("mine");
        assert_eq!(m.ensure_tracking_id(), "mine");
    }

    #[test]
    fn empty_tracking_id_is_treated_as_absent() {
        let mut m = Message::user("hi").with_tracking_id("");
        assert_ne!(m.ensure_tracking_id(), "");
    }

    #[test]
    fn for_model_strips_bookkeeping_but_keeps_content() {
        let mut m = Message::user("hello");
        m.ensure_tracking_id();
        m.metadata = Some(MessageMetadata {
            usage: Some(Usage::default()),
            ..Default::default()
        });

        let stripped = m.for_model();
        assert!(stripped.tracking_id.is_none());
        assert!(stripped.metadata.is_none());
        assert_eq!(stripped.text(), "hello");
        assert_eq!(stripped.role, Role::User);
    }

    #[test]
    fn tracking_id_survives_a_serde_round_trip() {
        let mut m = Message::user("hi");
        let id = m.ensure_tracking_id().to_string();

        let json = serde_json::to_string(&m).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tracking_id.as_deref(), Some(id.as_str()));
    }

    #[test]
    fn legacy_messages_without_tracking_id_still_deserialize() {
        // Sessions written before tracking ids existed must still load.
        let json = r#"{"role":"user","content":[{"type":"text","text":"hi"}]}"#;
        let m: Message = serde_json::from_str(json).unwrap();
        assert!(m.tracking_id.is_none());
        assert_eq!(m.text(), "hi");
    }
}
