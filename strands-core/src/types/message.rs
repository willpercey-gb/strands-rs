use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::content::ContentBlock;
use super::streaming::{Metrics, Usage};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Who is speaking in a message.
pub enum Role {
    /// The human, or a tool result standing in for one.
    User,
    /// The model.
    Assistant,
    /// System-level instruction. Most providers take this separately.
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
    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.usage.is_none() && self.metrics.is_none() && self.custom.is_empty()
    }
}

/// One turn in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Who is speaking.
    pub role: Role,
    /// The message body.
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
    /// A user message containing `text`.
    pub fn user(text: impl Into<String>) -> Self {
        Self::new(Role::User, vec![ContentBlock::Text { text: text.into() }])
    }

    /// An assistant message with the given content.
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
    /// Set the tracking id.
    pub fn with_tracking_id(mut self, id: impl Into<String>) -> Self {
        self.tracking_id = Some(id.into());
        self
    }

    /// Attach metadata.
    /// Set the metadata.
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

/// Clean up a message truncated by the model's output cap.
///
/// A response cut short at `max_tokens` can leave a `ToolUse` whose input was
/// never finished. Executing it would run the tool with arguments the model did
/// not actually choose, so every tool use is replaced with text explaining what
/// happened — complete-looking ones included, since there is no way to tell a
/// finished call from one truncated at exactly the right byte.
///
/// Non-tool content is preserved: whatever the model did manage to say is still
/// worth keeping.
///
/// Ported from upstream `event_loop/_recover_message_on_max_tokens_reached.py`.
pub fn recover_message_on_max_tokens(message: &Message) -> Message {
    let content = message
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::ToolUse { name, .. } => ContentBlock::Text {
                text: format!(
                    "The selected tool {name}'s tool use was incomplete due to \
                     maximum token limits being reached."
                ),
            },
            other => other.clone(),
        })
        .collect();

    Message {
        role: message.role,
        content,
        tracking_id: message.tracking_id.clone(),
        metadata: message.metadata.clone(),
    }
}

#[cfg(test)]
mod max_tokens_tests {
    use super::*;
    use crate::types::content::ContentBlock;
    use serde_json::json;

    fn truncated() -> Message {
        Message::assistant(vec![
            ContentBlock::Text {
                text: "Let me calculate that".into(),
            },
            ContentBlock::ToolUse {
                tool_use_id: "1".into(),
                name: "calculator".into(),
                input: json!({"expression": "2+"}),
            },
        ])
    }

    #[test]
    fn tool_uses_are_replaced_with_an_explanation() {
        // Executing a half-written call would run the tool with arguments the
        // model never actually chose.
        let recovered = recover_message_on_max_tokens(&truncated());

        assert!(!recovered.has_tool_use());
        assert!(recovered.text().contains("calculator"));
        assert!(recovered.text().contains("maximum token limits"));
    }

    #[test]
    fn other_content_survives() {
        let recovered = recover_message_on_max_tokens(&truncated());
        assert!(
            recovered.text().contains("Let me calculate that"),
            "whatever the model managed to say is still worth keeping"
        );
    }

    #[test]
    fn a_message_without_tool_uses_is_unchanged() {
        let message = Message::assistant(vec![ContentBlock::Text {
            text: "just text".into(),
        }]);
        assert_eq!(recover_message_on_max_tokens(&message).text(), "just text");
    }

    #[test]
    fn bookkeeping_is_preserved() {
        let mut message = truncated();
        let id = message.ensure_tracking_id().to_string();

        let recovered = recover_message_on_max_tokens(&message);
        assert_eq!(recovered.tracking_id.as_deref(), Some(id.as_str()));
        assert_eq!(recovered.role, Role::Assistant);
    }

    #[test]
    fn every_tool_use_is_replaced_not_just_the_last() {
        // There is no way to tell a finished call from one truncated at exactly
        // the right byte, so all of them go.
        let message = Message::assistant(vec![
            ContentBlock::ToolUse {
                tool_use_id: "1".into(),
                name: "first".into(),
                input: json!({}),
            },
            ContentBlock::ToolUse {
                tool_use_id: "2".into(),
                name: "second".into(),
                input: json!({}),
            },
        ]);

        let recovered = recover_message_on_max_tokens(&message);
        assert!(!recovered.has_tool_use());
        assert!(recovered.text().contains("first"));
        assert!(recovered.text().contains("second"));
    }
}
