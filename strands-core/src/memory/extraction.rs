//! Deciding what from a conversation is worth remembering.
//!
//! Storing every message would make memory a second, worse copy of the
//! transcript. Extraction selects the durable facts — preferences, decisions,
//! commitments — and leaves the rest.
//!
//! Ported from upstream `memory/extraction/`.

use async_trait::async_trait;

use crate::error::StrandsError;
use crate::types::message::{Message, Role};

use super::store::MemoryRecord;

/// When extraction should run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractionTrigger {
    /// After every agent invocation.
    EveryInvocation,
    /// Once the conversation has grown by at least this many messages.
    MessageCount(usize),
    /// Only when the caller asks.
    Manual,
}

impl ExtractionTrigger {
    /// Whether extraction should run given how many new messages there are.
    pub fn should_extract(&self, new_messages: usize) -> bool {
        match self {
            ExtractionTrigger::EveryInvocation => new_messages > 0,
            // A zero threshold would fire on an empty batch, producing empty
            // extractions forever.
            ExtractionTrigger::MessageCount(0) => new_messages > 0,
            ExtractionTrigger::MessageCount(n) => new_messages >= *n,
            ExtractionTrigger::Manual => false,
        }
    }
}

/// How extraction behaves for a store.
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// When extraction runs.
    pub trigger: ExtractionTrigger,
    /// Only extract from these roles.
    ///
    /// Defaults to user messages: the assistant's own output is the least
    /// reliable source of durable fact, since remembering it turns a guess into
    /// something the agent later treats as established.
    pub roles: Vec<Role>,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            trigger: ExtractionTrigger::EveryInvocation,
            roles: vec![Role::User],
        }
    }
}

impl ExtractionConfig {
    /// Set the trigger.
    pub fn with_trigger(mut self, trigger: ExtractionTrigger) -> Self {
        self.trigger = trigger;
        self
    }

    /// Set the roles.
    pub fn with_roles(mut self, roles: Vec<Role>) -> Self {
        self.roles = roles;
        self
    }

    /// Messages eligible for extraction under this config.
    pub fn filter<'a>(&self, messages: &'a [Message]) -> Vec<&'a Message> {
        messages
            .iter()
            .filter(|m| self.roles.contains(&m.role) && !m.text().is_empty())
            .collect()
    }
}

/// Turns conversation messages into records worth storing.
#[async_trait]
pub trait MemoryExtractor: Send + Sync {
    /// Turn eligible messages into records worth storing.
    async fn extract(&self, messages: &[&Message]) -> Result<Vec<MemoryRecord>, StrandsError>;
}

/// Stores each eligible message verbatim.
///
/// The baseline when no model-driven extractor is configured. Honest about what
/// it is: no summarization, no deduplication.
#[derive(Debug, Default, Clone, Copy)]
pub struct VerbatimExtractor;

#[async_trait]
impl MemoryExtractor for VerbatimExtractor {
    async fn extract(&self, messages: &[&Message]) -> Result<Vec<MemoryRecord>, StrandsError> {
        Ok(messages
            .iter()
            .map(|m| MemoryRecord::new(m.text()).with_metadata("role", format!("{:?}", m.role)))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_invocation_fires_on_any_new_message() {
        let t = ExtractionTrigger::EveryInvocation;
        assert!(t.should_extract(1));
        assert!(
            !t.should_extract(0),
            "an empty batch has nothing to extract"
        );
    }

    #[test]
    fn message_count_fires_at_the_threshold() {
        let t = ExtractionTrigger::MessageCount(3);
        assert!(!t.should_extract(2));
        assert!(t.should_extract(3));
        assert!(t.should_extract(10));
    }

    #[test]
    fn a_zero_threshold_still_requires_a_message() {
        assert!(!ExtractionTrigger::MessageCount(0).should_extract(0));
        assert!(ExtractionTrigger::MessageCount(0).should_extract(1));
    }

    #[test]
    fn manual_never_fires_automatically() {
        assert!(!ExtractionTrigger::Manual.should_extract(100));
    }

    #[test]
    fn extraction_defaults_to_user_messages_only() {
        // The assistant's own output is the least reliable source of durable
        // fact — remembering it turns a guess into something established.
        let config = ExtractionConfig::default();
        let messages = vec![
            Message::user("I prefer dark mode"),
            Message::assistant(vec![crate::types::content::ContentBlock::Text {
                text: "Noted".into(),
            }]),
        ];

        let filtered = config.filter(&messages);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].text(), "I prefer dark mode");
    }

    #[test]
    fn roles_are_configurable() {
        let config = ExtractionConfig::default().with_roles(vec![Role::User, Role::Assistant]);
        let messages = vec![
            Message::user("a"),
            Message::assistant(vec![crate::types::content::ContentBlock::Text {
                text: "b".into(),
            }]),
        ];
        assert_eq!(config.filter(&messages).len(), 2);
    }

    #[test]
    fn empty_messages_are_never_eligible() {
        let config = ExtractionConfig::default();
        let messages = vec![Message::user("")];
        assert!(config.filter(&messages).is_empty());
    }

    #[tokio::test]
    async fn verbatim_extractor_preserves_text_and_role() {
        let messages = [Message::user("remember this")];
        let refs: Vec<&Message> = messages.iter().collect();

        let records = VerbatimExtractor.extract(&refs).await.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].content, "remember this");
        assert!(records[0].metadata.contains_key("role"));
    }
}
