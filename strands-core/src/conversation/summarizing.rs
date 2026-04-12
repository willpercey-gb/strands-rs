use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use tracing::debug;

use crate::error::StrandsError;
use crate::model::Model;
use crate::types::content::ContentBlock;
use crate::types::message::{Message, Role};
use crate::types::streaming::{DeltaContent, StreamEvent};

use super::ConversationManager;

/// Conversation manager that uses the model to summarize older messages
/// when the conversation exceeds the window size.
///
/// Preserves the most recent messages intact and replaces older messages
/// with a model-generated summary.
pub struct SummarizingConversationManager {
    /// Total message count threshold that triggers summarization.
    pub window_size: usize,
    /// Number of recent messages to always preserve verbatim.
    pub preserve_recent: usize,
    /// Fraction of older messages to summarize (0.0-1.0).
    pub summary_ratio: f32,
    /// The model used to generate summaries.
    model: Arc<dyn Model>,
}

impl SummarizingConversationManager {
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            window_size: 40,
            preserve_recent: 10,
            summary_ratio: 0.3,
            model,
        }
    }

    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    pub fn with_preserve_recent(mut self, count: usize) -> Self {
        self.preserve_recent = count;
        self
    }

    pub fn with_summary_ratio(mut self, ratio: f32) -> Self {
        self.summary_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Summarize a set of messages into a single summary message.
    async fn summarize_messages(&self, messages: &[Message]) -> Result<String, StrandsError> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        // Build a prompt asking the model to summarize the conversation
        let mut conversation_text = String::new();
        for msg in messages {
            let role_label = match msg.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
            };
            let text = msg.text();
            if !text.is_empty() {
                conversation_text.push_str(&format!("{role_label}: {text}\n\n"));
            }
        }

        let summary_prompt = format!(
            "Summarize the following conversation concisely, preserving key facts, \
             decisions, and context that would be needed to continue the conversation. \
             Be brief but comprehensive.\n\n---\n\n{conversation_text}"
        );

        let summary_messages = vec![Message::user(summary_prompt)];
        let system_prompt = "You are a conversation summarizer. Output only the summary, nothing else.";

        let mut stream = self
            .model
            .stream(&summary_messages, Some(system_prompt), &[])
            .await?;

        let mut summary = String::new();
        while let Some(event_result) = stream.next().await {
            if let Ok(StreamEvent::ContentBlockDelta {
                delta: DeltaContent::TextDelta(text),
                ..
            }) = event_result
            {
                summary.push_str(&text);
            }
        }

        Ok(summary)
    }
}

#[async_trait]
impl ConversationManager for SummarizingConversationManager {
    async fn reduce_context(
        &self,
        messages: &mut Vec<Message>,
        _system_prompt: Option<&str>,
    ) -> Result<(), StrandsError> {
        if messages.len() <= self.window_size {
            return Ok(());
        }

        debug!(
            total = messages.len(),
            window = self.window_size,
            preserve = self.preserve_recent,
            "Summarizing conversation context"
        );

        // Split: messages to summarize vs messages to preserve
        let preserve_count = self.preserve_recent.min(messages.len());
        let split_point = messages.len() - preserve_count;

        // How many of the older messages to actually summarize
        let summarize_count = ((split_point as f32) * self.summary_ratio).ceil() as usize;
        let summarize_count = summarize_count.max(1).min(split_point);

        let to_summarize = &messages[..summarize_count];
        let summary_text = self.summarize_messages(to_summarize).await?;

        // Build the new message list:
        // [summary_message] + [remaining older messages] + [preserved recent messages]
        let mut new_messages = Vec::new();

        if !summary_text.is_empty() {
            new_messages.push(Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: format!("[Previous conversation summary]\n{summary_text}"),
                }],
            });
        }

        // Keep any older messages that weren't summarized
        if summarize_count < split_point {
            new_messages.extend_from_slice(&messages[summarize_count..split_point]);
        }

        // Keep all preserved recent messages
        new_messages.extend_from_slice(&messages[split_point..]);

        *messages = new_messages;

        debug!(new_len = messages.len(), "Context reduced via summarization");
        Ok(())
    }
}
