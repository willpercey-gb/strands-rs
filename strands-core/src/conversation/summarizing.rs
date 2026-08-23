use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use tracing::debug;

use crate::error::StrandsError;
use crate::model::Model;
use crate::types::content::ContentBlock;
use crate::types::content::SystemPrompt;
use crate::types::message::{Message, Role};
use crate::types::streaming::{DeltaContent, StreamEvent};

use super::{ConversationManager, ProactiveCompression, ReduceContext};

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
    /// When set, summarize once the context window passes the configured
    /// threshold rather than waiting for the message count to exceed
    /// `window_size`.
    proactive: Option<ProactiveCompression>,
}

impl SummarizingConversationManager {
    /// Create a new instance.
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            window_size: 40,
            preserve_recent: 10,
            summary_ratio: 0.3,
            model,
            proactive: None,
        }
    }

    /// Compress proactively once the context window is `threshold` full,
    /// instead of waiting for the message count to cross `window_size`.
    ///
    /// Requires a model that reports a context window limit; where none is
    /// known, only the message-count trigger applies.
    /// Set the proactive compression.
    pub fn with_proactive_compression(mut self, threshold: f64) -> Self {
        self.proactive = Some(ProactiveCompression::new(threshold));
        self
    }

    /// Set the window size.
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set the preserve recent.
    pub fn with_preserve_recent(mut self, count: usize) -> Self {
        self.preserve_recent = count;
        self
    }

    /// Set the summary ratio.
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
        let system_prompt = SystemPrompt::from(
            "You are a conversation summarizer. Output only the summary, nothing else.",
        );

        let mut stream = self
            .model
            .stream(&summary_messages, Some(&system_prompt), &[])
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
        ctx: ReduceContext<'_>,
    ) -> Result<(), StrandsError> {
        // Three independent reasons to summarize: the message count crossed
        // the window, the context window is measurably filling up, or the
        // provider already rejected the request as too large.
        let over_window = messages.len() > self.window_size;
        let over_threshold = self
            .proactive
            .as_ref()
            .is_some_and(|cfg| ctx.should_compress(cfg));

        if !over_window && !over_threshold && !ctx.overflow {
            return Ok(());
        }

        // Overflow recovery must actually shrink something; with too few
        // messages to split there is nothing this manager can do.
        if messages.len() <= 1 {
            return Ok(());
        }

        debug!(
            total = messages.len(),
            window = self.window_size,
            preserve = self.preserve_recent,
            utilization = ?ctx.utilization,
            overflow = ctx.overflow,
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
            new_messages.push(Message::new(
                Role::User,
                vec![ContentBlock::Text {
                    text: format!("[Previous conversation summary]\n{summary_text}"),
                }],
            ));
        }

        // Keep any older messages that weren't summarized
        if summarize_count < split_point {
            new_messages.extend_from_slice(&messages[summarize_count..split_point]);
        }

        // Keep all preserved recent messages
        new_messages.extend_from_slice(&messages[split_point..]);

        *messages = new_messages;

        debug!(
            new_len = messages.len(),
            "Context reduced via summarization"
        );
        Ok(())
    }

    fn proactive_compression(&self) -> Option<ProactiveCompression> {
        self.proactive
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelStream;
    use crate::types::content::SystemPrompt;
    use crate::types::streaming::{StopReason, StreamEvent};
    use crate::types::tools::ToolSpec;
    use futures::stream;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Reports a fixed context window so utilization is computable, and counts
    /// how many times it was asked to summarize.
    struct StubModel {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Model for StubModel {
        async fn stream(
            &self,
            _messages: &[Message],
            _system_prompt: Option<&SystemPrompt>,
            _tool_specs: &[ToolSpec],
        ) -> Result<ModelStream, StrandsError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let events = vec![
                Ok(StreamEvent::ContentBlockStart {
                    index: 0,
                    content_type: crate::types::streaming::ContentBlockType::Text,
                }),
                Ok(StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: DeltaContent::TextDelta("a summary".into()),
                }),
                Ok(StreamEvent::ContentBlockStop { index: 0 }),
                Ok(StreamEvent::MessageStop {
                    stop_reason: StopReason::EndTurn,
                }),
            ];
            Ok(Box::pin(stream::iter(events)))
        }

        fn model_id(&self) -> Option<&str> {
            Some("claude-opus-5")
        }
    }

    fn manager(calls: Arc<AtomicUsize>) -> SummarizingConversationManager {
        SummarizingConversationManager::new(Arc::new(StubModel { calls }))
    }

    #[tokio::test]
    async fn under_all_thresholds_nothing_happens() {
        let calls = Arc::new(AtomicUsize::new(0));
        let cm = manager(calls.clone()).with_window_size(100);
        let mut msgs = vec![Message::user("a"), Message::user("b")];

        cm.reduce_context(&mut msgs, ReduceContext::default())
            .await
            .unwrap();

        assert_eq!(msgs.len(), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 0, "no summarization expected");
    }

    #[tokio::test]
    async fn high_utilization_triggers_compression_below_the_window() {
        let calls = Arc::new(AtomicUsize::new(0));
        // Window of 100 is far above the 4 messages present, so only the
        // utilization trigger can fire here.
        let cm = manager(calls.clone())
            .with_window_size(100)
            .with_preserve_recent(1)
            .with_proactive_compression(0.7);

        let mut msgs = vec![
            Message::user("a"),
            Message::user("b"),
            Message::user("c"),
            Message::user("d"),
        ];

        cm.reduce_context(
            &mut msgs,
            ReduceContext::default().with_utilization(Some(0.9)),
        )
        .await
        .unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1, "expected a summarization");
    }

    #[tokio::test]
    async fn unknown_utilization_does_not_trigger_compression() {
        let calls = Arc::new(AtomicUsize::new(0));
        let cm = manager(calls.clone())
            .with_window_size(100)
            .with_proactive_compression(0.7);

        let mut msgs = vec![Message::user("a"), Message::user("b")];
        cm.reduce_context(&mut msgs, ReduceContext::default())
            .await
            .unwrap();

        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "unknown utilization must not be read as full"
        );
    }

    #[tokio::test]
    async fn overflow_forces_compression_regardless_of_counts() {
        let calls = Arc::new(AtomicUsize::new(0));
        let cm = manager(calls.clone())
            .with_window_size(100)
            .with_preserve_recent(1);

        let mut msgs = vec![Message::user("a"), Message::user("b")];
        cm.reduce_context(&mut msgs, ReduceContext::overflow(None))
            .await
            .unwrap();

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "overflow recovery must reduce even when under the window"
        );
    }

    #[tokio::test]
    async fn proactive_config_is_reported_only_when_enabled() {
        let calls = Arc::new(AtomicUsize::new(0));
        assert!(manager(calls.clone()).proactive_compression().is_none());
        assert!(manager(calls)
            .with_proactive_compression(0.5)
            .proactive_compression()
            .is_some());
    }
}
