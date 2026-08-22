use async_trait::async_trait;
use tracing::{debug, warn};

use crate::error::StrandsError;
use crate::types::message::Message;

use super::pin::{apply_pin_first, is_pinned};
use super::trim::{find_tool_pair_trim_point, find_valid_trim_point};
use super::{ConversationManager, ReduceContext};

/// Keeps a fixed window of recent messages, dropping the oldest when the
/// limit is exceeded.
///
/// Trimming never simply drops the oldest N messages: doing so can leave a
/// `ToolResult` without its `ToolUse` (or vice versa), which providers
/// reject. The window is instead reduced to the nearest boundary the provider
/// will accept, which may retain slightly more than `window_size` messages.
pub struct SlidingWindowConversationManager {
    /// Maximum number of messages to retain.
    ///
    /// `0` means "clear all messages on every reduction".
    pub window_size: usize,
    /// Number of opening messages to pin permanently on first reduction.
    ///
    /// Pinned messages are never evicted, so the conversation's framing
    /// survives however long the exchange runs.
    pub pin_first: Option<usize>,
    /// Whether `pin_first` has already been applied.
    pin_first_applied: std::sync::atomic::AtomicBool,
}

impl SlidingWindowConversationManager {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            pin_first: None,
            pin_first_applied: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Permanently pin the first `count` messages.
    pub fn with_pin_first(mut self, count: usize) -> Self {
        self.pin_first = Some(count);
        self
    }
}

impl Default for SlidingWindowConversationManager {
    fn default() -> Self {
        Self::new(40)
    }
}

#[async_trait]
impl ConversationManager for SlidingWindowConversationManager {
    async fn reduce_context(
        &self,
        messages: &mut Vec<Message>,
        _ctx: ReduceContext<'_>,
    ) -> Result<(), StrandsError> {
        use std::sync::atomic::Ordering;

        // Apply the opening pins once, before any reduction can evict them.
        if let Some(count) = self.pin_first {
            if !self.pin_first_applied.swap(true, Ordering::Relaxed) {
                apply_pin_first(messages, count);
            }
        }

        // window_size == 0 means "drop everything that isn't pinned".
        if self.window_size == 0 {
            // Compute pins up-front: `is_pinned` inspects neighbours, so it
            // cannot be evaluated while the list is being mutated.
            let pinned: Vec<bool> = (0..messages.len())
                .map(|i| is_pinned(messages, i))
                .collect();
            let mut index = 0;
            messages.retain(|_| {
                let keep = pinned[index];
                index += 1;
                keep
            });
            return Ok(());
        }

        if messages.len() <= self.window_size {
            return Ok(());
        }

        let start_index = messages.len() - self.window_size;

        // Walk forward to a boundary the provider will accept.
        let mut trim_index = find_valid_trim_point(messages, start_index);

        if trim_index >= messages.len() {
            // No plain user message available. Fall back to an
            // assistant(ToolUse) + user(ToolResult) boundary — providers treat
            // a complete pair as a valid continuation, and without this a
            // tool-heavy conversation could never be trimmed at all.
            match find_tool_pair_trim_point(messages, start_index) {
                Some(fallback) => {
                    debug!(
                        trim_index = fallback,
                        "No plain user message trim point; using tool-pair boundary"
                    );
                    trim_index = fallback;
                }
                None => {
                    warn!(
                        window_size = self.window_size,
                        message_count = messages.len(),
                        "Unable to trim conversation context: no valid trim point found"
                    );
                    return Ok(());
                }
            }
        }

        if trim_index == 0 {
            return Ok(());
        }

        // Drop everything before the trim point except pinned messages.
        let doomed: std::collections::HashSet<usize> =
            super::pin::unpinned_indices(messages, 0..trim_index)
                .into_iter()
                .collect();

        if doomed.is_empty() {
            warn!(
                window_size = self.window_size,
                message_count = messages.len(),
                "Every message in the trim range is pinned; unable to reduce"
            );
            return Ok(());
        }

        let mut index = 0;
        messages.retain(|_| {
            let keep = !doomed.contains(&index);
            index += 1;
            keep
        });

        debug!(
            removed = doomed.len(),
            remaining = messages.len(),
            "Reduced context via sliding window"
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::content::{ContentBlock, ToolResultContent, ToolResultStatus};
    use crate::types::message::Role;
    use serde_json::json;

    fn assistant_tool_use(id: &str) -> Message {
        Message::assistant(vec![ContentBlock::ToolUse {
            tool_use_id: id.into(),
            name: "t".into(),
            input: json!({}),
        }])
    }

    fn user_tool_result(id: &str) -> Message {
        Message::new(
            Role::User,
            vec![ContentBlock::ToolResult {
                tool_use_id: id.into(),
                status: ToolResultStatus::Success,
                content: vec![ToolResultContent::Text { text: "ok".into() }],
            }],
        )
    }

    #[tokio::test]
    async fn window_size_zero_clears_history() {
        let cm = SlidingWindowConversationManager::new(0);
        let mut msgs = vec![Message::user("a"), Message::user("b")];
        cm.reduce_context(&mut msgs, ReduceContext::default()).await.unwrap();
        assert!(msgs.is_empty());
    }

    #[tokio::test]
    async fn under_window_is_untouched() {
        let cm = SlidingWindowConversationManager::new(10);
        let mut msgs = vec![Message::user("a"), Message::user("b")];
        cm.reduce_context(&mut msgs, ReduceContext::default()).await.unwrap();
        assert_eq!(msgs.len(), 2);
    }

    #[tokio::test]
    async fn trim_does_not_orphan_a_tool_result() {
        // A naive drop of the oldest 2 would leave the history starting on a
        // ToolResult whose ToolUse is gone — which providers reject.
        let cm = SlidingWindowConversationManager::new(2);
        let mut msgs = vec![
            Message::user("start"),
            assistant_tool_use("1"),
            user_tool_result("1"),
            Message::user("next"),
        ];
        cm.reduce_context(&mut msgs, ReduceContext::default()).await.unwrap();

        assert!(
            !msgs[0].has_tool_result(),
            "history must not begin with an orphaned ToolResult: {msgs:?}"
        );
        assert_eq!(msgs[0].text(), "next");
    }

    #[tokio::test]
    async fn tool_only_conversation_still_trims_via_pair_boundary() {
        // No plain user message exists after the start index, so the
        // tool-pair fallback is the only way to reduce this at all.
        let cm = SlidingWindowConversationManager::new(2);
        let mut msgs = vec![
            assistant_tool_use("1"),
            user_tool_result("1"),
            assistant_tool_use("2"),
            user_tool_result("2"),
        ];
        let before = msgs.len();
        cm.reduce_context(&mut msgs, ReduceContext::default()).await.unwrap();

        assert!(msgs.len() < before, "expected a reduction");
        assert!(msgs[0].has_tool_use());
        assert!(msgs[1].has_tool_result());
    }

    #[tokio::test]
    async fn pinned_messages_survive_trimming() {
        let cm = SlidingWindowConversationManager::new(2).with_pin_first(1);
        let mut msgs = vec![
            Message::user("framing instructions"),
            Message::user("b"),
            Message::user("c"),
            Message::user("d"),
        ];
        cm.reduce_context(&mut msgs, ReduceContext::default()).await.unwrap();

        assert_eq!(
            msgs[0].text(),
            "framing instructions",
            "the pinned opening message must survive: {msgs:?}"
        );
        assert!(msgs.len() < 4, "everything else should still be reduced");
    }

    #[tokio::test]
    async fn window_size_zero_still_keeps_pins() {
        let cm = SlidingWindowConversationManager::new(0).with_pin_first(1);
        let mut msgs = vec![Message::user("keep"), Message::user("drop")];
        cm.reduce_context(&mut msgs, ReduceContext::default()).await.unwrap();

        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].text(), "keep");
    }

    #[tokio::test]
    async fn pin_first_is_applied_only_once() {
        // Applying it on every reduction would silently re-pin messages that
        // have since shifted into the first slots.
        let cm = SlidingWindowConversationManager::new(2).with_pin_first(1);

        let mut msgs = vec![Message::user("a"), Message::user("b"), Message::user("c")];
        cm.reduce_context(&mut msgs, ReduceContext::default()).await.unwrap();
        let pinned_after_first = super::super::pin::is_pinned(&msgs, 0);

        cm.reduce_context(&mut msgs, ReduceContext::default()).await.unwrap();
        let newly_pinned = (1..msgs.len()).any(|i| super::super::pin::is_pinned(&msgs, i));

        assert!(pinned_after_first);
        assert!(!newly_pinned, "later messages must not be pinned retroactively");
    }

    #[tokio::test]
    async fn a_fully_pinned_trim_range_is_left_intact() {
        let cm = SlidingWindowConversationManager::new(1).with_pin_first(3);
        let mut msgs = vec![Message::user("a"), Message::user("b"), Message::user("c")];
        cm.reduce_context(&mut msgs, ReduceContext::default()).await.unwrap();
        assert_eq!(msgs.len(), 3, "nothing is evictable, so nothing is evicted");
    }

    #[tokio::test]
    async fn untrimmable_history_is_left_intact_rather_than_corrupted() {
        let cm = SlidingWindowConversationManager::new(1);
        let mut msgs = vec![assistant_tool_use("1"), user_tool_result("1")];
        cm.reduce_context(&mut msgs, ReduceContext::default()).await.unwrap();
        // Better to exceed the window than to emit an invalid history.
        assert_eq!(msgs.len(), 2);
    }
}
