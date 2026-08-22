use async_trait::async_trait;
use tracing::{debug, warn};

use crate::error::StrandsError;
use crate::types::content::SystemPrompt;
use crate::types::message::Message;

use super::trim::{find_tool_pair_trim_point, find_valid_trim_point};
use super::ConversationManager;

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
}

impl SlidingWindowConversationManager {
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }
}

impl Default for SlidingWindowConversationManager {
    fn default() -> Self {
        Self { window_size: 40 }
    }
}

#[async_trait]
impl ConversationManager for SlidingWindowConversationManager {
    async fn reduce_context(
        &self,
        messages: &mut Vec<Message>,
        _system_prompt: Option<&SystemPrompt>,
    ) -> Result<(), StrandsError> {
        // window_size == 0 means "drop everything".
        if self.window_size == 0 {
            messages.clear();
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

        if trim_index > 0 {
            debug!(
                removed = trim_index,
                remaining = messages.len() - trim_index,
                "Reduced context via sliding window"
            );
            messages.drain(..trim_index);
        }

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
        cm.reduce_context(&mut msgs, None).await.unwrap();
        assert!(msgs.is_empty());
    }

    #[tokio::test]
    async fn under_window_is_untouched() {
        let cm = SlidingWindowConversationManager::new(10);
        let mut msgs = vec![Message::user("a"), Message::user("b")];
        cm.reduce_context(&mut msgs, None).await.unwrap();
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
        cm.reduce_context(&mut msgs, None).await.unwrap();

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
        cm.reduce_context(&mut msgs, None).await.unwrap();

        assert!(msgs.len() < before, "expected a reduction");
        assert!(msgs[0].has_tool_use());
        assert!(msgs[1].has_tool_result());
    }

    #[tokio::test]
    async fn untrimmable_history_is_left_intact_rather_than_corrupted() {
        let cm = SlidingWindowConversationManager::new(1);
        let mut msgs = vec![assistant_tool_use("1"), user_tool_result("1")];
        cm.reduce_context(&mut msgs, None).await.unwrap();
        // Better to exceed the window than to emit an invalid history.
        assert_eq!(msgs.len(), 2);
    }
}
