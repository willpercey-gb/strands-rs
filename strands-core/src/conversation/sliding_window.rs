use async_trait::async_trait;

use crate::error::StrandsError;
use crate::types::message::Message;

use super::ConversationManager;

/// Keeps a fixed window of recent messages, dropping the oldest
/// when the limit is exceeded.
pub struct SlidingWindowConversationManager {
    /// Maximum number of messages to retain.
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
        _system_prompt: Option<&str>,
    ) -> Result<(), StrandsError> {
        if messages.len() > self.window_size {
            let drain_count = messages.len() - self.window_size;
            messages.drain(..drain_count);
        }
        Ok(())
    }
}
