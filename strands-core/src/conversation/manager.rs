use async_trait::async_trait;

use crate::error::StrandsError;
use crate::types::message::Message;

/// Strategy for managing conversation context window limits.
///
/// Called before each model invocation to ensure the message history
/// fits within the model's context window.
#[async_trait]
pub trait ConversationManager: Send + Sync {
    /// Reduce context if the message list exceeds limits.
    /// Mutates messages in place (removes/summarizes older messages).
    async fn reduce_context(
        &self,
        messages: &mut Vec<Message>,
        system_prompt: Option<&str>,
    ) -> Result<(), StrandsError>;
}
