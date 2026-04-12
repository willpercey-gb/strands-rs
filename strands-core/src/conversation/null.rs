use async_trait::async_trait;

use crate::error::StrandsError;
use crate::types::message::Message;

use super::ConversationManager;

/// No-op conversation manager. Useful for short sessions where
/// context overflow is not a concern.
pub struct NullConversationManager;

#[async_trait]
impl ConversationManager for NullConversationManager {
    async fn reduce_context(
        &self,
        _messages: &mut Vec<Message>,
        _system_prompt: Option<&str>,
    ) -> Result<(), StrandsError> {
        Ok(())
    }
}
