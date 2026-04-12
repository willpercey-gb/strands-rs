use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::StrandsError;
use crate::types::{message::Message, streaming::StreamEvent, tools::ToolSpec};

/// A boxed async stream of model events.
pub type ModelStream = BoxStream<'static, Result<StreamEvent, StrandsError>>;

/// The core model provider trait. Implement this for each LLM backend.
///
/// Model adapters normalize provider-specific APIs into the unified
/// `StreamEvent` protocol that the agent loop consumes.
#[async_trait]
pub trait Model: Send + Sync {
    /// Stream a response from the model given conversation history and available tools.
    ///
    /// The implementation should:
    /// 1. Convert `messages` and `tool_specs` to the provider's format
    /// 2. Make the API call with streaming enabled
    /// 3. Return a stream of `StreamEvent` variants
    async fn stream(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
        tool_specs: &[ToolSpec],
    ) -> Result<ModelStream, StrandsError>;
}
