pub mod defaults;
/// Choosing a model per call, and failing over.
pub mod routing;
/// Character-based token estimation.
pub mod tokens;

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::StrandsError;
use crate::types::{
    content::SystemPrompt, message::Message, streaming::StreamEvent, tools::ToolSpec,
};

pub use defaults::{get_context_window_limit, DEFAULT_CONTEXT_WINDOW_LIMIT};
pub use routing::{
    FallbackStrategy, ModelRouter, PredicateStrategy, RoutingStrategy, StaticStrategy,
};

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
    ///
    /// `system_prompt` may be plain text or structured blocks carrying cache
    /// points. Adapters without cache-point support should call
    /// [`SystemPrompt::as_text`]; those with it should use
    /// [`SystemPrompt::split`].
    async fn stream(
        &self,
        messages: &[Message],
        system_prompt: Option<&SystemPrompt>,
        tool_specs: &[ToolSpec],
    ) -> Result<ModelStream, StrandsError>;

    /// The model id, when the adapter knows one.
    ///
    /// Used to resolve a context window limit from the built-in table.
    fn model_id(&self) -> Option<&str> {
        None
    }

    /// The model's context window, in tokens.
    ///
    /// Resolved from [`model_id`](Model::model_id) against the built-in table
    /// by default. Override to report a limit the table cannot know — a local
    /// deployment, or a configured override.
    ///
    /// Returning `None` is meaningful: callers disable proactive context
    /// management rather than compressing against a guessed limit.
    fn context_window_limit(&self) -> Option<u64> {
        self.model_id().and_then(get_context_window_limit)
    }

    /// Estimate the input tokens a call would consume.
    ///
    /// The default is a character heuristic (see [`tokens`]) — good enough to
    /// decide *when* to compress, not accurate enough for billing. Override
    /// where the provider exposes a native counting API.
    async fn count_tokens(
        &self,
        messages: &[Message],
        system_prompt: Option<&SystemPrompt>,
        tool_specs: &[ToolSpec],
    ) -> Result<u64, StrandsError> {
        Ok(tokens::estimate_tokens(messages, tool_specs, system_prompt))
    }

    /// Fraction of the context window a given input token count consumes.
    ///
    /// Returns `None` when no limit is known, so callers can distinguish
    /// "not close to full" from "we have no idea" — the two warrant different
    /// behaviour, and conflating them is how proactive compression ends up
    /// firing against an invented limit.
    fn estimate_utilization(&self, input_tokens: u64) -> Option<f64> {
        let limit = self.context_window_limit()?;
        if limit == 0 {
            return None;
        }
        Some(input_tokens as f64 / limit as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    struct KnownModel;

    #[async_trait]
    impl Model for KnownModel {
        async fn stream(
            &self,
            _messages: &[Message],
            _system_prompt: Option<&SystemPrompt>,
            _tool_specs: &[ToolSpec],
        ) -> Result<ModelStream, StrandsError> {
            Ok(Box::pin(stream::iter(vec![])))
        }

        fn model_id(&self) -> Option<&str> {
            Some("claude-opus-5")
        }
    }

    struct UnknownModel;

    #[async_trait]
    impl Model for UnknownModel {
        async fn stream(
            &self,
            _messages: &[Message],
            _system_prompt: Option<&SystemPrompt>,
            _tool_specs: &[ToolSpec],
        ) -> Result<ModelStream, StrandsError> {
            Ok(Box::pin(stream::iter(vec![])))
        }
    }

    #[test]
    fn limit_resolves_from_the_model_id() {
        assert_eq!(KnownModel.context_window_limit(), Some(1_000_000));
    }

    #[test]
    fn utilization_is_none_without_a_known_limit() {
        // Distinguishing "unknown" from "low" is the point — a default here
        // would let compression fire against a limit nobody verified.
        assert_eq!(UnknownModel.estimate_utilization(100), None);
    }

    #[test]
    fn utilization_is_a_ratio_of_the_window() {
        let u = KnownModel.estimate_utilization(500_000).unwrap();
        assert!((u - 0.5).abs() < f64::EPSILON, "expected 0.5, got {u}");

        // Above 1.0 signals overflow rather than saturating.
        let over = KnownModel.estimate_utilization(1_500_000).unwrap();
        assert!(over > 1.0);
    }

    #[tokio::test]
    async fn default_count_tokens_uses_the_heuristic() {
        let msgs = vec![Message::user("hello world")];
        let n = KnownModel.count_tokens(&msgs, None, &[]).await.unwrap();
        assert!(n > 0);
    }
}
