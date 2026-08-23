use thiserror::Error;

#[derive(Error, Debug)]
/// Everything that can go wrong in an agent run.
pub enum StrandsError {
    #[error("Model error: {0}")]
    /// The provider rejected or failed the request.
    Model(String),

    #[error("Tool error: {tool_name}: {message}")]
    /// A tool failed. Usually surfaced to the model as a tool result rather
    /// than raised, so it can try something else.
    Tool {
        /// Which tool failed.
        tool_name: String,
        /// What went wrong.
        message: String,
    },

    #[error("Tool not found: {0}")]
    /// The model asked for a tool that is not registered.
    ToolNotFound(String),

    #[error("Max cycles reached ({0})")]
    /// The hard cycle backstop was hit. Prefer [`Limits`](crate::agent::Limits),
    /// which stops cleanly instead of erroring.
    MaxCycles(usize),

    #[error("Max tokens reached")]
    /// The provider truncated the response at its output cap.
    MaxTokens,

    #[error("Cancelled")]
    /// The caller cancelled the run.
    Cancelled,

    /// Non-retryable model failure: quota exhausted, rate-limit hit,
    /// authentication failed, etc. The agent event loop will surface
    /// this immediately rather than burning further retries that are
    /// guaranteed to fail the same way.
    #[error("Provider quota / auth: {0}")]
    Quota(String),

    /// The request exceeded the model's context window.
    ///
    /// Distinct from a generic model error because it is *recoverable*: the
    /// conversation manager can reduce the history and the same call will then
    /// succeed. Collapsing it into `Model` would make that retry impossible to
    /// trigger.
    #[error("Context window overflow: {0}")]
    ContextWindowOverflow(String),

    #[error("Conversation management error: {0}")]
    /// The conversation manager could not reduce the history.
    ConversationManagement(String),

    #[error("Session error: {0}")]
    /// Session persistence failed.
    Session(String),

    #[error("Serialization error: {0}")]
    /// JSON encoding or decoding failed.
    Serialization(#[from] serde_json::Error),

    #[error("{0}")]
    /// Anything not covered above.
    Other(String),
}

/// A result whose error is a [`StrandsError`].
pub type Result<T> = std::result::Result<T, StrandsError>;

impl StrandsError {
    /// Whether reducing the conversation could make this call succeed.
    pub fn is_context_overflow(&self) -> bool {
        matches!(self, StrandsError::ContextWindowOverflow(_))
    }

    /// Whether retrying this call unchanged could ever succeed.
    ///
    /// Quota, auth and cancellation are all permanent for this request;
    /// burning the retry budget on them costs money and time for nothing.
    pub fn is_retryable(&self) -> bool {
        !matches!(
            self,
            StrandsError::Quota(_)
                | StrandsError::Cancelled
                | StrandsError::ContextWindowOverflow(_)
        )
    }
}

/// Substrings providers use to report a context-window overflow.
///
/// Provider-agnostic: Anthropic, OpenAI, Google, Ollama, llama.cpp and Mistral
/// all phrase this differently, and none of them use a distinct status code.
const OVERFLOW_NEEDLES: &[&str] = &[
    "context window",
    "context length",
    "too many tokens",
    "maximum context",
    "prompt is too long",
    "input length and `max_tokens` exceed",
    "reduce the length of the messages",
    "exceeds the maximum",
    "context_length_exceeded",
    "requested tokens exceed",
];

/// Classify a provider error message, detecting context-window overflow.
///
/// Overflow is checked before the quota needles: "reduce the length" is
/// recoverable by trimming, whereas treating it as a quota failure would
/// short-circuit the retry that would have fixed it.
pub fn classify_provider_failure(message: impl Into<String>) -> StrandsError {
    let msg = message.into();
    let lc = msg.to_lowercase();

    if OVERFLOW_NEEDLES.iter().any(|n| lc.contains(n)) {
        return StrandsError::ContextWindowOverflow(msg);
    }

    classify_cli_failure(msg)
}

/// Heuristic: scan a CLI's stderr / output for tell-tale quota / auth
/// substrings. If matched, classify the failure as
/// [`StrandsError::Quota`] so the agent retry loop short-circuits
/// instead of burning more credits on a request that will fail the
/// same way.
///
/// Provider-agnostic — looks for substrings common across Anthropic,
/// OpenAI, Google, OpenRouter (case-insensitive).
pub fn classify_cli_failure(message: impl Into<String>) -> StrandsError {
    let msg = message.into();
    let lc = msg.to_lowercase();
    const QUOTA_NEEDLES: &[&str] = &[
        "exhausted your capacity",
        "exhausted your quota",
        "quota exceeded",
        "rate limit",
        "rate-limit",
        "too many requests",
        "429",
        "insufficient_quota",
        "billing",
        "credit balance",
        "max attempts reached",
        "not logged in",
        "please run /login",
        "authentication failed",
        "401 unauthorized",
        "403 forbidden",
        "permission denied",
        "invalid api key",
    ];
    if QUOTA_NEEDLES.iter().any(|n| lc.contains(n)) {
        StrandsError::Quota(msg)
    } else {
        StrandsError::Other(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_phrasings_are_detected_across_providers() {
        for message in [
            "This model's maximum context length is 8192 tokens",
            "prompt is too long: 250000 tokens > 200000",
            "input length and `max_tokens` exceed context limit",
            "Please reduce the length of the messages",
            "error code: context_length_exceeded",
        ] {
            assert!(
                classify_provider_failure(message).is_context_overflow(),
                "expected overflow for: {message}"
            );
        }
    }

    #[test]
    fn overflow_wins_over_quota_when_both_could_match() {
        // "reduce the length" is recoverable by trimming; classifying it as a
        // quota failure would short-circuit the retry that would have fixed it.
        let error =
            classify_provider_failure("rate limit note: please reduce the length of the messages");
        assert!(error.is_context_overflow(), "got {error:?}");
    }

    #[test]
    fn quota_phrasings_still_classify_as_quota() {
        for message in [
            "429 Too Many Requests",
            "You have exhausted your quota",
            "invalid api key",
        ] {
            assert!(
                matches!(classify_provider_failure(message), StrandsError::Quota(_)),
                "expected quota for: {message}"
            );
        }
    }

    #[test]
    fn unrecognised_failures_stay_generic() {
        assert!(matches!(
            classify_provider_failure("connection reset by peer"),
            StrandsError::Other(_)
        ));
    }

    #[test]
    fn retryability_excludes_the_permanent_failures() {
        assert!(StrandsError::Model("transient".into()).is_retryable());
        assert!(!StrandsError::Quota("out".into()).is_retryable());
        assert!(!StrandsError::Cancelled.is_retryable());
        assert!(
            !StrandsError::ContextWindowOverflow("too big".into()).is_retryable(),
            "an unchanged retry cannot fix an oversized request"
        );
    }
}
