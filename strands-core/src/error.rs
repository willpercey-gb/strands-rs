use thiserror::Error;

#[derive(Error, Debug)]
pub enum StrandsError {
    #[error("Model error: {0}")]
    Model(String),

    #[error("Tool error: {tool_name}: {message}")]
    Tool { tool_name: String, message: String },

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Max cycles reached ({0})")]
    MaxCycles(usize),

    #[error("Max tokens reached")]
    MaxTokens,

    #[error("Cancelled")]
    Cancelled,

    /// Non-retryable model failure: quota exhausted, rate-limit hit,
    /// authentication failed, etc. The agent event loop will surface
    /// this immediately rather than burning further retries that are
    /// guaranteed to fail the same way.
    #[error("Provider quota / auth: {0}")]
    Quota(String),

    #[error("Conversation management error: {0}")]
    ConversationManagement(String),

    #[error("Session error: {0}")]
    Session(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, StrandsError>;

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
