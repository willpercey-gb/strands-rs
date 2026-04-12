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
