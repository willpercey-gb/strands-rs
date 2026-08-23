//! Keeping a conversation inside the model's context window.
//!
//! A [`ConversationManager`] is consulted before every model call, and again
//! if the provider rejects a request as too large. Reduction is never a plain
//! "drop the oldest N": see [`trim`] for why, and [`pin`] for protecting
//! messages from eviction.

/// The [`ConversationManager`] trait and its reduction context.
pub mod manager;
/// A manager that never reduces.
pub mod null;
/// Protecting messages from eviction.
pub mod pin;
/// Keep a fixed window of recent messages.
pub mod sliding_window;
/// Replace older messages with a model-written summary.
pub mod summarizing;
/// Finding boundaries that do not sever tool pairs.
pub mod trim;

pub use manager::{
    ConversationManager, ProactiveCompression, ReduceContext, DEFAULT_COMPRESSION_THRESHOLD,
};
pub use null::NullConversationManager;
pub use pin::{apply_pin_first, is_pinned, pin_message, unpin_message};
pub use sliding_window::SlidingWindowConversationManager;
pub use summarizing::SummarizingConversationManager;
