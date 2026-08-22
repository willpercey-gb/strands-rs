pub mod manager;
pub mod null;
pub mod pin;
pub mod sliding_window;
pub mod summarizing;
pub mod trim;

pub use manager::{
    ConversationManager, ProactiveCompression, ReduceContext, DEFAULT_COMPRESSION_THRESHOLD,
};
pub use null::NullConversationManager;
pub use pin::{apply_pin_first, is_pinned, pin_message, unpin_message};
pub use sliding_window::SlidingWindowConversationManager;
pub use summarizing::SummarizingConversationManager;
