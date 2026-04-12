use crate::types::streaming::StreamEvent;

/// Callback handler for real-time stream events.
///
/// Set on the agent to receive streaming events as they arrive from the model,
/// before they're accumulated into content blocks. Useful for updating UIs
/// in real-time as text or tool calls stream in.
pub trait CallbackHandler: Send + Sync {
    fn on_stream_event(&self, event: &StreamEvent);
}

/// Blanket impl so closures can be used as callback handlers.
impl<F: Fn(&StreamEvent) + Send + Sync> CallbackHandler for F {
    fn on_stream_event(&self, event: &StreamEvent) {
        self(event);
    }
}
