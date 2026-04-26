//! OpenRouter model adapter for strands-core. Speaks the OpenAI-compatible
//! `/chat/completions` SSE endpoint.

pub mod client;
pub(crate) mod stream;
pub(crate) mod types;

pub use client::OpenRouterModel;
