//! The data the agent and its models exchange.
//!
//! [`message::Message`] and [`content::ContentBlock`] model a conversation;
//! [`streaming::StreamEvent`] is the protocol adapters emit; [`tools::ToolSpec`]
//! describes a tool to the model. Shapes follow the Bedrock API that upstream
//! models, so adapters can map to provider payloads without re-deriving them.

/// Where a claim came from.
pub mod citations;
/// Message content blocks.
pub mod content;
/// Images, documents, audio and video.
pub mod media;
/// Conversation messages.
pub mod message;
/// The event protocol adapters emit.
pub mod streaming;
/// Tool specifications.
pub mod tools;
