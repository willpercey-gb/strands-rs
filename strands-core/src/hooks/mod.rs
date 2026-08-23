//! Lifecycle callbacks.
//!
//! A [`Hook`] observes the agent loop and can steer it — cancelling a tool
//! call, retrying a model call, overriding messages, or pausing for human
//! input. Hooks receive `&mut HookEvent` and write to the event's mutable
//! fields; see [`events`] for what each event exposes.

/// The lifecycle events hooks receive.
pub mod events;
/// Registration, ordering and dispatch.
pub mod registry;

pub use events::HookEvent;
pub use registry::{Hook, HookRegistry};
