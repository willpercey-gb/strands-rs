//! Claude Code CLI (`claude -p`) one-shot model adapter for strands-core.
//!
//! Spawns the local `claude` binary in non-interactive print mode with
//! `--output-format stream-json --include-partial-messages`, reads its
//! stdout line by line, and translates each event into the strands
//! `StreamEvent` flow.
//!
//! Authentication uses whatever the local CLI is logged in with —
//! Anthropic OAuth, ANTHROPIC_API_KEY, or whatever the user configured.
//! This adapter does not pass credentials of its own.

pub mod client;
pub(crate) mod stream;

pub use client::ClaudeCliModel;
