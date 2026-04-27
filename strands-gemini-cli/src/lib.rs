//! Google Gemini CLI (`gemini -p --output-format stream-json`) one-shot
//! model adapter for strands-core.
//!
//! Spawns the local `gemini` binary in non-interactive (`-p`) mode with
//! `--output-format stream-json`, reads its stdout line by line, and
//! translates each event into the strands `StreamEvent` flow.
//!
//! Authentication uses whatever the local CLI is logged in with — the
//! adapter does not pass credentials of its own.

pub mod client;
pub(crate) mod stream;

pub use client::{ApprovalMode, GeminiCliModel};
