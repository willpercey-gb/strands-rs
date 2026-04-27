//! OpenAI Codex CLI (`codex exec --json`) one-shot model adapter for strands-core.
//!
//! Spawns the local `codex` binary in non-interactive exec mode with
//! `--json`, reads its stdout line by line, and translates each
//! `ThreadEvent` into the strands `StreamEvent` flow.
//!
//! Authentication uses whatever the local CLI is logged in with — the
//! adapter does not pass credentials of its own.

pub mod client;
pub(crate) mod stream;

pub use client::{CodexCliModel, SandboxPolicy};
