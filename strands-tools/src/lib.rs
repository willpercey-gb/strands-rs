//! Ready-made tools for strands-rs agents.
//!
//! These live outside `strands-core` so the core crate stays dependency-light:
//! an agent that only needs the loop should not pull in an HTTP stack.
//!
//! Each tool is behind a feature; `shell`, `file-editor`, `sleep` and `stop`
//! are on by default, `http` is opt-in.
//!
//! Ported from upstream `vended_tools/`.

// Public API must be documented: this crate is published, so anything
// undocumented here shows up as a gap on docs.rs.
#![warn(missing_docs)]

#[cfg(feature = "file-editor")]
pub mod file_editor;
#[cfg(feature = "http")]
pub mod http_request;
#[cfg(feature = "sandbox")]
pub mod sandbox;
#[cfg(feature = "shell")]
pub mod shell;
#[cfg(feature = "sleep")]
pub mod sleep;
#[cfg(feature = "stop")]
pub mod stop;

#[cfg(feature = "file-editor")]
pub use file_editor::FileEditorTool;
#[cfg(feature = "http")]
pub use http_request::HttpRequestTool;
#[cfg(feature = "sandbox")]
pub use sandbox::{Command, CommandOutput, DockerSandbox, LocalEnvironment, Sandbox};
#[cfg(feature = "shell")]
pub use shell::ShellTool;
#[cfg(feature = "sleep")]
pub use sleep::SleepTool;
#[cfg(feature = "stop")]
pub use stop::StopTool;
