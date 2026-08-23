//! A tool that runs shell commands.
//!
//! # Safety
//!
//! This hands a model the ability to execute arbitrary commands on the host. It
//! is opt-in for that reason, and ships with a timeout, an output cap, and an
//! optional allowlist. None of those make it safe to expose to untrusted input —
//! the right isolation for that is a sandbox, not a filter.
//!
//! Upstream renamed this from `bash` to `shell` in v1.50.
//!
//! Ported from upstream `vended_tools/shell/`.

use std::collections::HashSet;
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};
use strands_core::error::StrandsError;
use strands_core::tool::{Tool, ToolContext, ToolOutput};
use strands_core::types::tools::{ToolAnnotations, ToolSpec};
use tokio::io::AsyncReadExt;
use tracing::{debug, warn};

/// Default wall-clock limit for one command.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// Default cap on captured output, in bytes.
///
/// A command that prints megabytes would otherwise blow the context window in
/// a single tool result.
pub const DEFAULT_MAX_OUTPUT: usize = 100_000;

/// Runs shell commands on the host.
pub struct ShellTool {
    timeout: Duration,
    max_output: usize,
    working_dir: Option<std::path::PathBuf>,
    /// When set, only these program names may be invoked.
    allowlist: Option<HashSet<String>>,
}

impl Default for ShellTool {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_TIMEOUT,
            max_output: DEFAULT_MAX_OUTPUT,
            working_dir: None,
            allowlist: None,
        }
    }
}

impl ShellTool {
    /// Create with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the max output.
    pub fn with_max_output(mut self, bytes: usize) -> Self {
        self.max_output = bytes;
        self
    }

    /// Set the working dir.
    pub fn with_working_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Restrict execution to the named programs.
    ///
    /// A coarse guard, not a security boundary: the command still runs through
    /// a shell, so anything on the allowlist that can spawn a subprocess
    /// (`find -exec`, `xargs`, an interpreter) escapes it. Use a sandbox when
    /// the input is untrusted.
    pub fn with_allowlist<I, S>(mut self, programs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowlist = Some(programs.into_iter().map(Into::into).collect());
        self
    }

    /// The first bare word of a command, used for the allowlist check.
    fn leading_program(command: &str) -> Option<&str> {
        command.split_whitespace().next()
    }

    fn check_allowed(&self, command: &str) -> Result<(), String> {
        let Some(allowlist) = &self.allowlist else {
            return Ok(());
        };

        match Self::leading_program(command) {
            Some(program) if allowlist.contains(program) => Ok(()),
            Some(program) => Err(format!("command '{program}' is not on the allowlist")),
            None => Err("empty command".to_string()),
        }
    }

    /// Trim output to the cap, saying so rather than silently truncating.
    fn cap(&self, text: String, stream: &str) -> String {
        if text.len() <= self.max_output {
            return text;
        }
        let kept: String = text.chars().take(self.max_output).collect();
        format!("{kept}\n[{stream} truncated at {} bytes]", self.max_output)
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "shell",
            "Run a shell command and return its stdout, stderr and exit code.",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run"
                    }
                },
                "required": ["command"]
            }),
        )
        .with_annotations(ToolAnnotations {
            read_only_hint: Some(false),
            // A shell command can do anything; saying otherwise would mislead
            // a permission layer reading these hints.
            destructive_hint: Some(true),
            idempotent_hint: Some(false),
            open_world_hint: Some(true),
            ..Default::default()
        })
    }

    async fn invoke(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, StrandsError> {
        let Some(command) = input.get("command").and_then(Value::as_str) else {
            return Ok(ToolOutput::error("'command' must be a string"));
        };

        if let Err(reason) = self.check_allowed(command) {
            warn!(command, reason, "Shell command refused");
            return Ok(ToolOutput::error(reason));
        }

        debug!(command, "Running shell command");

        let shell = if cfg!(windows) { "cmd" } else { "sh" };
        let flag = if cfg!(windows) { "/C" } else { "-c" };

        let mut cmd = tokio::process::Command::new(shell);
        cmd.arg(flag)
            .arg(command)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null());

        if let Some(dir) = &self.working_dir {
            cmd.current_dir(dir);
        }

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(e) => return Ok(ToolOutput::error(format!("failed to start shell: {e}"))),
        };

        let mut stdout_pipe = child.stdout.take();
        let mut stderr_pipe = child.stderr.take();

        let capture = async {
            let mut stdout = String::new();
            let mut stderr = String::new();
            if let Some(pipe) = stdout_pipe.as_mut() {
                let _ = pipe.read_to_string(&mut stdout).await;
            }
            if let Some(pipe) = stderr_pipe.as_mut() {
                let _ = pipe.read_to_string(&mut stderr).await;
            }
            let status = child.wait().await;
            (stdout, stderr, status)
        };

        let (stdout, stderr, status) = match tokio::time::timeout(self.timeout, capture).await {
            Ok(result) => result,
            Err(_) => {
                warn!(command, ?self.timeout, "Shell command timed out");
                return Ok(ToolOutput::error(format!(
                    "command timed out after {:?}",
                    self.timeout
                )));
            }
        };

        let exit_code = match status {
            Ok(status) => status.code(),
            Err(e) => return Ok(ToolOutput::error(format!("failed to wait on shell: {e}"))),
        };

        let payload = json!({
            "stdout": self.cap(stdout, "stdout"),
            "stderr": self.cap(stderr, "stderr"),
            "exit_code": exit_code,
        });

        // A non-zero exit is reported as a tool error so the model reliably
        // notices; the output is still attached for it to act on.
        if exit_code == Some(0) {
            Ok(ToolOutput::success(payload))
        } else {
            Ok(ToolOutput {
                content: payload,
                is_error: true,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn run(tool: &ShellTool, command: &str) -> ToolOutput {
        tool.invoke(json!({"command": command}), &ToolContext::default())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn captures_stdout_and_a_zero_exit() {
        let out = run(&ShellTool::new(), "echo hello").await;
        assert!(!out.is_error);
        assert!(out.content["stdout"].as_str().unwrap().contains("hello"));
        assert_eq!(out.content["exit_code"], 0);
    }

    #[tokio::test]
    async fn a_non_zero_exit_is_an_error_with_output_attached() {
        let out = run(&ShellTool::new(), "echo oops >&2; exit 3").await;
        assert!(out.is_error, "a failing command should read as an error");
        assert_eq!(out.content["exit_code"], 3);
        assert!(out.content["stderr"].as_str().unwrap().contains("oops"));
    }

    #[tokio::test]
    async fn output_is_capped_and_says_so() {
        let tool = ShellTool::new().with_max_output(50);
        let out = run(&tool, "printf 'x%.0s' $(seq 1 500)").await;
        let stdout = out.content["stdout"].as_str().unwrap();
        assert!(
            stdout.len() < 200,
            "expected truncation, got {}",
            stdout.len()
        );
        assert!(stdout.contains("truncated"), "truncation must be visible");
    }

    #[tokio::test]
    async fn a_hanging_command_times_out() {
        let tool = ShellTool::new().with_timeout(Duration::from_millis(100));
        let out = run(&tool, "sleep 5").await;
        assert!(out.is_error);
        assert!(out.content.as_str().unwrap().contains("timed out"));
    }

    #[tokio::test]
    async fn the_allowlist_refuses_other_programs() {
        let tool = ShellTool::new().with_allowlist(["echo"]);

        assert!(!run(&tool, "echo fine").await.is_error);

        let refused = run(&tool, "rm -rf /tmp/nope").await;
        assert!(refused.is_error);
        assert!(refused.content.as_str().unwrap().contains("allowlist"));
    }

    #[tokio::test]
    async fn runs_in_the_configured_directory() {
        let dir = std::env::temp_dir();
        let tool = ShellTool::new().with_working_dir(&dir);
        let out = run(&tool, "pwd").await;
        assert!(!out.content["stdout"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_missing_command_is_rejected() {
        let out = ShellTool::new()
            .invoke(json!({}), &ToolContext::default())
            .await
            .unwrap();
        assert!(out.is_error);
    }

    #[test]
    fn annotated_as_destructive_and_open_world() {
        // A permission layer reads these; understating them would be worse than
        // omitting them.
        let spec = ShellTool::new().spec();
        let annotations = spec.annotations.expect("annotations present");
        assert!(annotations.is_destructive());
        assert!(annotations.is_open_world());
        assert!(!annotations.is_read_only());
    }
}
