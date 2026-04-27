use std::process::Stdio;

use async_stream::try_stream;
use async_trait::async_trait;
use strands_core::model::{Model, ModelStream};
use strands_core::types::message::{Message, Role};
use strands_core::types::tools::ToolSpec;
use strands_core::StrandsError;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

use crate::stream::GeminiCliState;

/// Approval mode passed to `gemini --approval-mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalMode {
    Default,
    AutoEdit,
    Yolo,
    Plan,
}

impl ApprovalMode {
    fn as_arg(self) -> &'static str {
        match self {
            ApprovalMode::Default => "default",
            ApprovalMode::AutoEdit => "auto_edit",
            ApprovalMode::Yolo => "yolo",
            ApprovalMode::Plan => "plan",
        }
    }
}

/// One-shot wrapper around the local `gemini` CLI in non-interactive
/// mode. Each call to `stream` spawns a fresh subprocess, pipes the
/// prompt in via stdin (with `-p ""` to force headless mode), reads
/// `--output-format stream-json` lines from stdout, and produces the
/// equivalent strands `StreamEvent` flow.
pub struct GeminiCliModel {
    /// Path to the executable. Defaults to `"gemini"` (resolved via PATH).
    pub command: String,
    /// Model alias passed to `--model` / `-m`. Defaults to
    /// `"gemini-2.5-flash"`.
    pub model: String,
    /// Optional system prompt prepended to the rendered conversation.
    /// Gemini CLI has no `--system` flag, so this is folded into the
    /// prompt.
    pub system_prompt: Option<String>,
    /// Working directory the CLI runs in. Defaults to the parent
    /// process's cwd.
    pub cwd: Option<std::path::PathBuf>,
    /// Approval mode for tool calls. When `None` Gemini uses its
    /// default (interactive). For headless use set to `Yolo` so tool
    /// calls don't block waiting for stdin approval.
    pub approval_mode: Option<ApprovalMode>,
    /// Pass `--sandbox` to wrap tool execution in the platform sandbox.
    pub sandbox: bool,
    /// Pass `--debug` so the CLI emits internal events to stderr —
    /// useful when diagnosing wrapper bugs, noisy in normal use.
    pub debug: bool,
}

impl GeminiCliModel {
    /// Create a wrapper for the named model alias. Defaults: command =
    /// `"gemini"`, no system prompt, cwd inherited, `approval_mode = None`.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            command: "gemini".into(),
            model: model.into(),
            system_prompt: None,
            cwd: None,
            approval_mode: None,
            sandbox: false,
            debug: false,
        }
    }

    pub fn with_command(mut self, cmd: impl Into<String>) -> Self {
        self.command = cmd.into();
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn with_cwd(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.cwd = Some(dir.into());
        self
    }

    pub fn with_approval_mode(mut self, mode: ApprovalMode) -> Self {
        self.approval_mode = Some(mode);
        self
    }

    /// Convenience: shorthand for `with_approval_mode(ApprovalMode::Yolo)`.
    /// The CLI's `-y` flag bypasses every tool-call confirmation —
    /// only safe for unattended runs.
    pub fn with_yolo(mut self, on: bool) -> Self {
        self.approval_mode = if on {
            Some(ApprovalMode::Yolo)
        } else {
            None
        };
        self
    }

    pub fn with_sandbox(mut self, on: bool) -> Self {
        self.sandbox = on;
        self
    }

    pub fn with_debug(mut self, on: bool) -> Self {
        self.debug = on;
        self
    }
}

#[async_trait]
impl Model for GeminiCliModel {
    async fn stream(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
        _tool_specs: &[ToolSpec],
    ) -> Result<ModelStream, StrandsError> {
        let system = system_prompt
            .map(|s| s.to_string())
            .or_else(|| self.system_prompt.clone());
        let prompt = render_prompt(messages, system.as_deref());
        let model = self.model.clone();
        let command = self.command.clone();
        let cwd = self.cwd.clone();
        let approval_mode = self.approval_mode;
        let sandbox = self.sandbox;
        let debug = self.debug;

        let stream = try_stream! {
            let mut cmd = Command::new(&command);
            // `-p ""` puts gemini in headless mode and the rendered
            // conversation is appended via stdin (gemini's docs say:
            // "-p Run in non-interactive (headless) mode with the given
            // prompt. Appended to input on stdin (if any).").
            cmd.arg("-p").arg("");
            cmd.arg("--output-format").arg("stream-json");
            cmd.arg("--model").arg(&model);
            if let Some(mode) = approval_mode {
                cmd.arg("--approval-mode").arg(mode.as_arg());
            }
            if sandbox {
                cmd.arg("--sandbox");
            }
            if debug {
                cmd.arg("--debug");
            }
            if let Some(dir) = &cwd {
                cmd.current_dir(dir);
            }
            cmd.stdin(Stdio::piped());
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::piped());

            let mut child = cmd.spawn().map_err(|e| {
                StrandsError::Other(format!(
                    "failed to spawn `{command}`: {e} (is the Gemini CLI installed and on PATH?)"
                ))
            })?;
            if let Some(mut stdin) = child.stdin.take() {
                stdin
                    .write_all(prompt.as_bytes())
                    .await
                    .map_err(|e| StrandsError::Other(format!("write stdin: {e}")))?;
            }

            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| StrandsError::Other("gemini stdout missing".into()))?;
            let mut reader = BufReader::new(stdout);
            let mut state = GeminiCliState::new();
            let mut line = String::new();
            loop {
                line.clear();
                let n = reader
                    .read_line(&mut line)
                    .await
                    .map_err(|e| StrandsError::Other(format!("read stdout: {e}")))?;
                if n == 0 {
                    break;
                }
                for evt in state.ingest_line(&line) {
                    yield evt;
                }
            }
            for evt in state.flush() {
                yield evt;
            }

            let status = child
                .wait()
                .await
                .map_err(|e| StrandsError::Other(format!("wait: {e}")))?;
            if !status.success() {
                let mut err_buf = String::new();
                if let Some(mut stderr) = child.stderr.take() {
                    use tokio::io::AsyncReadExt;
                    let _ = stderr.read_to_string(&mut err_buf).await;
                }
                Err(strands_core::classify_cli_failure(format!(
                    "gemini exited with status {status}: {err_buf}"
                )))?;
            }
        };

        Ok(Box::pin(stream))
    }
}

/// Render the conversation as a plain text prompt. Gemini exec is
/// stateless from our side; the rendered transcript is fed via stdin.
fn render_prompt(messages: &[Message], system: Option<&str>) -> String {
    let mut out = String::new();
    if let Some(s) = system {
        out.push_str("System: ");
        out.push_str(s);
        out.push('\n');
    }
    for m in messages {
        match m.role {
            Role::User => out.push_str("User: "),
            Role::Assistant => out.push_str("Assistant: "),
            Role::System => out.push_str("System: "),
        }
        out.push_str(&m.text());
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use strands_core::ContentBlock;

    #[test]
    fn renders_multi_turn_prompt_with_system() {
        let msgs = vec![
            Message::user("first"),
            Message::assistant(vec![ContentBlock::Text { text: "ack".into() }]),
            Message::user("second"),
        ];
        let p = render_prompt(&msgs, Some("be terse"));
        assert!(p.starts_with("System: be terse\n"));
        assert!(p.contains("User: first"));
        assert!(p.contains("Assistant: ack"));
        assert!(p.contains("User: second"));
    }

    #[test]
    fn yolo_helper_sets_approval_mode() {
        let m = GeminiCliModel::new("gemini-2.5-flash").with_yolo(true);
        assert_eq!(m.approval_mode, Some(ApprovalMode::Yolo));
    }
}
