use std::process::Stdio;

use async_stream::try_stream;
use async_trait::async_trait;
use strands_core::model::{Model, ModelStream};
use strands_core::types::message::{Message, Role};
use strands_core::types::streaming::StreamEvent;
use strands_core::types::tools::ToolSpec;
use strands_core::StrandsError;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

use crate::stream::ClaudeCliState;

/// One-shot wrapper around the local `claude -p` CLI. Each call to
/// `stream` spawns a fresh subprocess, pipes the prompt in via stdin,
/// reads `--output-format stream-json` lines from stdout, and produces
/// the equivalent strands `StreamEvent` flow.
pub struct ClaudeCliModel {
    /// Path to the executable. Defaults to `"claude"` (resolved via PATH).
    pub command: String,
    /// Model alias passed to `--model`. Defaults to `"sonnet"`.
    pub model: String,
    /// Optional system prompt forwarded as `--append-system-prompt`.
    pub system_prompt: Option<String>,
    /// Working directory the CLI runs in (where it'll resolve CLAUDE.md
    /// auto-context, plugins, etc.). Defaults to the parent process's cwd.
    pub cwd: Option<std::path::PathBuf>,
    /// When true, pass `--bare` to skip hooks/LSP/plugin sync. Note that
    /// `--bare` requires `ANTHROPIC_API_KEY` since OAuth/keychain are
    /// disabled — leave at `false` for OAuth-logged-in users.
    pub bare: bool,
}

impl ClaudeCliModel {
    /// Create a wrapper for the named model alias. Defaults: command =
    /// `"claude"`, no system prompt, cwd inherited, `bare = false`.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            command: "claude".into(),
            model: model.into(),
            system_prompt: None,
            cwd: None,
            bare: false,
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

    pub fn with_bare(mut self, bare: bool) -> Self {
        self.bare = bare;
        self
    }
}

#[async_trait]
impl Model for ClaudeCliModel {
    async fn stream(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
        _tool_specs: &[ToolSpec],
    ) -> Result<ModelStream, StrandsError> {
        // Concatenate the conversation into a single prompt — the CLI
        // is one-shot and stateless from our perspective.
        let prompt = render_prompt(messages);
        let model = self.model.clone();
        let command = self.command.clone();
        let cwd = self.cwd.clone();
        let bare = self.bare;
        let system = system_prompt
            .map(|s| s.to_string())
            .or_else(|| self.system_prompt.clone());

        let stream = try_stream! {
            let mut cmd = Command::new(&command);
            cmd.arg("-p");
            cmd.arg("--output-format").arg("stream-json");
            cmd.arg("--include-partial-messages");
            cmd.arg("--input-format").arg("text");
            cmd.arg("--verbose");
            cmd.arg("--model").arg(&model);
            if bare {
                cmd.arg("--bare");
            }
            if let Some(sys) = &system {
                cmd.arg("--append-system-prompt").arg(sys);
            }
            if let Some(dir) = &cwd {
                cmd.current_dir(dir);
            }
            cmd.stdin(Stdio::piped());
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::piped());

            let mut child = cmd.spawn().map_err(|e| {
                StrandsError::Other(format!(
                    "failed to spawn `{command}`: {e} (is the Claude CLI installed and on PATH?)"
                ))
            })?;
            // Pipe the prompt in via stdin and close it so the CLI stops reading.
            if let Some(mut stdin) = child.stdin.take() {
                stdin
                    .write_all(prompt.as_bytes())
                    .await
                    .map_err(|e| StrandsError::Other(format!("write stdin: {e}")))?;
            }

            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| StrandsError::Other("claude stdout missing".into()))?;
            let mut reader = BufReader::new(stdout);
            let mut state = ClaudeCliState::new();
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
                Err(StrandsError::Other(format!(
                    "claude exited with status {status}: {err_buf}"
                )))?;
            }
        };

        Ok(Box::pin(stream))
    }
}

/// Render the conversation as a plain text prompt for the CLI.
/// `User: ...\nAssistant: ...\nUser: <last>\n`.
fn render_prompt(messages: &[Message]) -> String {
    let mut out = String::new();
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
    fn renders_multi_turn_prompt() {
        let msgs = vec![
            Message::user("first question"),
            Message::assistant(vec![ContentBlock::Text {
                text: "first answer".into(),
            }]),
            Message::user("follow-up"),
        ];
        let p = render_prompt(&msgs);
        assert!(p.contains("User: first question"));
        assert!(p.contains("Assistant: first answer"));
        assert!(p.contains("User: follow-up"));
    }
}
