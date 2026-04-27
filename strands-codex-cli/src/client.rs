use std::process::Stdio;

use async_stream::try_stream;
use async_trait::async_trait;
use strands_core::model::{Model, ModelStream};
use strands_core::types::message::{Message, Role};
use strands_core::types::tools::ToolSpec;
use strands_core::StrandsError;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

use crate::stream::CodexCliState;

/// Sandbox policy passed to `codex exec --sandbox`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxPolicy {
    ReadOnly,
    WorkspaceWrite,
    DangerFullAccess,
}

impl SandboxPolicy {
    fn as_arg(self) -> &'static str {
        match self {
            SandboxPolicy::ReadOnly => "read-only",
            SandboxPolicy::WorkspaceWrite => "workspace-write",
            SandboxPolicy::DangerFullAccess => "danger-full-access",
        }
    }
}

/// One-shot wrapper around the local `codex exec` CLI. Each call to
/// `stream` spawns a fresh subprocess, pipes the prompt in via stdin
/// (`-` positional), reads `--json` events from stdout, and produces
/// the equivalent strands `StreamEvent` flow.
pub struct CodexCliModel {
    /// Path to the executable. Defaults to `"codex"` (resolved via PATH).
    pub command: String,
    /// Model alias passed to `--model`. Optional — when `None` codex
    /// uses whatever is configured in `$CODEX_HOME/config.toml`.
    pub model: Option<String>,
    /// Optional system prompt prepended to the rendered conversation.
    /// Codex has no `--system` flag, so this is folded into the prompt.
    pub system_prompt: Option<String>,
    /// Working directory the CLI runs in. Defaults to the parent
    /// process's cwd.
    pub cwd: Option<std::path::PathBuf>,
    /// Sandbox policy passed via `--sandbox`. When `None` codex uses
    /// its default.
    pub sandbox: Option<SandboxPolicy>,
    /// Pass `--full-auto` (on-request approvals + workspace-write
    /// sandbox). Mutually exclusive with `dangerously_bypass`.
    pub full_auto: bool,
    /// Pass `--dangerously-bypass-approvals-and-sandbox`. Skips all
    /// approvals and sandbox entirely — only safe in already-isolated
    /// environments.
    pub dangerously_bypass: bool,
    /// Pass `--skip-git-repo-check` so codex runs outside a git repo.
    pub skip_git_repo_check: bool,
    /// Pass `--ephemeral` so the session isn't persisted to disk.
    pub ephemeral: bool,
}

impl CodexCliModel {
    /// Create a wrapper with no model override. Defaults: command =
    /// `"codex"`, no system prompt, cwd inherited, no sandbox/approval
    /// preset, `skip_git_repo_check = false`, `ephemeral = false`.
    pub fn new() -> Self {
        Self {
            command: "codex".into(),
            model: None,
            system_prompt: None,
            cwd: None,
            sandbox: None,
            full_auto: false,
            dangerously_bypass: false,
            skip_git_repo_check: false,
            ephemeral: false,
        }
    }

    /// Convenience: create a wrapper that targets a specific model alias.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
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

    pub fn with_sandbox(mut self, policy: SandboxPolicy) -> Self {
        self.sandbox = Some(policy);
        self
    }

    pub fn with_full_auto(mut self, on: bool) -> Self {
        self.full_auto = on;
        self
    }

    /// Pass `--dangerously-bypass-approvals-and-sandbox` to the spawned
    /// subprocess. Skips all approvals — only use in sandboxes with no
    /// internet access. Setting this disables `full_auto` (mutually
    /// exclusive on the codex side).
    pub fn with_dangerously_bypass(mut self, on: bool) -> Self {
        self.dangerously_bypass = on;
        if on {
            self.full_auto = false;
        }
        self
    }

    pub fn with_skip_git_repo_check(mut self, on: bool) -> Self {
        self.skip_git_repo_check = on;
        self
    }

    pub fn with_ephemeral(mut self, on: bool) -> Self {
        self.ephemeral = on;
        self
    }
}

impl Default for CodexCliModel {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Model for CodexCliModel {
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
        let sandbox = self.sandbox;
        let full_auto = self.full_auto;
        let dangerously_bypass = self.dangerously_bypass;
        let skip_git_repo_check = self.skip_git_repo_check;
        let ephemeral = self.ephemeral;

        let stream = try_stream! {
            let mut cmd = Command::new(&command);
            cmd.arg("exec");
            cmd.arg("--json");
            // `-` makes codex read the prompt from stdin instead of an arg.
            cmd.arg("-");
            if let Some(m) = &model {
                cmd.arg("--model").arg(m);
            }
            if let Some(dir) = &cwd {
                cmd.arg("--cd").arg(dir);
            }
            if let Some(s) = sandbox {
                cmd.arg("--sandbox").arg(s.as_arg());
            }
            if dangerously_bypass {
                cmd.arg("--dangerously-bypass-approvals-and-sandbox");
            } else if full_auto {
                cmd.arg("--full-auto");
            }
            if skip_git_repo_check {
                cmd.arg("--skip-git-repo-check");
            }
            if ephemeral {
                cmd.arg("--ephemeral");
            }
            cmd.stdin(Stdio::piped());
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::piped());

            let mut child = cmd.spawn().map_err(|e| {
                StrandsError::Other(format!(
                    "failed to spawn `{command}`: {e} (is the Codex CLI installed and on PATH?)"
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
                .ok_or_else(|| StrandsError::Other("codex stdout missing".into()))?;
            let mut reader = BufReader::new(stdout);
            let mut state = CodexCliState::new();
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
                    "codex exited with status {status}: {err_buf}"
                )))?;
            }
        };

        Ok(Box::pin(stream))
    }
}

/// Render the conversation as a plain text prompt for the CLI. Codex
/// `exec` is one-shot and stateless from our side, so we concatenate
/// all turns the same way the claude-cli adapter does.
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
    fn dangerously_bypass_disables_full_auto() {
        let m = CodexCliModel::new()
            .with_full_auto(true)
            .with_dangerously_bypass(true);
        assert!(m.dangerously_bypass);
        assert!(!m.full_auto);
    }
}
