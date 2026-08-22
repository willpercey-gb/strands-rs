//! Sandboxes — running commands somewhere other than the host process.
//!
//! The [`ShellTool`](crate::ShellTool) runs on the host. That is fine when the
//! agent's inputs are trusted and unacceptable when they are not: an allowlist
//! does not contain a shell, and a timeout does not undo a deleted file. A
//! sandbox moves execution somewhere a mistake is survivable.
//!
//! Ported from upstream `sandbox/`.

use std::collections::HashMap;
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use strands_core::error::StrandsError;
use tokio::io::AsyncReadExt;
use tracing::{debug, warn};

/// What a command produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    pub stdout: String,
    pub stderr: String,
    /// `None` when the process was killed by a signal.
    pub exit_code: Option<i32>,
}

impl CommandOutput {
    pub fn is_success(&self) -> bool {
        self.exit_code == Some(0)
    }
}

/// A command to run.
#[derive(Debug, Clone)]
pub struct Command {
    pub command: String,
    pub timeout: Duration,
    pub env: HashMap<String, String>,
    pub working_dir: Option<String>,
}

impl Command {
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            timeout: Duration::from_secs(120),
            env: HashMap::new(),
            working_dir: None,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    pub fn with_working_dir(mut self, dir: impl Into<String>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }
}

/// Somewhere commands can be run.
#[async_trait]
pub trait Sandbox: Send + Sync {
    /// A name for logs and diagnostics.
    fn name(&self) -> &str;

    /// Whether this actually isolates anything.
    ///
    /// Exists so a caller can refuse to run untrusted input on a sandbox that
    /// is one in name only — see [`LocalEnvironment`].
    fn is_isolated(&self) -> bool;

    /// Prepare the sandbox. Called once before use.
    async fn start(&self) -> Result<(), StrandsError> {
        Ok(())
    }

    /// Run a command.
    async fn run(&self, command: &Command) -> Result<CommandOutput, StrandsError>;

    /// Tear down the sandbox.
    async fn stop(&self) -> Result<(), StrandsError> {
        Ok(())
    }
}

/// Runs commands directly on the host, with no isolation whatsoever.
///
/// Named to be hard to reach for by accident — upstream calls it
/// `not_a_sandbox_local_environment` for the same reason. It exists so the
/// [`Sandbox`] trait can be satisfied during development and testing; using it
/// with untrusted input provides exactly nothing.
pub struct LocalEnvironment;

#[async_trait]
impl Sandbox for LocalEnvironment {
    fn name(&self) -> &str {
        "local (not a sandbox)"
    }

    fn is_isolated(&self) -> bool {
        false
    }

    async fn run(&self, command: &Command) -> Result<CommandOutput, StrandsError> {
        warn!(
            command = %command.command,
            "Running on the host with no isolation"
        );
        run_local(command).await
    }
}

/// Runs commands inside a Docker container.
///
/// Shells out to the `docker` CLI rather than binding a Docker SDK, which keeps
/// this dependency-free and works with any daemon the CLI can reach — including
/// a remote one via `DOCKER_HOST`.
pub struct DockerSandbox {
    image: String,
    container_name: String,
    /// Extra flags for `docker run`, e.g. resource limits.
    extra_args: Vec<String>,
    network: Option<String>,
}

impl DockerSandbox {
    pub fn new(image: impl Into<String>) -> Self {
        Self {
            image: image.into(),
            container_name: format!("strands-{}", uuid_like()),
            // Off by default: an agent that can reach the network from inside
            // the sandbox has escaped most of the point of one.
            network: Some("none".to_string()),
            extra_args: Vec::new(),
        }
    }

    /// Give the container network access.
    ///
    /// Off by default. Enabling it means the sandbox contains filesystem and
    /// process damage but not exfiltration.
    pub fn with_network(mut self, network: impl Into<String>) -> Self {
        self.network = Some(network.into());
        self
    }

    pub fn with_extra_args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.extra_args = args.into_iter().map(Into::into).collect();
        self
    }

    pub fn container_name(&self) -> &str {
        &self.container_name
    }

    fn docker_args(&self, command: &Command) -> Vec<String> {
        let mut args = vec![
            "run".to_string(),
            "--rm".to_string(),
            "--name".to_string(),
            self.container_name.clone(),
        ];

        if let Some(network) = &self.network {
            args.push("--network".to_string());
            args.push(network.clone());
        }

        for (key, value) in &command.env {
            args.push("--env".to_string());
            args.push(format!("{key}={value}"));
        }

        if let Some(dir) = &command.working_dir {
            args.push("--workdir".to_string());
            args.push(dir.clone());
        }

        args.extend(self.extra_args.iter().cloned());
        args.push(self.image.clone());
        args.push("sh".to_string());
        args.push("-c".to_string());
        args.push(command.command.clone());
        args
    }
}

/// A container name suffix that does not need the `uuid` crate here.
fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}")
}

#[async_trait]
impl Sandbox for DockerSandbox {
    fn name(&self) -> &str {
        "docker"
    }

    fn is_isolated(&self) -> bool {
        true
    }

    async fn start(&self) -> Result<(), StrandsError> {
        // Fail here rather than on the first command, so a missing daemon is
        // reported at setup instead of looking like a tool failure later.
        let output = tokio::process::Command::new("docker")
            .arg("version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map_err(|e| {
                StrandsError::Other(format!("docker CLI not available: {e}"))
            })?;

        if !output.success() {
            return Err(StrandsError::Other(
                "docker is installed but the daemon is not reachable".to_string(),
            ));
        }
        Ok(())
    }

    async fn run(&self, command: &Command) -> Result<CommandOutput, StrandsError> {
        debug!(image = %self.image, command = %command.command, "Running in Docker");

        let mut cmd = tokio::process::Command::new("docker");
        cmd.args(self.docker_args(command))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null());

        capture(cmd, command.timeout).await
    }

    async fn stop(&self) -> Result<(), StrandsError> {
        // Best effort: `--rm` already removes the container on exit, so this
        // only matters for one still running.
        let _ = tokio::process::Command::new("docker")
            .args(["kill", &self.container_name])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
        Ok(())
    }
}

async fn run_local(command: &Command) -> Result<CommandOutput, StrandsError> {
    let shell = if cfg!(windows) { "cmd" } else { "sh" };
    let flag = if cfg!(windows) { "/C" } else { "-c" };

    let mut cmd = tokio::process::Command::new(shell);
    cmd.arg(flag)
        .arg(&command.command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null());

    for (key, value) in &command.env {
        cmd.env(key, value);
    }
    if let Some(dir) = &command.working_dir {
        cmd.current_dir(dir);
    }

    capture(cmd, command.timeout).await
}

/// Run a prepared command, capturing both streams under a timeout.
async fn capture(
    mut cmd: tokio::process::Command,
    timeout: Duration,
) -> Result<CommandOutput, StrandsError> {
    let mut child = cmd
        .spawn()
        .map_err(|e| StrandsError::Other(format!("failed to spawn: {e}")))?;

    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();

    let work = async {
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

    match tokio::time::timeout(timeout, work).await {
        Ok((stdout, stderr, status)) => Ok(CommandOutput {
            stdout,
            stderr,
            exit_code: status.ok().and_then(|s| s.code()),
        }),
        Err(_) => Err(StrandsError::Other(format!(
            "command timed out after {timeout:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_local_environment_reports_that_it_isolates_nothing() {
        // A caller must be able to refuse untrusted input on a sandbox that is
        // one in name only.
        assert!(!LocalEnvironment.is_isolated());
        assert!(LocalEnvironment.name().contains("not a sandbox"));
    }

    #[test]
    fn docker_reports_that_it_isolates() {
        assert!(DockerSandbox::new("alpine").is_isolated());
    }

    #[tokio::test]
    async fn local_execution_captures_output_and_exit_code() {
        let output = LocalEnvironment
            .run(&Command::new("echo hello"))
            .await
            .unwrap();

        assert!(output.is_success());
        assert!(output.stdout.contains("hello"));
    }

    #[tokio::test]
    async fn a_failing_command_reports_its_exit_code() {
        let output = LocalEnvironment
            .run(&Command::new("exit 7"))
            .await
            .unwrap();
        assert_eq!(output.exit_code, Some(7));
        assert!(!output.is_success());
    }

    #[tokio::test]
    async fn environment_variables_reach_the_command() {
        let output = LocalEnvironment
            .run(&Command::new("echo $STRANDS_TEST").with_env("STRANDS_TEST", "visible"))
            .await
            .unwrap();
        assert!(output.stdout.contains("visible"));
    }

    #[tokio::test]
    async fn a_hanging_command_times_out() {
        let result = LocalEnvironment
            .run(&Command::new("sleep 5").with_timeout(Duration::from_millis(100)))
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn docker_disables_networking_by_default() {
        // An agent that can reach the network from inside the sandbox has
        // escaped most of the point of one.
        let sandbox = DockerSandbox::new("alpine");
        let args = sandbox.docker_args(&Command::new("echo hi"));

        let network_index = args.iter().position(|a| a == "--network").expect("--network");
        assert_eq!(args[network_index + 1], "none");
    }

    #[test]
    fn docker_args_carry_env_workdir_and_the_command() {
        let sandbox = DockerSandbox::new("alpine").with_network("bridge");
        let args = sandbox.docker_args(
            &Command::new("ls -la")
                .with_env("KEY", "value")
                .with_working_dir("/work"),
        );

        assert!(args.contains(&"--rm".to_string()));
        assert!(args.contains(&"KEY=value".to_string()));
        assert!(args.contains(&"/work".to_string()));
        assert!(args.contains(&"alpine".to_string()));
        assert_eq!(args.last().unwrap(), "ls -la");

        let network_index = args.iter().position(|a| a == "--network").unwrap();
        assert_eq!(args[network_index + 1], "bridge");
    }

    #[test]
    fn container_names_are_unique_per_sandbox() {
        let a = DockerSandbox::new("alpine");
        std::thread::sleep(Duration::from_millis(2));
        let b = DockerSandbox::new("alpine");
        assert_ne!(a.container_name(), b.container_name());
    }
}
