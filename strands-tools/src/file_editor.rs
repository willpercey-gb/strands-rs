//! A tool for reading and editing files.
//!
//! Edits are expressed as exact string replacements rather than line numbers or
//! diffs: a model's line numbering drifts as soon as it makes one edit, whereas
//! an exact-match replacement either applies or fails loudly.
//!
//! Ported from upstream `vended_tools/file_editor/`.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};
use strands_core::error::StrandsError;
use strands_core::tool::{Tool, ToolContext, ToolOutput};
use strands_core::types::tools::{ToolAnnotations, ToolSpec};
use tracing::debug;

/// Cap on how much of a file is returned in one read.
pub const DEFAULT_MAX_READ_BYTES: usize = 200_000;

/// Reads and edits files beneath a root directory.
pub struct FileEditorTool {
    root: Option<PathBuf>,
    max_read_bytes: usize,
}

impl Default for FileEditorTool {
    fn default() -> Self {
        Self {
            root: None,
            max_read_bytes: DEFAULT_MAX_READ_BYTES,
        }
    }
}

impl FileEditorTool {
    /// Create with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Confine every operation to `root`.
    ///
    /// Strongly recommended: without it the model can read and rewrite anything
    /// the process can.
    /// Set the root.
    pub fn with_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.root = Some(root.into());
        self
    }

    /// Set the max read bytes.
    pub fn with_max_read_bytes(mut self, bytes: usize) -> Self {
        self.max_read_bytes = bytes;
        self
    }

    /// Resolve a caller-supplied path, refusing anything outside the root.
    ///
    /// Checks the *lexical* path for traversal before touching the filesystem,
    /// then verifies the canonical path when the file exists — the second check
    /// is what catches a symlink pointing outside.
    fn resolve(&self, path: &str) -> Result<PathBuf, String> {
        let candidate = Path::new(path);

        let Some(root) = &self.root else {
            return Ok(candidate.to_path_buf());
        };

        if candidate
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(format!("path must not contain '..': {path}"));
        }

        let joined = if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            root.join(candidate)
        };

        // For an existing file, canonicalize both sides so a symlink out of the
        // root is caught. For a new file, the lexical check above stands.
        match (joined.canonicalize(), root.canonicalize()) {
            (Ok(real), Ok(real_root)) if !real.starts_with(&real_root) => {
                Err(format!("path escapes the configured root: {path}"))
            }
            _ if !joined.starts_with(root) => {
                Err(format!("path escapes the configured root: {path}"))
            }
            _ => Ok(joined),
        }
    }
}

#[async_trait]
impl Tool for FileEditorTool {
    fn name(&self) -> &str {
        "file_editor"
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "file_editor",
            "Read, create or edit a file. Use command='view' to read, \
             'create' to write a new file, and 'replace' to substitute an exact \
             string. Replacement requires old_str to appear exactly once.",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "create", "replace"],
                        "description": "The operation to perform"
                    },
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {
                        "type": "string",
                        "description": "File content, for command='create'"
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Exact text to replace, for command='replace'"
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement text, for command='replace'"
                    }
                },
                "required": ["command", "path"]
            }),
        )
        .with_annotations(ToolAnnotations {
            read_only_hint: Some(false),
            destructive_hint: Some(true),
            idempotent_hint: Some(false),
            open_world_hint: Some(false),
            ..Default::default()
        })
    }

    async fn invoke(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, StrandsError> {
        let Some(command) = input.get("command").and_then(Value::as_str) else {
            return Ok(ToolOutput::error("'command' must be a string"));
        };
        let Some(path_arg) = input.get("path").and_then(Value::as_str) else {
            return Ok(ToolOutput::error("'path' must be a string"));
        };

        let path = match self.resolve(path_arg) {
            Ok(path) => path,
            Err(reason) => return Ok(ToolOutput::error(reason)),
        };

        match command {
            "view" => {
                debug!(?path, "Reading file");
                match tokio::fs::read_to_string(&path).await {
                    Ok(content) if content.len() > self.max_read_bytes => {
                        let kept: String = content.chars().take(self.max_read_bytes).collect();
                        Ok(ToolOutput::success(format!(
                            "{kept}\n[truncated at {} bytes]",
                            self.max_read_bytes
                        )))
                    }
                    Ok(content) => Ok(ToolOutput::success(content)),
                    Err(e) => Ok(ToolOutput::error(format!("could not read {path:?}: {e}"))),
                }
            }

            "create" => {
                let content = input
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default();

                if let Some(parent) = path.parent() {
                    if let Err(e) = tokio::fs::create_dir_all(parent).await {
                        return Ok(ToolOutput::error(format!(
                            "could not create {parent:?}: {e}"
                        )));
                    }
                }

                debug!(?path, bytes = content.len(), "Writing file");
                match tokio::fs::write(&path, content).await {
                    Ok(()) => Ok(ToolOutput::success(format!(
                        "Wrote {} bytes to {}",
                        content.len(),
                        path.display()
                    ))),
                    Err(e) => Ok(ToolOutput::error(format!("could not write {path:?}: {e}"))),
                }
            }

            "replace" => {
                let Some(old) = input.get("old_str").and_then(Value::as_str) else {
                    return Ok(ToolOutput::error("'old_str' is required for replace"));
                };
                let new = input.get("new_str").and_then(Value::as_str).unwrap_or("");

                let content = match tokio::fs::read_to_string(&path).await {
                    Ok(content) => content,
                    Err(e) => {
                        return Ok(ToolOutput::error(format!("could not read {path:?}: {e}")))
                    }
                };

                // Requiring a unique match is the whole safety property: a
                // replacement that matches twice would silently edit somewhere
                // the model did not intend.
                let matches = content.matches(old).count();
                match matches {
                    0 => {
                        return Ok(ToolOutput::error(
                            "old_str was not found in the file; it must match exactly",
                        ))
                    }
                    1 => {}
                    n => {
                        return Ok(ToolOutput::error(format!(
                            "old_str matched {n} times; it must match exactly once. \
                             Include more surrounding context to disambiguate."
                        )))
                    }
                }

                let updated = content.replacen(old, new, 1);
                match tokio::fs::write(&path, updated).await {
                    Ok(()) => Ok(ToolOutput::success(format!(
                        "Replaced one occurrence in {}",
                        path.display()
                    ))),
                    Err(e) => Ok(ToolOutput::error(format!("could not write {path:?}: {e}"))),
                }
            }

            other => Ok(ToolOutput::error(format!(
                "unknown command '{other}'; expected view, create or replace"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("strands-editor-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    async fn call(tool: &FileEditorTool, input: Value) -> ToolOutput {
        tool.invoke(input, &ToolContext::default()).await.unwrap()
    }

    #[tokio::test]
    async fn create_then_view_round_trip() {
        let root = temp_root();
        let tool = FileEditorTool::new().with_root(&root);

        let created = call(
            &tool,
            json!({"command": "create", "path": "a.txt", "content": "hello"}),
        )
        .await;
        assert!(!created.is_error);

        let viewed = call(&tool, json!({"command": "view", "path": "a.txt"})).await;
        assert_eq!(viewed.content.as_str().unwrap(), "hello");

        let _ = std::fs::remove_file(root.join("a.txt"));
    }

    #[tokio::test]
    async fn replace_substitutes_a_unique_match() {
        let root = temp_root();
        let tool = FileEditorTool::new().with_root(&root);

        call(
            &tool,
            json!({"command": "create", "path": "b.txt", "content": "one two three"}),
        )
        .await;

        let replaced = call(
            &tool,
            json!({"command": "replace", "path": "b.txt", "old_str": "two", "new_str": "2"}),
        )
        .await;
        assert!(!replaced.is_error);

        let viewed = call(&tool, json!({"command": "view", "path": "b.txt"})).await;
        assert_eq!(viewed.content.as_str().unwrap(), "one 2 three");

        let _ = std::fs::remove_file(root.join("b.txt"));
    }

    #[tokio::test]
    async fn an_ambiguous_replacement_is_refused() {
        // Editing the wrong occurrence silently is far worse than failing.
        let root = temp_root();
        let tool = FileEditorTool::new().with_root(&root);

        call(
            &tool,
            json!({"command": "create", "path": "c.txt", "content": "x x x"}),
        )
        .await;

        let out = call(
            &tool,
            json!({"command": "replace", "path": "c.txt", "old_str": "x", "new_str": "y"}),
        )
        .await;

        assert!(out.is_error);
        assert!(out.content.as_str().unwrap().contains("matched 3 times"));

        let viewed = call(&tool, json!({"command": "view", "path": "c.txt"})).await;
        assert_eq!(
            viewed.content.as_str().unwrap(),
            "x x x",
            "a refused edit must not modify the file"
        );

        let _ = std::fs::remove_file(root.join("c.txt"));
    }

    #[tokio::test]
    async fn a_missing_match_is_reported() {
        let root = temp_root();
        let tool = FileEditorTool::new().with_root(&root);

        call(
            &tool,
            json!({"command": "create", "path": "d.txt", "content": "abc"}),
        )
        .await;

        let out = call(
            &tool,
            json!({"command": "replace", "path": "d.txt", "old_str": "zzz", "new_str": "y"}),
        )
        .await;
        assert!(out.is_error);
        assert!(out.content.as_str().unwrap().contains("not found"));

        let _ = std::fs::remove_file(root.join("d.txt"));
    }

    #[tokio::test]
    async fn traversal_outside_the_root_is_refused() {
        let tool = FileEditorTool::new().with_root(temp_root());

        for path in ["../escape.txt", "../../etc/passwd"] {
            let out = call(&tool, json!({"command": "view", "path": path})).await;
            assert!(out.is_error, "expected {path} to be refused");
        }
    }

    #[tokio::test]
    async fn view_truncates_a_large_file_visibly() {
        let root = temp_root();
        let tool = FileEditorTool::new()
            .with_root(&root)
            .with_max_read_bytes(10);

        call(
            &tool,
            json!({"command": "create", "path": "big.txt", "content": "x".repeat(500)}),
        )
        .await;

        let out = call(&tool, json!({"command": "view", "path": "big.txt"})).await;
        assert!(out.content.as_str().unwrap().contains("truncated"));

        let _ = std::fs::remove_file(root.join("big.txt"));
    }

    #[tokio::test]
    async fn an_unknown_command_is_rejected() {
        let tool = FileEditorTool::new().with_root(temp_root());
        let out = call(&tool, json!({"command": "delete", "path": "a.txt"})).await;
        assert!(out.is_error);
        assert!(out.content.as_str().unwrap().contains("unknown command"));
    }

    #[tokio::test]
    async fn create_makes_missing_directories() {
        let root = temp_root();
        let tool = FileEditorTool::new().with_root(&root);

        let out = call(
            &tool,
            json!({"command": "create", "path": "nested/deep/e.txt", "content": "x"}),
        )
        .await;
        assert!(!out.is_error, "{:?}", out.content);

        let _ = std::fs::remove_dir_all(root.join("nested"));
    }
}
