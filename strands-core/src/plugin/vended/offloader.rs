//! Context offloading — moving oversized tool results out of the conversation.
//!
//! A single tool result can be larger than the whole rest of the context. This
//! plugin writes such results to [`Storage`] and leaves a short reference in
//! their place, with a retrieval tool the model can use to read back the parts
//! it actually needs.
//!
//! The trade is deliberate: the model loses immediate access to the full text
//! in exchange for the conversation surviving at all. A result that fits is
//! never touched.
//!
//! Ported from upstream `vended_plugins/context_offloader/`.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use tracing::debug;

use crate::error::StrandsError;
use crate::storage::Storage;
use crate::tool::{Tool, ToolContext, ToolOutput};
use crate::types::tools::{ToolAnnotations, ToolSpec};

/// Results larger than this are offloaded.
pub const DEFAULT_OFFLOAD_THRESHOLD: usize = 8_000;

/// How much of an offloaded result is previewed inline.
pub const DEFAULT_PREVIEW_CHARS: usize = 500;

/// Decides whether a given result should be offloaded.
///
/// Beyond the size threshold, because size is not the only reason: a result may
/// be small but sensitive, or large but needed in full.
pub type ShouldOffload = Arc<dyn Fn(&str, &str) -> bool + Send + Sync>;

/// Moves oversized tool results into storage.
pub struct ContextOffloader {
    storage: Arc<dyn Storage>,
    threshold: usize,
    preview_chars: usize,
    should_offload: Option<ShouldOffload>,
}

impl ContextOffloader {
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self {
            storage,
            threshold: DEFAULT_OFFLOAD_THRESHOLD,
            preview_chars: DEFAULT_PREVIEW_CHARS,
            should_offload: None,
        }
    }

    pub fn with_threshold(mut self, chars: usize) -> Self {
        self.threshold = chars;
        self
    }

    pub fn with_preview_chars(mut self, chars: usize) -> Self {
        self.preview_chars = chars;
        self
    }

    /// Override the offload decision.
    ///
    /// Receives the tool name and the result text. Upstream v1.51's
    /// `should_offload` callback.
    pub fn with_should_offload(
        mut self,
        predicate: impl Fn(&str, &str) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.should_offload = Some(Arc::new(predicate));
        self
    }

    /// Whether this result should be moved out of the conversation.
    pub fn should_offload(&self, tool_name: &str, text: &str) -> bool {
        match &self.should_offload {
            Some(predicate) => predicate(tool_name, text),
            None => text.len() > self.threshold,
        }
    }

    /// Offload `text`, returning the replacement to leave in the conversation.
    pub async fn offload(
        &self,
        tool_name: &str,
        tool_use_id: &str,
        text: &str,
    ) -> Result<String, StrandsError> {
        let key = format!("offload/{tool_use_id}");
        self.storage
            .write(&key, text.as_bytes().to_vec())
            .await?;

        let preview: String = text.chars().take(self.preview_chars).collect();
        debug!(
            tool_name,
            key,
            bytes = text.len(),
            "Offloaded oversized tool result"
        );

        // The reference names the exact key, so the model can retrieve it
        // without guessing — a vaguer marker would make the tool unusable.
        Ok(format!(
            "[Result offloaded: {} characters stored at key '{key}'. \
             Use the retrieve_offloaded tool to read it, optionally with a search term.]\n\n\
             Preview:\n{preview}",
            text.len()
        ))
    }

    /// The retrieval tool the model uses to read offloaded content back.
    pub fn retrieval_tool(&self) -> RetrieveOffloadedTool {
        RetrieveOffloadedTool {
            storage: self.storage.clone(),
            max_chars: 20_000,
        }
    }
}

/// Reads back offloaded content, optionally filtered by a search term.
pub struct RetrieveOffloadedTool {
    storage: Arc<dyn Storage>,
    max_chars: usize,
}

impl RetrieveOffloadedTool {
    pub fn with_max_chars(mut self, chars: usize) -> Self {
        self.max_chars = chars;
        self
    }

    /// Lines matching `needle`, with surrounding context.
    ///
    /// Grep rather than substring: retrieving a 10MB document as one blob would
    /// re-create the problem offloading solved.
    fn search(content: &str, needle: &str, max_chars: usize) -> String {
        let needle_lower = needle.to_lowercase();
        let lines: Vec<&str> = content.lines().collect();

        let mut out = String::new();
        let mut matches = 0;

        for (index, line) in lines.iter().enumerate() {
            if !line.to_lowercase().contains(&needle_lower) {
                continue;
            }
            matches += 1;

            let start = index.saturating_sub(1);
            let end = (index + 2).min(lines.len());
            for (offset, context_line) in lines[start..end].iter().enumerate() {
                out.push_str(&format!("{}: {context_line}\n", start + offset + 1));
            }
            out.push_str("---\n");

            if out.len() >= max_chars {
                out.push_str(&format!("[truncated after {matches} matches]\n"));
                break;
            }
        }

        if matches == 0 {
            format!("No lines matched '{needle}'.")
        } else {
            out
        }
    }
}

#[async_trait]
impl Tool for RetrieveOffloadedTool {
    fn name(&self) -> &str {
        "retrieve_offloaded"
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "retrieve_offloaded",
            "Read back a tool result that was offloaded out of the conversation. \
             Pass a search term to get only the matching lines, which is usually \
             what you want for a large document.",
            json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The storage key named in the offload notice"
                    },
                    "search": {
                        "type": "string",
                        "description": "Only return lines matching this term"
                    }
                },
                "required": ["key"]
            }),
        )
        .with_annotations(ToolAnnotations {
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
            ..Default::default()
        })
    }

    async fn invoke(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, StrandsError> {
        let Some(key) = input.get("key").and_then(Value::as_str) else {
            return Ok(ToolOutput::error("'key' must be a string"));
        };

        let Some(bytes) = self.storage.read(key).await? else {
            return Ok(ToolOutput::error(format!(
                "nothing stored under key '{key}'"
            )));
        };

        let content = String::from_utf8_lossy(&bytes);

        match input.get("search").and_then(Value::as_str) {
            Some(needle) if !needle.is_empty() => Ok(ToolOutput::success(Self::search(
                &content,
                needle,
                self.max_chars,
            ))),
            _ if content.len() > self.max_chars => {
                let kept: String = content.chars().take(self.max_chars).collect();
                Ok(ToolOutput::success(format!(
                    "{kept}\n[truncated at {} characters — pass a search term to narrow this]",
                    self.max_chars
                )))
            }
            _ => Ok(ToolOutput::success(content.into_owned())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::InMemoryStorage;

    fn offloader() -> (ContextOffloader, Arc<InMemoryStorage>) {
        let storage = Arc::new(InMemoryStorage::new());
        (ContextOffloader::new(storage.clone()), storage)
    }

    #[test]
    fn small_results_are_left_alone() {
        let (offloader, _) = offloader();
        assert!(!offloader.should_offload("t", "short"));
    }

    #[test]
    fn large_results_are_offloaded() {
        let (offloader, _) = offloader();
        let big = "x".repeat(DEFAULT_OFFLOAD_THRESHOLD + 1);
        assert!(offloader.should_offload("t", &big));
    }

    #[test]
    fn the_predicate_overrides_size() {
        // Size is not the only reason to offload — a result can be small but
        // sensitive, or large but needed in full.
        let (_, storage) = offloader();
        let offloader = ContextOffloader::new(storage)
            .with_should_offload(|tool, _| tool == "always_offload");

        assert!(offloader.should_offload("always_offload", "tiny"));
        assert!(!offloader.should_offload("other", &"x".repeat(100_000)));
    }

    #[tokio::test]
    async fn offloading_stores_the_content_and_leaves_a_reference() {
        let (offloader, storage) = offloader();
        let content = "line one\nline two\n".repeat(100);

        let replacement = offloader.offload("search", "call_1", &content).await.unwrap();

        assert!(replacement.contains("offload/call_1"), "{replacement}");
        assert!(replacement.contains("retrieve_offloaded"));
        assert!(replacement.contains("Preview:"));
        assert!(
            replacement.len() < content.len(),
            "the replacement must be smaller than what it replaced"
        );

        assert!(storage.read("offload/call_1").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn retrieval_reads_the_content_back() {
        let (offloader, _) = offloader();
        offloader.offload("t", "call_1", "the full content").await.unwrap();

        let out = offloader
            .retrieval_tool()
            .invoke(json!({"key": "offload/call_1"}), &ToolContext::default())
            .await
            .unwrap();

        assert!(!out.is_error);
        assert_eq!(out.content.as_str().unwrap(), "the full content");
    }

    #[tokio::test]
    async fn retrieval_of_an_unknown_key_is_an_error() {
        let (offloader, _) = offloader();
        let out = offloader
            .retrieval_tool()
            .invoke(json!({"key": "offload/nope"}), &ToolContext::default())
            .await
            .unwrap();
        assert!(out.is_error);
    }

    #[tokio::test]
    async fn search_returns_matching_lines_with_context() {
        let (offloader, _) = offloader();
        let content = "alpha\nbeta\nTARGET here\ndelta\nepsilon";
        offloader.offload("t", "call_1", content).await.unwrap();

        let out = offloader
            .retrieval_tool()
            .invoke(
                json!({"key": "offload/call_1", "search": "target"}),
                &ToolContext::default(),
            )
            .await
            .unwrap();

        let text = out.content.as_str().unwrap();
        assert!(text.contains("TARGET here"), "{text}");
        assert!(text.contains("beta"), "context line before should be included");
        assert!(text.contains("delta"), "context line after should be included");
        assert!(!text.contains("alpha"), "unrelated lines should be excluded");
    }

    #[tokio::test]
    async fn search_reports_when_nothing_matches() {
        let (offloader, _) = offloader();
        offloader.offload("t", "call_1", "nothing here").await.unwrap();

        let out = offloader
            .retrieval_tool()
            .invoke(
                json!({"key": "offload/call_1", "search": "zebra"}),
                &ToolContext::default(),
            )
            .await
            .unwrap();

        assert!(out.content.as_str().unwrap().contains("No lines matched"));
    }

    #[tokio::test]
    async fn a_full_read_of_a_huge_blob_is_truncated_with_advice() {
        // Returning the whole thing would re-create the problem offloading
        // solved.
        let (offloader, _) = offloader();
        let content = "x".repeat(50_000);
        offloader.offload("t", "call_1", &content).await.unwrap();

        let out = offloader
            .retrieval_tool()
            .with_max_chars(1_000)
            .invoke(json!({"key": "offload/call_1"}), &ToolContext::default())
            .await
            .unwrap();

        let text = out.content.as_str().unwrap();
        assert!(text.contains("truncated"));
        assert!(text.contains("search term"), "should advise how to narrow it");
    }

    #[test]
    fn retrieval_is_annotated_read_only() {
        let storage = Arc::new(InMemoryStorage::new());
        let spec = ContextOffloader::new(storage).retrieval_tool().spec();
        let annotations = spec.annotations.expect("annotations");
        assert!(annotations.is_read_only());
        assert!(!annotations.is_destructive());
    }
}
