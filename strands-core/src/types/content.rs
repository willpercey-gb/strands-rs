use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::citations::CitationsContentBlock;
use super::media::{AudioContent, DocumentContent, ImageContent, VideoContent};

/// A cache point marks a prefix boundary providers can reuse across calls.
///
/// Everything *before* the point is eligible for prompt caching, which is why
/// placement matters more than the value itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CachePoint {
    /// Cache point kind. Providers currently only define `"default"`.
    #[serde(rename = "type")]
    pub cache_type: String,
    /// How long the provider should retain the cached prefix, e.g. `"5m"`,
    /// `"1h"`. Only honoured by providers that accept Anthropic-compatible
    /// `cache_control` fields; ignored elsewhere.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

impl Default for CachePoint {
    fn default() -> Self {
        Self {
            cache_type: "default".to_string(),
            ttl: None,
        }
    }
}

impl CachePoint {
    /// Create with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a retention hint, e.g. `"5m"` or `"1h"`.
    /// Set the ttl.
    pub fn with_ttl(mut self, ttl: impl Into<String>) -> Self {
        self.ttl = Some(ttl.into());
        self
    }
}

/// Reasoning the model produced on the way to its answer.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningContent {
    /// The reasoning text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Token verifying the reasoning was produced by the model. Must be
    /// round-tripped back to the provider or the reasoning is rejected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Reasoning the provider encrypted for safety reasons, base64-encoded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted_content: Option<String>,
}

/// Text submitted for guardrail evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardContent {
    /// The text to evaluate.
    pub text: String,
    /// Qualifiers describing the block's role. Optional — providers treat an
    /// absent list as unqualified.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub qualifiers: Vec<String>,
}

/// A single block of content within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text.
    Text {
        /// The text.
        text: String,
    },
    /// A tool call the model is requesting.
    ToolUse {
        /// Identifier the matching `ToolResult` must echo back.
        tool_use_id: String,
        /// Name of the tool to invoke.
        name: String,
        /// Arguments, matching the tool's input schema.
        input: Value,
    },
    /// The outcome of a tool call.
    ToolResult {
        /// Identifier of the `ToolUse` this answers.
        tool_use_id: String,
        /// Whether the call succeeded.
        status: ToolResultStatus,
        /// What the tool returned.
        content: Vec<ToolResultContent>,
    },
    /// An image.
    Image(ImageContent),
    /// A document.
    Document(DocumentContent),
    /// Audio.
    Audio(AudioContent),
    /// Video.
    Video(VideoContent),
    /// Reasoning the model produced on the way to its answer.
    Reasoning(ReasoningContent),
    /// Generated text together with the sources it cites.
    Citations(CitationsContentBlock),
    /// Text submitted for guardrail evaluation.
    GuardContent(GuardContent),
    /// A prompt-cache boundary. Positional, not content.
    CachePoint(CachePoint),
}

impl ContentBlock {
    /// Extract text content if this is a Text block.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text { text } => Some(text),
            _ => None,
        }
    }

    /// Check if this is a ToolUse block.
    pub fn is_tool_use(&self) -> bool {
        matches!(self, ContentBlock::ToolUse { .. })
    }

    /// Check if this is a ToolResult block.
    pub fn is_tool_result(&self) -> bool {
        matches!(self, ContentBlock::ToolResult { .. })
    }

    /// Whether this block carries content for the model to read, as opposed to
    /// SDK-level bookkeeping.
    ///
    /// Cache points are positional markers, not content: adapters that count or
    /// concatenate blocks should skip them.
    pub fn is_content(&self) -> bool {
        !matches!(self, ContentBlock::CachePoint(_))
    }

    /// Convenience constructor for an inline base64 image.
    pub fn image(format: super::media::ImageFormat, data: impl Into<String>) -> Self {
        ContentBlock::Image(ImageContent {
            format,
            source: super::media::MediaSource::bytes(data),
        })
    }

    /// Convenience constructor for a cache point with default settings.
    pub fn cache_point() -> Self {
        ContentBlock::CachePoint(CachePoint::default())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Whether a tool call succeeded.
pub enum ToolResultStatus {
    /// The tool completed.
    Success,
    /// The tool failed; the content explains why.
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
/// One part of what a tool returned.
pub enum ToolResultContent {
    /// Textual output.
    Text {
        /// The text.
        text: String,
    },
    /// Structured output.
    Json {
        /// The value.
        value: Value,
    },
    /// An image the tool produced.
    Image(ImageContent),
    /// A document the tool produced.
    Document(DocumentContent),
}

/// A block within a system prompt.
///
/// System prompts are no longer a bare string: a cache point placed inside one
/// lets providers reuse the prefix across calls, which is the single largest
/// cost lever for agents with long instructions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SystemContentBlock {
    /// A run of system-prompt text.
    Text {
        /// The text.
        text: String,
    },
    /// A prompt-cache boundary within the system prompt.
    CachePoint(CachePoint),
}

/// A system prompt, either plain text or structured blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    /// Plain text, for providers with no block support.
    Text(String),
    /// Structured blocks, which can carry cache points.
    Blocks(Vec<SystemContentBlock>),
}

impl SystemPrompt {
    /// Split into the two forms adapters need: a plain-text rendering for
    /// providers that accept only a string, and the block list for those that
    /// support cache points.
    ///
    /// Mirrors upstream `split_system_prompt`.
    pub fn split(&self) -> (Option<String>, Vec<SystemContentBlock>) {
        match self {
            SystemPrompt::Text(text) => (
                Some(text.clone()),
                vec![SystemContentBlock::Text { text: text.clone() }],
            ),
            SystemPrompt::Blocks(blocks) => {
                let text_parts: Vec<&str> = blocks
                    .iter()
                    .filter_map(|b| match b {
                        SystemContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();

                let text = if text_parts.is_empty() {
                    None
                } else {
                    Some(text_parts.join("\n"))
                };

                (text, blocks.clone())
            }
        }
    }

    /// The plain-text rendering, for providers with no block support.
    pub fn as_text(&self) -> Option<String> {
        self.split().0
    }
}

impl From<String> for SystemPrompt {
    fn from(value: String) -> Self {
        SystemPrompt::Text(value)
    }
}

impl From<&str> for SystemPrompt {
    fn from(value: &str) -> Self {
        SystemPrompt::Text(value.to_string())
    }
}

impl From<Vec<SystemContentBlock>> for SystemPrompt {
    fn from(value: Vec<SystemContentBlock>) -> Self {
        SystemPrompt::Blocks(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_prompt_splits_to_both_forms() {
        let p = SystemPrompt::from("be helpful");
        let (text, blocks) = p.split();
        assert_eq!(text.as_deref(), Some("be helpful"));
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn block_prompt_joins_text_and_keeps_cache_points() {
        let p = SystemPrompt::Blocks(vec![
            SystemContentBlock::Text { text: "one".into() },
            SystemContentBlock::CachePoint(CachePoint::new().with_ttl("1h")),
            SystemContentBlock::Text { text: "two".into() },
        ]);
        let (text, blocks) = p.split();
        assert_eq!(text.as_deref(), Some("one\ntwo"));
        assert_eq!(blocks.len(), 3, "cache point must survive the split");
    }

    #[test]
    fn block_prompt_without_text_has_no_string_rendering() {
        let p = SystemPrompt::Blocks(vec![SystemContentBlock::CachePoint(CachePoint::new())]);
        assert_eq!(p.split().0, None);
    }

    #[test]
    fn cache_points_are_not_content() {
        assert!(!ContentBlock::cache_point().is_content());
        assert!(ContentBlock::Text { text: "x".into() }.is_content());
    }

    #[test]
    fn cache_point_ttl_is_omitted_when_absent() {
        let json = serde_json::to_string(&CachePoint::new()).unwrap();
        assert!(
            !json.contains("ttl"),
            "unset ttl must not serialize: {json}"
        );

        let json = serde_json::to_string(&CachePoint::new().with_ttl("5m")).unwrap();
        assert!(json.contains("5m"));
    }
}
