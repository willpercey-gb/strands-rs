//! Character-based token estimation.
//!
//! A dependency-free approximation used for proactive context management —
//! deciding *when* to compress, not what to bill. Text is estimated at
//! chars/4 and JSON at chars/2, matching upstream's fallback heuristic.
//!
//! Accuracy varies by provider and tokenizer. A `Model` implementation with
//! access to a native counting API should override
//! [`Model::count_tokens`](super::Model::count_tokens) instead of relying on
//! this.

use serde_json::Value;

use crate::types::content::{ContentBlock, SystemContentBlock, SystemPrompt};
use crate::types::message::Message;
use crate::types::tools::ToolSpec;

/// Estimate tokens for a text run: roughly four characters per token.
pub fn estimate_text(text: &str) -> u64 {
    div_ceil(text.chars().count() as u64, 4)
}

/// Estimate tokens for a JSON value: roughly two characters per token.
///
/// Denser than prose because structural punctuation tokenizes poorly.
pub fn estimate_json(value: &Value) -> u64 {
    match serde_json::to_string(value) {
        Ok(s) => div_ceil(s.chars().count() as u64, 2),
        Err(_) => 0,
    }
}

fn div_ceil(n: u64, d: u64) -> u64 {
    n.div_ceil(d)
}

/// Estimate the tokens contributed by a single content block.
///
/// Binary payloads (image, audio, video, document bytes) are deliberately not
/// counted: their token cost is provider-specific and unrelated to their byte
/// length, so a character heuristic would be actively misleading.
pub fn estimate_content_block(block: &ContentBlock) -> u64 {
    match block {
        ContentBlock::Text { text } => estimate_text(text),
        ContentBlock::ToolUse { name, input, .. } => estimate_text(name) + estimate_json(input),
        ContentBlock::ToolResult { content, .. } => content
            .iter()
            .map(|item| match item {
                crate::types::content::ToolResultContent::Text { text } => estimate_text(text),
                crate::types::content::ToolResultContent::Json { value } => estimate_json(value),
                // Binary parts are not estimated — see the note above.
                _ => 0,
            })
            .sum(),
        ContentBlock::Reasoning(r) => r.text.as_deref().map(estimate_text).unwrap_or(0),
        ContentBlock::GuardContent(g) => estimate_text(&g.text),
        ContentBlock::Citations(c) => c
            .content
            .iter()
            .filter_map(|item| item.text.as_deref())
            .map(estimate_text)
            .sum(),
        // Cache points are positional markers, and media is not estimable.
        ContentBlock::CachePoint(_)
        | ContentBlock::Image(_)
        | ContentBlock::Audio(_)
        | ContentBlock::Video(_)
        | ContentBlock::Document(_) => 0,
    }
}

/// Estimate the total input tokens for a model call.
pub fn estimate_tokens(
    messages: &[Message],
    tool_specs: &[ToolSpec],
    system_prompt: Option<&SystemPrompt>,
) -> u64 {
    let mut total = 0u64;

    if let Some(prompt) = system_prompt {
        let (_, blocks) = prompt.split();
        for block in &blocks {
            if let SystemContentBlock::Text { text } = block {
                total += estimate_text(text);
            }
        }
    }

    for message in messages {
        for block in &message.content {
            total += estimate_content_block(block);
        }
    }

    for spec in tool_specs {
        // The whole spec crosses the wire as JSON, schema included.
        total += estimate_text(&spec.name);
        total += estimate_text(&spec.description);
        total += estimate_json(&spec.input_schema);
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::content::{CachePoint, ToolResultContent, ToolResultStatus};
    use crate::types::media::ImageFormat;
    use serde_json::json;

    #[test]
    fn text_is_four_chars_per_token_rounded_up() {
        assert_eq!(estimate_text(""), 0);
        assert_eq!(estimate_text("abcd"), 1);
        assert_eq!(estimate_text("abcde"), 2);
    }

    #[test]
    fn text_estimate_counts_characters_not_bytes() {
        // A multi-byte character is one character, not three.
        assert_eq!(estimate_text("日本語語"), 1);
    }

    #[test]
    fn json_is_denser_than_prose() {
        let value = json!({"key": "value"});
        assert!(
            estimate_json(&value) > estimate_text(&value.to_string()),
            "JSON should estimate higher per character than prose"
        );
    }

    #[test]
    fn binary_content_is_not_estimated() {
        // Byte length says nothing useful about a provider's image token cost.
        let img = ContentBlock::image(ImageFormat::Png, "A".repeat(10_000));
        assert_eq!(estimate_content_block(&img), 0);
        assert_eq!(
            estimate_content_block(&ContentBlock::CachePoint(CachePoint::new())),
            0
        );
    }

    #[test]
    fn tool_use_counts_name_and_input() {
        let block = ContentBlock::ToolUse {
            tool_use_id: "1".into(),
            name: "search".into(),
            input: json!({"query": "rust"}),
        };
        assert!(estimate_content_block(&block) > 0);
    }

    #[test]
    fn tool_result_counts_text_and_json_parts_only() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "1".into(),
            status: ToolResultStatus::Success,
            content: vec![
                ToolResultContent::Text {
                    text: "abcdefgh".into(),
                },
                ToolResultContent::Json { value: json!({}) },
            ],
        };
        assert_eq!(estimate_content_block(&block), 2 + 1);
    }

    #[test]
    fn system_prompt_and_tools_are_included() {
        let msgs = vec![Message::user("hello there")];
        let specs = vec![ToolSpec::new(
            "t",
            "does a thing",
            json!({"type": "object"}),
        )];
        let prompt = SystemPrompt::from("you are helpful");

        let with_all = estimate_tokens(&msgs, &specs, Some(&prompt));
        let messages_only = estimate_tokens(&msgs, &[], None);

        assert!(
            with_all > messages_only,
            "system prompt and tool specs must contribute to the estimate"
        );
    }

    #[test]
    fn empty_input_estimates_zero() {
        assert_eq!(estimate_tokens(&[], &[], None), 0);
    }
}
