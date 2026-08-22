//! Shared helpers for finding safe conversation trim boundaries.
//!
//! Trimming a conversation is not simply "drop the oldest N messages".
//! Providers reject histories that begin with an orphaned `ToolResult`, or
//! that contain a `ToolUse` whose matching `ToolResult` was dropped. These
//! helpers walk a proposed boundary forward until it lands somewhere the
//! provider will accept.
//!
//! Ported from upstream `agent/conversation_manager/compression/context_compression.py`.

use crate::types::message::{Message, Role};

/// Find a valid trim point at or after `start_index`.
///
/// A valid trim point must:
/// 1. be a user message (required by most providers),
/// 2. not be an orphaned `ToolResult`,
/// 3. not be a `ToolUse` unless its `ToolResult` immediately follows.
///
/// Returns `messages.len()` when no such point exists.
pub fn find_valid_trim_point(messages: &[Message], start_index: usize) -> usize {
    let mut trim_index = start_index;

    while trim_index < messages.len() {
        let message = &messages[trim_index];

        if message.role != Role::User {
            trim_index += 1;
            continue;
        }

        if message.has_tool_result() {
            trim_index += 1;
            continue;
        }

        if message.has_tool_use() {
            let next_has_tool_result = messages
                .get(trim_index + 1)
                .is_some_and(|m| m.has_tool_result());
            if !next_has_tool_result {
                trim_index += 1;
                continue;
            }
        }

        break;
    }

    trim_index
}

/// Find the first `assistant(ToolUse)` + `user(ToolResult)` boundary at or
/// after `start_index`.
///
/// Used as a fallback when [`find_valid_trim_point`] finds no plain user
/// message. Providers treat a complete tool-use/tool-result pair as a valid
/// conversation continuation, so trimming to such a boundary keeps tool-heavy
/// conversations trimmable at all — without it, an agent that only ever
/// exchanges tool calls can never reduce its context.
pub fn find_tool_pair_trim_point(messages: &[Message], start_index: usize) -> Option<usize> {
    (start_index..messages.len()).find(|&index| {
        messages[index].has_tool_use()
            && messages
                .get(index + 1)
                .is_some_and(|next| next.role == Role::User && next.has_tool_result())
    })
}

/// Adjust a split point forward so it does not sever a tool-use/tool-result
/// pair.
///
/// Returns `None` when the split point cannot be made valid — the caller
/// decides whether that is an error (reactive overflow recovery) or merely
/// something to log (routine management).
pub fn adjust_split_point_for_tool_pairs(
    messages: &[Message],
    split_point: usize,
) -> Option<usize> {
    if split_point > messages.len() {
        return None;
    }
    if split_point == messages.len() {
        return Some(split_point);
    }

    let mut split_point = split_point;
    while split_point < messages.len() {
        let message = &messages[split_point];

        // The oldest message cannot be a ToolResult — it needs a preceding
        // ToolUse that we are about to drop.
        let orphaned_result = message.has_tool_result();

        // A ToolUse may lead only if its ToolResult immediately follows.
        let dangling_use = message.has_tool_use()
            && !messages
                .get(split_point + 1)
                .is_some_and(|m| m.has_tool_result());

        if orphaned_result || dangling_use {
            split_point += 1;
        } else {
            return Some(split_point);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::content::{ContentBlock, ToolResultContent, ToolResultStatus};
    use serde_json::json;

    fn user(text: &str) -> Message {
        Message::user(text)
    }

    fn assistant_tool_use(id: &str) -> Message {
        Message::assistant(vec![ContentBlock::ToolUse {
            tool_use_id: id.into(),
            name: "t".into(),
            input: json!({}),
        }])
    }

    fn user_tool_result(id: &str) -> Message {
        Message::new(
            Role::User,
            vec![ContentBlock::ToolResult {
                tool_use_id: id.into(),
                status: ToolResultStatus::Success,
                content: vec![ToolResultContent::Text { text: "ok".into() }],
            }],
        )
    }

    fn assistant_text(text: &str) -> Message {
        Message::assistant(vec![ContentBlock::Text { text: text.into() }])
    }

    #[test]
    fn trim_point_lands_on_a_plain_user_message() {
        let msgs = vec![
            user("one"),
            assistant_text("a"),
            user("two"),
            assistant_text("b"),
        ];
        assert_eq!(find_valid_trim_point(&msgs, 1), 2);
    }

    #[test]
    fn trim_point_skips_an_orphaned_tool_result() {
        let msgs = vec![
            assistant_tool_use("1"),
            user_tool_result("1"),
            user("plain"),
        ];
        // Index 1 is a toolResult — not a legal start. Walk to the plain user.
        assert_eq!(find_valid_trim_point(&msgs, 1), 2);
    }

    #[test]
    fn trim_point_returns_len_when_no_valid_point_exists() {
        let msgs = vec![assistant_tool_use("1"), user_tool_result("1")];
        assert_eq!(find_valid_trim_point(&msgs, 0), msgs.len());
    }

    #[test]
    fn tool_pair_fallback_finds_the_use_result_boundary() {
        let msgs = vec![
            user("start"),
            assistant_tool_use("1"),
            user_tool_result("1"),
            assistant_tool_use("2"),
            user_tool_result("2"),
        ];
        assert_eq!(find_tool_pair_trim_point(&msgs, 1), Some(1));
        assert_eq!(find_tool_pair_trim_point(&msgs, 2), Some(3));
    }

    #[test]
    fn tool_pair_fallback_returns_none_without_a_complete_pair() {
        let msgs = vec![user("start"), assistant_tool_use("1")];
        assert_eq!(find_tool_pair_trim_point(&msgs, 0), None);
    }

    #[test]
    fn split_point_walks_past_a_dangling_tool_use() {
        let msgs = vec![
            assistant_tool_use("1"),
            assistant_text("no result followed"),
            user("plain"),
        ];
        assert_eq!(adjust_split_point_for_tool_pairs(&msgs, 0), Some(1));
    }

    #[test]
    fn split_point_at_end_is_valid() {
        let msgs = vec![user("a")];
        assert_eq!(adjust_split_point_for_tool_pairs(&msgs, 1), Some(1));
    }

    #[test]
    fn split_point_beyond_end_is_rejected() {
        let msgs = vec![user("a")];
        assert_eq!(adjust_split_point_for_tool_pairs(&msgs, 2), None);
    }
}
