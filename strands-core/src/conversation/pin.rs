//! Message pinning — protecting messages from eviction during context reduction.
//!
//! A pin is stored on the message itself (`metadata.custom.pinned`), so it
//! survives session save/restore rather than living in manager-side state that
//! a restart would lose.
//!
//! Pins propagate across a tool pair: pinning an `assistant(ToolUse)` implicitly
//! protects the `user(ToolResult)` that answers it, and vice versa. Without
//! that, pinning half a pair would produce exactly the orphaned-tool-block
//! history that [`super::trim`](crate::conversation::trim) exists to avoid.
//!
//! Ported from upstream
//! `agent/conversation_manager/compression/pin_message.py`.

use std::collections::HashSet;

use crate::types::content::ContentBlock;
use crate::types::message::{Message, MessageMetadata};

const PINNED_KEY: &str = "pinned";

/// Collect the tool-use ids referenced by a message, from either side of a pair.
fn tool_use_ids(message: &Message) -> HashSet<&str> {
    message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolUse { tool_use_id, .. } => Some(tool_use_id.as_str()),
            ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
            _ => None,
        })
        .collect()
}

/// Whether the message carries an explicit pin flag.
fn has_pinned_flag(message: &Message) -> bool {
    message
        .metadata
        .as_ref()
        .and_then(|m| m.custom.get(PINNED_KEY))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Whether the message at `index` is protected from eviction.
///
/// True when the message is pinned directly, or when an adjacent message
/// sharing a tool-use id is pinned.
pub fn is_pinned(messages: &[Message], index: usize) -> bool {
    let Some(message) = messages.get(index) else {
        return false;
    };

    if has_pinned_flag(message) {
        return true;
    }

    let my_ids = tool_use_ids(message);
    if my_ids.is_empty() {
        return false;
    }

    // Only the immediate neighbours can be this message's tool-pair partner.
    let neighbours = [index.checked_sub(1), index.checked_add(1)];
    neighbours
        .into_iter()
        .flatten()
        .filter_map(|i| messages.get(i))
        .any(|neighbour| {
            has_pinned_flag(neighbour) && !my_ids.is_disjoint(&tool_use_ids(neighbour))
        })
}

/// Pin the message at `index`, protecting it from eviction.
pub fn pin_message(messages: &mut [Message], index: usize) {
    let Some(message) = messages.get_mut(index) else {
        return;
    };
    message
        .metadata
        .get_or_insert_with(MessageMetadata::default)
        .custom
        .insert(PINNED_KEY.to_string(), serde_json::Value::Bool(true));
}

/// Unpin the message at `index`, allowing it to be evicted again.
///
/// Clears the metadata envelope entirely when nothing else is left in it, so
/// an unpinned message serializes identically to one that was never pinned.
pub fn unpin_message(messages: &mut [Message], index: usize) {
    let Some(message) = messages.get_mut(index) else {
        return;
    };
    let Some(metadata) = message.metadata.as_mut() else {
        return;
    };

    metadata.custom.remove(PINNED_KEY);
    if metadata.is_empty() {
        message.metadata = None;
    }
}

/// Pin the first `count` messages permanently.
///
/// Typically used to protect the opening instructions of a conversation, which
/// stay relevant no matter how long the exchange runs.
pub fn apply_pin_first(messages: &mut [Message], count: usize) {
    for i in 0..count.min(messages.len()) {
        pin_message(messages, i);
    }
}

/// Indices in `range` that are *not* protected, i.e. eligible for eviction.
pub fn unpinned_indices(messages: &[Message], range: std::ops::Range<usize>) -> Vec<usize> {
    range.filter(|&i| !is_pinned(messages, i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::content::{ToolResultContent, ToolResultStatus};
    use crate::types::message::Role;
    use serde_json::json;

    fn tool_use(id: &str) -> Message {
        Message::assistant(vec![ContentBlock::ToolUse {
            tool_use_id: id.into(),
            name: "t".into(),
            input: json!({}),
        }])
    }

    fn tool_result(id: &str) -> Message {
        Message::new(
            Role::User,
            vec![ContentBlock::ToolResult {
                tool_use_id: id.into(),
                status: ToolResultStatus::Success,
                content: vec![ToolResultContent::Text { text: "ok".into() }],
            }],
        )
    }

    #[test]
    fn unpinned_by_default() {
        let msgs = vec![Message::user("a")];
        assert!(!is_pinned(&msgs, 0));
    }

    #[test]
    fn pin_and_unpin_round_trip() {
        let mut msgs = vec![Message::user("a")];
        pin_message(&mut msgs, 0);
        assert!(is_pinned(&msgs, 0));

        unpin_message(&mut msgs, 0);
        assert!(!is_pinned(&msgs, 0));
        assert!(
            msgs[0].metadata.is_none(),
            "an unpinned message should serialize like one never pinned"
        );
    }

    #[test]
    fn unpin_preserves_other_metadata() {
        let mut msgs = vec![Message::user("a")];
        pin_message(&mut msgs, 0);
        msgs[0]
            .metadata
            .as_mut()
            .unwrap()
            .custom
            .insert("other".into(), json!("keep me"));

        unpin_message(&mut msgs, 0);
        assert!(
            msgs[0].metadata.is_some(),
            "unrelated metadata must survive"
        );
        assert!(!is_pinned(&msgs, 0));
    }

    #[test]
    fn pinning_a_tool_use_protects_its_result() {
        // Evicting half a pair leaves an orphan the provider rejects, so the
        // pin has to cover both halves.
        let mut msgs = vec![tool_use("1"), tool_result("1")];
        pin_message(&mut msgs, 0);

        assert!(is_pinned(&msgs, 0));
        assert!(is_pinned(&msgs, 1), "the matching result must be protected");
    }

    #[test]
    fn pinning_a_tool_result_protects_its_use() {
        let mut msgs = vec![tool_use("1"), tool_result("1")];
        pin_message(&mut msgs, 1);
        assert!(
            is_pinned(&msgs, 0),
            "the originating call must be protected"
        );
    }

    #[test]
    fn pair_protection_requires_a_matching_id() {
        // Adjacency alone is not enough — a different tool call is unrelated.
        let mut msgs = vec![tool_use("1"), tool_result("2")];
        pin_message(&mut msgs, 0);
        assert!(!is_pinned(&msgs, 1));
    }

    #[test]
    fn pair_protection_does_not_reach_beyond_neighbours() {
        let mut msgs = vec![tool_use("1"), Message::user("gap"), tool_result("1")];
        pin_message(&mut msgs, 0);
        assert!(
            !is_pinned(&msgs, 2),
            "a non-adjacent message is not a tool-pair partner"
        );
    }

    #[test]
    fn pin_first_protects_the_opening_messages() {
        let mut msgs = vec![Message::user("a"), Message::user("b"), Message::user("c")];
        apply_pin_first(&mut msgs, 2);

        assert!(is_pinned(&msgs, 0));
        assert!(is_pinned(&msgs, 1));
        assert!(!is_pinned(&msgs, 2));
    }

    #[test]
    fn pin_first_tolerates_a_count_beyond_the_end() {
        let mut msgs = vec![Message::user("a")];
        apply_pin_first(&mut msgs, 10);
        assert!(is_pinned(&msgs, 0));
    }

    #[test]
    fn unpinned_indices_skips_protected_messages() {
        let mut msgs = vec![Message::user("a"), Message::user("b"), Message::user("c")];
        pin_message(&mut msgs, 1);
        assert_eq!(unpinned_indices(&msgs, 0..3), vec![0, 2]);
    }

    #[test]
    fn pins_survive_a_serde_round_trip() {
        // The pin lives on the message, so a restored session keeps it.
        let mut msgs = vec![Message::user("a")];
        pin_message(&mut msgs, 0);

        let json = serde_json::to_string(&msgs[0]).unwrap();
        let restored: Message = serde_json::from_str(&json).unwrap();
        assert!(is_pinned(&[restored], 0));
    }
}
