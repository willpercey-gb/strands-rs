//! Translator from `gemini --output-format stream-json` line events
//! into `strands_core::StreamEvent`. The CLI emits one JSON object per
//! line:
//!
//! - `{"type":"init", "session_id":"...", "model":"...", ...}`
//! - `{"type":"message", "role":"user", "content":"<echoed input>"}`
//! - `{"type":"message", "role":"assistant", "content":"<chunk>", "delta":true}`
//!   — chunked streaming. `content` is the *next chunk* (not cumulative).
//!   Without `delta:true` it's the *full* assistant message in one go.
//! - `{"type":"result", "status":"success", "stats":{ ... }}` — terminal.
//!
//! We translate assistant messages into `ContentBlockDelta { TextDelta }`,
//! the result event into `Metadata { usage }` + `MessageStop`. Other
//! events (init, user echo) are no-ops.

use serde::Deserialize;
use strands_core::types::message::Role;
use strands_core::types::streaming::{
    ContentBlockType, DeltaContent, StopReason, StreamEvent, Usage,
};

#[derive(Default)]
pub(crate) struct GeminiCliState {
    saw_message_start: bool,
    text_block_open: bool,
    text_block_index: usize,
    /// Whether we've seen at least one assistant chunk this turn. Used
    /// by `non-delta` full-message events to avoid re-emitting text
    /// that already streamed in via deltas.
    saw_assistant_chunk: bool,
    /// Tracks the cumulative text emitted on the assistant block so
    /// that a final non-delta full message can produce just the
    /// remaining tail rather than duplicating what was already sent.
    emitted_text: String,
}

#[derive(Deserialize)]
struct StatsRaw {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    duration_ms: Option<u64>,
}

impl GeminiCliState {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn ingest_line(&mut self, line: &str) -> Vec<StreamEvent> {
        let line = line.trim();
        if line.is_empty() {
            return Vec::new();
        }
        let value: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("gemini-cli json decode failed: {e} line={line}");
                return Vec::new();
            }
        };
        let kind = value.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match kind {
            "init" => Vec::new(),
            "message" => self.handle_message(&value),
            "result" => self.handle_result(&value),
            other => {
                tracing::trace!("gemini-cli unrecognised type: {other}");
                Vec::new()
            }
        }
    }

    pub(crate) fn flush(&mut self) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        if self.text_block_open {
            self.close_text_block(&mut out);
            out.push(StreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
            });
        }
        out
    }

    fn handle_message(&mut self, value: &serde_json::Value) -> Vec<StreamEvent> {
        let role = value.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role != "assistant" {
            // user echo or system message — not interesting downstream.
            return Vec::new();
        }
        let content = match value.get("content").and_then(|c| c.as_str()) {
            Some(c) => c,
            None => return Vec::new(),
        };
        let is_delta = value
            .get("delta")
            .and_then(|d| d.as_bool())
            .unwrap_or(false);

        let mut out = Vec::new();
        self.ensure_text_block_open(&mut out);

        if is_delta {
            self.saw_assistant_chunk = true;
            self.emitted_text.push_str(content);
            out.push(StreamEvent::ContentBlockDelta {
                index: self.text_block_index,
                delta: DeltaContent::TextDelta(content.to_string()),
            });
        } else {
            // Non-delta message: the CLI emitted the whole assistant
            // turn as a single event (or a final "complete" message
            // after streaming). If we already streamed deltas with the
            // same content, only emit the remaining tail.
            let to_emit = if self.saw_assistant_chunk
                && content.starts_with(&self.emitted_text)
            {
                &content[self.emitted_text.len()..]
            } else {
                content
            };
            if !to_emit.is_empty() {
                self.emitted_text.push_str(to_emit);
                out.push(StreamEvent::ContentBlockDelta {
                    index: self.text_block_index,
                    delta: DeltaContent::TextDelta(to_emit.to_string()),
                });
            }
        }
        out
    }

    fn handle_result(&mut self, value: &serde_json::Value) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        self.close_text_block(&mut out);
        if let Some(stats) = value.get("stats") {
            out.push(StreamEvent::Metadata {
                usage: extract_usage(stats),
            });
        }
        let status = value.get("status").and_then(|s| s.as_str()).unwrap_or("");
        let stop_reason = match status {
            "success" => StopReason::EndTurn,
            "cancelled" => StopReason::Cancelled,
            _ => StopReason::EndTurn,
        };
        out.push(StreamEvent::MessageStop { stop_reason });
        out
    }

    fn ensure_text_block_open(&mut self, out: &mut Vec<StreamEvent>) {
        if !self.saw_message_start {
            self.saw_message_start = true;
            out.push(StreamEvent::MessageStart {
                role: Role::Assistant,
            });
        }
        if !self.text_block_open {
            self.text_block_open = true;
            out.push(StreamEvent::ContentBlockStart {
                index: self.text_block_index,
                content_type: ContentBlockType::Text,
            });
        }
    }

    fn close_text_block(&mut self, out: &mut Vec<StreamEvent>) {
        if self.text_block_open {
            out.push(StreamEvent::ContentBlockStop {
                index: self.text_block_index,
            });
            self.text_block_open = false;
        }
    }
}

fn extract_usage(stats: &serde_json::Value) -> Usage {
    let raw: StatsRaw = serde_json::from_value(stats.clone()).unwrap_or(StatsRaw {
        input_tokens: None,
        output_tokens: None,
        duration_ms: None,
    });
    Usage {
        input_tokens: raw.input_tokens,
        output_tokens: raw.output_tokens,
        total_duration_ns: raw.duration_ms.map(|ms| ms.saturating_mul(1_000_000)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(lines: &[&str]) -> Vec<StreamEvent> {
        let mut state = GeminiCliState::new();
        let mut events = Vec::new();
        for l in lines {
            events.extend(state.ingest_line(l));
        }
        events.extend(state.flush());
        events
    }

    fn assembled(events: &[StreamEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentBlockDelta {
                    delta: DeltaContent::TextDelta(t),
                    ..
                } => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    #[test]
    fn delta_chunks_concatenate_as_emitted() {
        let events = run(&[
            r#"{"type":"init","session_id":"s","model":"gemini-2.5-flash"}"#,
            r#"{"type":"message","role":"user","content":"hi"}"#,
            r#"{"type":"message","role":"assistant","content":"Hello","delta":true}"#,
            r#"{"type":"message","role":"assistant","content":" there","delta":true}"#,
            r#"{"type":"result","status":"success","stats":{"input_tokens":7,"output_tokens":2,"duration_ms":900}}"#,
        ]);
        assert_eq!(assembled(&events), "Hello there");
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::Metadata { usage: Usage { input_tokens: Some(7), output_tokens: Some(2), .. } }
        )));
        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::MessageStop { stop_reason: StopReason::EndTurn })));
    }

    #[test]
    fn non_delta_full_message_emits_once() {
        let events = run(&[
            r#"{"type":"init","session_id":"s","model":"gemini-2.5-flash"}"#,
            r#"{"type":"message","role":"assistant","content":"complete answer"}"#,
            r#"{"type":"result","status":"success","stats":{"input_tokens":1,"output_tokens":2}}"#,
        ]);
        assert_eq!(assembled(&events), "complete answer");
    }

    #[test]
    fn non_delta_after_deltas_emits_only_tail() {
        // Defensive: if the CLI sends a "complete" message after
        // deltas with the same prefix, we should only emit the
        // remaining tail, not the full text again.
        let events = run(&[
            r#"{"type":"message","role":"assistant","content":"Hello","delta":true}"#,
            r#"{"type":"message","role":"assistant","content":"Hello world"}"#,
            r#"{"type":"result","status":"success","stats":{}}"#,
        ]);
        assert_eq!(assembled(&events), "Hello world");
    }

    #[test]
    fn user_messages_are_ignored() {
        let events = run(&[
            r#"{"type":"message","role":"user","content":"echo me"}"#,
            r#"{"type":"message","role":"assistant","content":"reply","delta":true}"#,
            r#"{"type":"result","status":"success","stats":{}}"#,
        ]);
        assert_eq!(assembled(&events), "reply");
    }

    #[test]
    fn malformed_lines_are_dropped() {
        let events = run(&["not json", "{}", ""]);
        assert!(assembled(&events).is_empty());
    }
}
