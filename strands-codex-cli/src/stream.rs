//! Translator from `codex exec --json` ThreadEvent lines into
//! `strands_core::StreamEvent`. The CLI emits one JSON object per line
//! with a `type` discriminator (`thread.started`, `turn.started`,
//! `item.started|updated|completed`, `turn.completed`, `turn.failed`,
//! `error`).
//!
//! Codex's `item.updated` carries the full accumulated text for an
//! `agent_message` or `reasoning` item — not a delta — so we maintain
//! per-item-id "previous text" cursors and emit only the diff. This
//! matches the streaming semantics the rest of strands expects.
//!
//! Reasoning items are wrapped in `<thinking>...</thinking>` exactly
//! like the claude-cli adapter does, so the harness pipeline's
//! xml_unwrap routes them to ReasoningDelta naturally.

use std::collections::HashMap;

use serde::Deserialize;
use strands_core::types::message::Role;
use strands_core::types::streaming::{
    ContentBlockType, DeltaContent, StopReason, StreamEvent, Usage,
};

#[derive(Default)]
pub(crate) struct CodexCliState {
    saw_message_start: bool,
    text_block_open: bool,
    in_thinking: bool,
    text_block_index: usize,
    /// Per-item-id cursors so item.updated only emits the new tail.
    seen_text: HashMap<String, usize>,
}

#[derive(Deserialize)]
struct UsageRaw {
    #[serde(default)]
    input_tokens: Option<i64>,
    #[serde(default)]
    output_tokens: Option<i64>,
}

impl CodexCliState {
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
                tracing::warn!("codex-cli json decode failed: {e} line={line}");
                return Vec::new();
            }
        };
        let kind = value.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match kind {
            "thread.started" => Vec::new(),
            "turn.started" => {
                let mut out = Vec::new();
                self.ensure_message_start(&mut out);
                out
            }
            "item.started" | "item.updated" | "item.completed" => {
                self.handle_item_event(&value, kind)
            }
            "turn.completed" => {
                let mut out = Vec::new();
                self.close_text_block(&mut out);
                if let Some(usage) = value.get("usage") {
                    out.push(StreamEvent::Metadata {
                        usage: extract_usage(usage),
                    });
                }
                out.push(StreamEvent::MessageStop {
                    stop_reason: StopReason::EndTurn,
                });
                out
            }
            "turn.failed" => {
                let mut out = Vec::new();
                self.close_text_block(&mut out);
                let msg = value
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("turn failed");
                tracing::warn!("codex turn.failed: {msg}");
                out.push(StreamEvent::MessageStop {
                    stop_reason: StopReason::EndTurn,
                });
                out
            }
            "error" => {
                let msg = value
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("codex error");
                tracing::warn!("codex error event: {msg}");
                Vec::new()
            }
            other => {
                tracing::trace!("codex-cli unrecognised type: {other}");
                Vec::new()
            }
        }
    }

    /// Drain any open blocks at end-of-stream — guards against codex
    /// exiting mid-turn without a `turn.completed` event.
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

    fn handle_item_event(&mut self, value: &serde_json::Value, kind: &str) -> Vec<StreamEvent> {
        let item = match value.get("item") {
            Some(i) => i,
            None => return Vec::new(),
        };
        let id = match item.get("id").and_then(|i| i.as_str()) {
            Some(s) => s.to_string(),
            None => return Vec::new(),
        };
        let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");

        let mut out = Vec::new();
        match item_type {
            "agent_message" => {
                let text = item.get("text").and_then(|t| t.as_str()).unwrap_or("");
                self.emit_text_diff(&id, text, false, &mut out);
                if kind == "item.completed" {
                    self.seen_text.remove(&id);
                }
            }
            "reasoning" => {
                let text = item.get("text").and_then(|t| t.as_str()).unwrap_or("");
                self.emit_text_diff(&id, text, true, &mut out);
                if kind == "item.completed" {
                    // Close the synthetic <thinking> wrapper when the
                    // reasoning item is done so subsequent agent_message
                    // text isn't misclassified.
                    if self.in_thinking {
                        self.ensure_text_block_open(&mut out);
                        out.push(StreamEvent::ContentBlockDelta {
                            index: self.text_block_index,
                            delta: DeltaContent::TextDelta("</thinking>".into()),
                        });
                        self.in_thinking = false;
                    }
                    self.seen_text.remove(&id);
                }
            }
            _ => {
                // command_execution / file_change / mcp_tool_call / etc:
                // useful for richer UI, but the strands StreamEvent enum
                // doesn't model side-effect items. Ignoring them keeps
                // the assistant text channel clean.
            }
        }
        out
    }

    fn emit_text_diff(
        &mut self,
        id: &str,
        full_text: &str,
        is_thinking: bool,
        out: &mut Vec<StreamEvent>,
    ) {
        let prev = self.seen_text.get(id).copied().unwrap_or(0);
        // Codex *should* only ever extend; if it shrinks (e.g. a
        // re-render), reset and re-emit the whole text rather than
        // panicking on a slice out of bounds.
        let tail = if full_text.len() >= prev {
            &full_text[prev..]
        } else {
            full_text
        };
        if tail.is_empty() {
            self.seen_text.insert(id.to_string(), full_text.len());
            return;
        }

        self.ensure_text_block_open(out);
        if is_thinking && !self.in_thinking {
            self.in_thinking = true;
            out.push(StreamEvent::ContentBlockDelta {
                index: self.text_block_index,
                delta: DeltaContent::TextDelta("<thinking>".into()),
            });
        } else if !is_thinking && self.in_thinking {
            out.push(StreamEvent::ContentBlockDelta {
                index: self.text_block_index,
                delta: DeltaContent::TextDelta("</thinking>".into()),
            });
            self.in_thinking = false;
        }

        out.push(StreamEvent::ContentBlockDelta {
            index: self.text_block_index,
            delta: DeltaContent::TextDelta(tail.to_string()),
        });
        self.seen_text.insert(id.to_string(), full_text.len());
    }

    fn ensure_message_start(&mut self, out: &mut Vec<StreamEvent>) {
        if !self.saw_message_start {
            self.saw_message_start = true;
            out.push(StreamEvent::MessageStart {
                role: Role::Assistant,
            });
        }
    }

    fn ensure_text_block_open(&mut self, out: &mut Vec<StreamEvent>) {
        self.ensure_message_start(out);
        if !self.text_block_open {
            self.text_block_open = true;
            out.push(StreamEvent::ContentBlockStart {
                index: self.text_block_index,
                content_type: ContentBlockType::Text,
            });
        }
    }

    fn close_text_block(&mut self, out: &mut Vec<StreamEvent>) {
        if self.in_thinking {
            self.ensure_text_block_open(out);
            out.push(StreamEvent::ContentBlockDelta {
                index: self.text_block_index,
                delta: DeltaContent::TextDelta("</thinking>".into()),
            });
            self.in_thinking = false;
        }
        if self.text_block_open {
            out.push(StreamEvent::ContentBlockStop {
                index: self.text_block_index,
            });
            self.text_block_open = false;
        }
    }
}

fn extract_usage(usage: &serde_json::Value) -> Usage {
    let raw: UsageRaw = serde_json::from_value(usage.clone()).unwrap_or(UsageRaw {
        input_tokens: None,
        output_tokens: None,
    });
    Usage {
        input_tokens: raw.input_tokens.map(|n| n.max(0) as u64),
        output_tokens: raw.output_tokens.map(|n| n.max(0) as u64),
        total_duration_ns: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(lines: &[&str]) -> Vec<StreamEvent> {
        let mut state = CodexCliState::new();
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
    fn agent_message_streamed_via_item_updated_emits_diffs_only() {
        let events = run(&[
            r#"{"type":"thread.started","thread_id":"t"}"#,
            r#"{"type":"turn.started"}"#,
            r#"{"type":"item.started","item":{"id":"item_0","type":"agent_message","text":""}}"#,
            r#"{"type":"item.updated","item":{"id":"item_0","type":"agent_message","text":"Hi"}}"#,
            r#"{"type":"item.updated","item":{"id":"item_0","type":"agent_message","text":"Hi there"}}"#,
            r#"{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"Hi there!"}}"#,
            r#"{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":3}}"#,
        ]);
        assert_eq!(assembled(&events), "Hi there!");
        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::MessageStop { stop_reason: StopReason::EndTurn })));
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::Metadata { usage: Usage { input_tokens: Some(10), output_tokens: Some(3), .. } }
        )));
    }

    #[test]
    fn reasoning_item_wrapped_in_thinking_tags() {
        let events = run(&[
            r#"{"type":"turn.started"}"#,
            r#"{"type":"item.started","item":{"id":"r0","type":"reasoning","text":""}}"#,
            r#"{"type":"item.updated","item":{"id":"r0","type":"reasoning","text":"hmm"}}"#,
            r#"{"type":"item.completed","item":{"id":"r0","type":"reasoning","text":"hmm let me think"}}"#,
            r#"{"type":"item.started","item":{"id":"a0","type":"agent_message","text":""}}"#,
            r#"{"type":"item.completed","item":{"id":"a0","type":"agent_message","text":"answer"}}"#,
            r#"{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}"#,
        ]);
        let s = assembled(&events);
        assert!(s.starts_with("<thinking>"), "got: {s}");
        assert!(s.contains("</thinking>"));
        assert!(s.ends_with("answer"));
    }

    #[test]
    fn malformed_lines_are_dropped() {
        let events = run(&["not json", "{}", ""]);
        // No agent_message events, but flush may inject MessageStop —
        // assert it didn't blow up + produced no text deltas.
        assert!(assembled(&events).is_empty());
    }

    #[test]
    fn turn_failed_still_emits_message_stop() {
        let events = run(&[
            r#"{"type":"turn.started"}"#,
            r#"{"type":"item.completed","item":{"id":"a0","type":"agent_message","text":"partial"}}"#,
            r#"{"type":"turn.failed","error":{"message":"context length exceeded"}}"#,
        ]);
        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::MessageStop { .. })));
        assert_eq!(assembled(&events), "partial");
    }

    #[test]
    fn shrinking_text_does_not_panic() {
        // Defensive: codex shouldn't ever shrink, but if it does we
        // re-emit the whole new text rather than slicing OOB.
        let events = run(&[
            r#"{"type":"turn.started"}"#,
            r#"{"type":"item.updated","item":{"id":"a0","type":"agent_message","text":"hello world"}}"#,
            r#"{"type":"item.updated","item":{"id":"a0","type":"agent_message","text":"hi"}}"#,
            r#"{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}"#,
        ]);
        // Implementation-defined exact concat, but it must contain "hi"
        // and not panic.
        assert!(assembled(&events).contains("hi"));
    }
}
