//! Translator from `claude --output-format stream-json` line events into
//! `strands_core::StreamEvent`. The CLI emits one JSON object per line,
//! mirroring the Anthropic Messages API SSE format under a
//! `stream_event` envelope, plus `system` / `result` framing.
//!
//! Mapping rules:
//! - `stream_event.event.type:content_block_delta` with `delta.type:text_delta`
//!   → `ContentBlockDelta { delta: TextDelta }` on the text block.
//! - `stream_event.event.type:content_block_delta` with `delta.type:thinking_delta`
//!   → emitted as a TextDelta with `<thinking>...</thinking>` wrappers so the
//!   harness pipeline's xml_unwrap routes it to ReasoningDelta naturally.
//! - `stream_event.event.type:message_stop` → `MessageStop { EndTurn }`.
//! - `result` (terminal) carries final usage.
//! - All other events (`system`, `rate_limit_event`, `assistant` echoes,
//!   `signature_delta`, `content_block_start`, `content_block_stop`)
//!   are no-ops.

use serde::Deserialize;
use strands_core::types::message::Role;
use strands_core::types::streaming::{
    ContentBlockType, DeltaContent, StopReason, StreamEvent, Usage,
};

#[derive(Default)]
pub(crate) struct ClaudeCliState {
    saw_message_start: bool,
    text_block_open: bool,
    /// Whether we have an open synthetic text block representing thinking
    /// content (so xml_unwrap can absorb it on the harness side).
    in_thinking: bool,
    text_block_index: usize,
}

impl ClaudeCliState {
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
                tracing::warn!("claude-cli json decode failed: {e} line={line}");
                return Vec::new();
            }
        };
        let kind = value.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match kind {
            "stream_event" => {
                let inner = match value.get("event") {
                    Some(e) => e,
                    None => return Vec::new(),
                };
                self.handle_stream_event(inner)
            }
            "result" => {
                let usage_obj = value.get("usage");
                let mut events = Vec::new();
                self.close_text_block(&mut events);
                if let Some(usage) = usage_obj {
                    events.push(StreamEvent::Metadata {
                        usage: extract_usage(usage),
                    });
                }
                events
            }
            // Quiet events we deliberately ignore.
            "system" | "rate_limit_event" | "assistant" | "user" => Vec::new(),
            other => {
                tracing::trace!("claude-cli unrecognised type: {other}");
                Vec::new()
            }
        }
    }

    fn handle_stream_event(&mut self, event: &serde_json::Value) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        let kind = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match kind {
            "message_start" => {
                if !self.saw_message_start {
                    self.saw_message_start = true;
                    out.push(StreamEvent::MessageStart {
                        role: Role::Assistant,
                    });
                }
            }
            "content_block_start" => {
                // No-op: we discriminate text vs thinking entirely from
                // the delta's `type` field below, and lazily open the
                // strands text block on the first delta so we don't
                // emit empty blocks for signature-only blocks.
            }
            "content_block_delta" => {
                let delta = match event.get("delta") {
                    Some(d) => d,
                    None => return out,
                };
                let dtype = delta.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match dtype {
                    "text_delta" => {
                        if let Some(t) = delta.get("text").and_then(|t| t.as_str()) {
                            self.ensure_text_block_open(&mut out);
                            // If we were in a thinking block, the harness
                            // pipeline will already have seen the open tag;
                            // close it before emitting plain text.
                            if self.in_thinking {
                                out.push(StreamEvent::ContentBlockDelta {
                                    index: self.text_block_index,
                                    delta: DeltaContent::TextDelta("</thinking>".into()),
                                });
                                self.in_thinking = false;
                            }
                            out.push(StreamEvent::ContentBlockDelta {
                                index: self.text_block_index,
                                delta: DeltaContent::TextDelta(t.to_string()),
                            });
                        }
                    }
                    "thinking_delta" => {
                        if let Some(t) = delta.get("thinking").and_then(|t| t.as_str()) {
                            self.ensure_text_block_open(&mut out);
                            // First fragment of a thinking run: emit the
                            // opening tag so xml_unwrap on the consumer
                            // side starts collecting reasoning.
                            if !self.in_thinking {
                                self.in_thinking = true;
                                out.push(StreamEvent::ContentBlockDelta {
                                    index: self.text_block_index,
                                    delta: DeltaContent::TextDelta("<thinking>".into()),
                                });
                            }
                            out.push(StreamEvent::ContentBlockDelta {
                                index: self.text_block_index,
                                delta: DeltaContent::TextDelta(t.to_string()),
                            });
                        }
                    }
                    "signature_delta" | "input_json_delta" => {
                        // Reasoning signature & tool-input deltas — ignored
                        // for the one-shot CLI wrapper.
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                // If we had an open thinking block and never received
                // the implicit close, emit the close-tag now so the
                // unwrapper doesn't keep buffering downstream.
                if self.in_thinking {
                    out.push(StreamEvent::ContentBlockDelta {
                        index: self.text_block_index,
                        delta: DeltaContent::TextDelta("</thinking>".into()),
                    });
                    self.in_thinking = false;
                }
            }
            "message_delta" => {
                // Carries final stop_reason; emit MessageStop.
                let reason = event
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(|s| s.as_str())
                    .unwrap_or("end_turn");
                self.close_text_block(&mut out);
                out.push(StreamEvent::MessageStop {
                    stop_reason: map_stop_reason(reason),
                });
            }
            "message_stop" => {
                // Already handled via message_delta in most flows; this
                // is a safety net for cases where the CLI omits delta.
                if self.text_block_open {
                    self.close_text_block(&mut out);
                    out.push(StreamEvent::MessageStop {
                        stop_reason: StopReason::EndTurn,
                    });
                }
            }
            _ => {}
        }
        out
    }

    fn ensure_text_block_open(&mut self, out: &mut Vec<StreamEvent>) {
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

fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" | "stop_sequence" => StopReason::EndTurn,
        "max_tokens" => StopReason::MaxTokens,
        "tool_use" => StopReason::ToolUse,
        _ => StopReason::EndTurn,
    }
}

fn extract_usage(usage: &serde_json::Value) -> Usage {
    #[derive(Deserialize)]
    struct Raw {
        #[serde(default)]
        input_tokens: Option<u64>,
        #[serde(default)]
        output_tokens: Option<u64>,
    }
    let raw: Raw = serde_json::from_value(usage.clone()).unwrap_or(Raw {
        input_tokens: None,
        output_tokens: None,
    });
    Usage {
        input_tokens: raw.input_tokens,
        output_tokens: raw.output_tokens,
        total_duration_ns: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(lines: &[&str]) -> Vec<StreamEvent> {
        let mut state = ClaudeCliState::new();
        let mut events = Vec::new();
        for l in lines {
            events.extend(state.ingest_line(l));
        }
        events
    }

    #[test]
    fn text_delta_emits_through() {
        let events = run(&[
            r#"{"type":"system","subtype":"init","session_id":"x"}"#,
            r#"{"type":"stream_event","event":{"type":"message_start"}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"text"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_stop","index":0}}"#,
            r#"{"type":"stream_event","event":{"type":"message_delta","delta":{"stop_reason":"end_turn"}}}"#,
            r#"{"type":"stream_event","event":{"type":"message_stop"}}"#,
        ]);
        let assembled: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentBlockDelta {
                    delta: DeltaContent::TextDelta(t),
                    ..
                } => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(assembled, "hi there");
        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::MessageStop { stop_reason: StopReason::EndTurn })));
    }

    #[test]
    fn thinking_delta_wrapped_in_xml_for_pipeline() {
        let events = run(&[
            r#"{"type":"stream_event","event":{"type":"message_start"}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" let me think"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_stop","index":0}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_start","index":1,"content_block":{"type":"text"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}}"#,
            r#"{"type":"stream_event","event":{"type":"message_delta","delta":{"stop_reason":"end_turn"}}}"#,
        ]);
        let assembled: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentBlockDelta {
                    delta: DeltaContent::TextDelta(t),
                    ..
                } => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        // Thinking content should arrive wrapped in <thinking>...</thinking>
        // so the harness pipeline routes it to ReasoningDelta.
        assert!(assembled.starts_with("<thinking>"));
        assert!(assembled.contains("</thinking>"));
        assert!(assembled.ends_with("answer"));
    }

    #[test]
    fn result_event_carries_usage() {
        let events = run(&[
            r#"{"type":"stream_event","event":{"type":"message_start"}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"text"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}}"#,
            r#"{"type":"stream_event","event":{"type":"message_delta","delta":{"stop_reason":"end_turn"}}}"#,
            r#"{"type":"result","subtype":"success","usage":{"input_tokens":42,"output_tokens":13}}"#,
        ]);
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::Metadata {
                usage: Usage { input_tokens: Some(42), output_tokens: Some(13), .. }
            }
        )));
    }

    #[test]
    fn malformed_lines_are_dropped() {
        let events = run(&["not json", "{}", ""]);
        // No panic, no spurious events.
        assert!(events.is_empty());
    }
}
