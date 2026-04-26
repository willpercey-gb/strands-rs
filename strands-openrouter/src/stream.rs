//! Line-buffered SSE → strands `StreamEvent` adapter.

use std::collections::HashMap;

use strands_core::types::streaming::{
    ContentBlockType, DeltaContent, StopReason, StreamEvent, Usage,
};

use crate::types::{StreamChunk, StreamDelta};

/// Stateful translator that consumes SSE `data:` lines (one OpenAI chunk
/// each) and yields the strands `StreamEvent` sequence the agent loop
/// expects.
#[derive(Default)]
pub(crate) struct OpenAiStreamState {
    text_block_open: bool,
    text_block_index: usize,
    tool_block_index_for_oai_index: HashMap<usize, usize>,
    next_block_index: usize,
    saw_message_start: bool,
}

impl OpenAiStreamState {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Feed a complete chunk; returns the events to emit. Caller is
    /// responsible for ordering, but this method emits in the order the
    /// agent loop expects: MessageStart → ContentBlockStart →
    /// ContentBlockDelta(s) → ContentBlockStop → MessageStop / Metadata.
    pub(crate) fn ingest(&mut self, chunk: StreamChunk) -> Vec<StreamEvent> {
        let mut out = Vec::new();

        for choice in chunk.choices {
            if !self.saw_message_start {
                self.saw_message_start = true;
                out.push(StreamEvent::MessageStart {
                    role: strands_core::types::message::Role::Assistant,
                });
            }
            self.handle_delta(choice.delta, &mut out);
            if let Some(reason) = choice.finish_reason {
                if self.text_block_open {
                    out.push(StreamEvent::ContentBlockStop {
                        index: self.text_block_index,
                    });
                    self.text_block_open = false;
                }
                // Close any open tool blocks too.
                let mut indices: Vec<usize> = self
                    .tool_block_index_for_oai_index
                    .values()
                    .copied()
                    .collect();
                indices.sort_unstable();
                for idx in indices {
                    out.push(StreamEvent::ContentBlockStop { index: idx });
                }
                self.tool_block_index_for_oai_index.clear();
                out.push(StreamEvent::MessageStop {
                    stop_reason: map_stop_reason(&reason),
                });
            }
        }

        if let Some(u) = chunk.usage {
            out.push(StreamEvent::Metadata {
                usage: Usage {
                    input_tokens: u.prompt_tokens,
                    output_tokens: u.completion_tokens,
                    total_duration_ns: None,
                },
            });
        }

        out
    }

    fn handle_delta(&mut self, delta: StreamDelta, out: &mut Vec<StreamEvent>) {
        if let Some(content) = delta.content {
            if !content.is_empty() {
                if !self.text_block_open {
                    let idx = self.next_block_index;
                    self.next_block_index += 1;
                    self.text_block_index = idx;
                    self.text_block_open = true;
                    out.push(StreamEvent::ContentBlockStart {
                        index: idx,
                        content_type: ContentBlockType::Text,
                    });
                }
                out.push(StreamEvent::ContentBlockDelta {
                    index: self.text_block_index,
                    delta: DeltaContent::TextDelta(content),
                });
            }
        }

        for tc in delta.tool_calls {
            let block_idx = match self.tool_block_index_for_oai_index.get(&tc.index) {
                Some(&i) => i,
                None => {
                    let idx = self.next_block_index;
                    self.next_block_index += 1;
                    self.tool_block_index_for_oai_index.insert(tc.index, idx);
                    let id = tc.id.clone().unwrap_or_else(|| format!("tc_{}", idx));
                    let name = tc
                        .function
                        .as_ref()
                        .and_then(|f| f.name.clone())
                        .unwrap_or_default();
                    out.push(StreamEvent::ContentBlockStart {
                        index: idx,
                        content_type: ContentBlockType::ToolUse {
                            tool_use_id: id,
                            name,
                        },
                    });
                    idx
                }
            };
            if let Some(args) = tc.function.and_then(|f| f.arguments) {
                if !args.is_empty() {
                    out.push(StreamEvent::ContentBlockDelta {
                        index: block_idx,
                        delta: DeltaContent::ToolInputDelta(args),
                    });
                }
            }
        }
    }
}

fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::EndTurn,
        "length" => StopReason::MaxTokens,
        "tool_calls" | "function_call" => StopReason::ToolUse,
        "content_filter" => StopReason::ContentFiltered,
        _ => StopReason::EndTurn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn run(chunks: Vec<StreamChunk>) -> Vec<StreamEvent> {
        let mut state = OpenAiStreamState::new();
        chunks.into_iter().flat_map(|c| state.ingest(c)).collect()
    }

    fn td_only(events: &[StreamEvent]) -> String {
        let mut s = String::new();
        for e in events {
            if let StreamEvent::ContentBlockDelta {
                delta: DeltaContent::TextDelta(t),
                ..
            } = e
            {
                s.push_str(t);
            }
        }
        s
    }

    #[test]
    fn assembles_basic_text() {
        let events = run(vec![
            StreamChunk {
                choices: vec![StreamChoice {
                    index: 0,
                    delta: StreamDelta {
                        content: Some("hel".into()),
                        ..Default::default()
                    },
                    finish_reason: None,
                }],
                usage: None,
            },
            StreamChunk {
                choices: vec![StreamChoice {
                    index: 0,
                    delta: StreamDelta {
                        content: Some("lo".into()),
                        ..Default::default()
                    },
                    finish_reason: None,
                }],
                usage: None,
            },
            StreamChunk {
                choices: vec![StreamChoice {
                    index: 0,
                    delta: StreamDelta::default(),
                    finish_reason: Some("stop".into()),
                }],
                usage: Some(StreamUsage {
                    prompt_tokens: Some(7),
                    completion_tokens: Some(2),
                }),
            },
        ]);
        assert_eq!(td_only(&events), "hello");
        assert!(matches!(events.last(), Some(StreamEvent::Metadata { .. })));
    }

    #[test]
    fn maps_finish_reasons() {
        assert!(matches!(map_stop_reason("stop"), StopReason::EndTurn));
        assert!(matches!(map_stop_reason("length"), StopReason::MaxTokens));
        assert!(matches!(map_stop_reason("tool_calls"), StopReason::ToolUse));
        assert!(matches!(
            map_stop_reason("content_filter"),
            StopReason::ContentFiltered
        ));
    }

    #[test]
    fn tool_call_assembly() {
        let chunk_open = StreamChunk {
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta {
                    tool_calls: vec![StreamDeltaToolCall {
                        index: 0,
                        id: Some("call_1".into()),
                        function: Some(StreamDeltaToolFn {
                            name: Some("get_weather".into()),
                            arguments: Some("{\"city".into()),
                        }),
                    }],
                    ..Default::default()
                },
                finish_reason: None,
            }],
            usage: None,
        };
        let chunk_args = StreamChunk {
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta {
                    tool_calls: vec![StreamDeltaToolCall {
                        index: 0,
                        id: None,
                        function: Some(StreamDeltaToolFn {
                            name: None,
                            arguments: Some("\":\"London\"}".into()),
                        }),
                    }],
                    ..Default::default()
                },
                finish_reason: None,
            }],
            usage: None,
        };
        let chunk_finish = StreamChunk {
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta::default(),
                finish_reason: Some("tool_calls".into()),
            }],
            usage: None,
        };
        let events = run(vec![chunk_open, chunk_args, chunk_finish]);

        let mut input = String::new();
        for e in &events {
            if let StreamEvent::ContentBlockDelta {
                delta: DeltaContent::ToolInputDelta(s),
                ..
            } = e
            {
                input.push_str(s);
            }
        }
        assert_eq!(input, "{\"city\":\"London\"}");
        assert!(matches!(
            events.iter().find(|e| matches!(e, StreamEvent::MessageStop { .. })),
            Some(StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse
            })
        ));
    }
}
