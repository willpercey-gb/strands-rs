use serde::{Deserialize, Serialize};

use super::message::Role;

/// Events emitted during streaming model responses.
/// Unified protocol between model adapters and the agent loop.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// The model began a response.
    MessageStart {
        /// Role of the speaker, always `Assistant` in practice.
        role: Role,
    },
    /// A new content block opened.
    ContentBlockStart {
        /// Position of the block within the message.
        index: usize,
        /// What kind of block this is.
        content_type: ContentBlockType,
    },
    /// An incremental update to an open block.
    ContentBlockDelta {
        /// Position of the block being updated.
        index: usize,
        /// The increment.
        delta: DeltaContent,
    },
    /// A content block closed.
    ContentBlockStop {
        /// Position of the block that closed.
        index: usize,
    },
    /// The response finished.
    MessageStop {
        /// Why the model stopped.
        stop_reason: StopReason,
    },
    /// Usage and performance figures for the call.
    Metadata {
        /// Token counts.
        usage: Usage,
        /// Latency figures.
        metrics: Metrics,
    },
}

#[derive(Debug, Clone)]
/// What kind of block a stream just opened.
pub enum ContentBlockType {
    /// A run of text.
    Text,
    /// A tool call.
    ToolUse {
        /// Identifier the matching result must echo back.
        tool_use_id: String,
        /// Name of the tool being called.
        name: String,
        /// Signature tying the model's reasoning to this tool call. Providers
        /// that emit one reject the call if it is not echoed back.
        reasoning_signature: Option<String>,
    },
    /// Reasoning the model is working through.
    Reasoning,
}

#[derive(Debug, Clone)]
/// An incremental update to an open content block.
pub enum DeltaContent {
    /// More response text.
    TextDelta(String),
    /// More of a tool call's JSON arguments. Fragments concatenate.
    ToolInputDelta(String),
    /// Incremental reasoning text.
    ReasoningDelta(String),
    /// A reasoning signature, delivered as its own delta by some providers
    /// (notably Gemini thought signatures).
    ReasoningSignature(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Why the model, or the agent loop, stopped.
pub enum StopReason {
    /// Normal completion of the response.
    EndTurn,
    /// Model requested a tool.
    ToolUse,
    /// The provider's per-call output cap was reached.
    MaxTokens,
    /// A stop sequence was encountered.
    StopSequence,
    /// Execution was cancelled by the caller.
    Cancelled,
    /// Content was filtered due to a policy violation.
    ContentFiltered,
    /// A guardrail intervened.
    GuardrailIntervention,
    /// The agent paused for human input.
    Interrupt,
    /// The agent paused for durable checkpoint persistence.
    Checkpoint,
    /// The configured turn limit was reached.
    LimitTurns,
    /// The configured output-token limit was reached.
    LimitOutputTokens,
    /// The configured total-token limit was reached.
    LimitTotalTokens,
}

impl StopReason {
    /// Whether this reason ends the agent loop rather than continuing it.
    pub fn is_terminal(&self) -> bool {
        !matches!(self, StopReason::ToolUse)
    }
}

/// Token usage for a model interaction.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    /// Tokens sent in the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// Tokens the model generated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Total tokens (input + output).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    /// Tokens served from the provider's prompt cache.
    ///
    /// Billed at a large discount, so tracking this separately is what makes
    /// cache-point placement measurable rather than guesswork.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
    /// Tokens written into the provider's prompt cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_input_tokens: Option<u64>,
}

impl Usage {
    /// Add another usage record into this one, field-wise.
    ///
    /// `None + Some(n) == Some(n)`: a provider that reports only some fields
    /// should not zero out what another cycle did report.
    pub fn accumulate(&mut self, other: &Usage) {
        fn add(a: &mut Option<u64>, b: Option<u64>) {
            if let Some(b) = b {
                *a = Some(a.unwrap_or(0) + b);
            }
        }
        add(&mut self.input_tokens, other.input_tokens);
        add(&mut self.output_tokens, other.output_tokens);
        add(&mut self.total_tokens, other.total_tokens);
        add(
            &mut self.cache_read_input_tokens,
            other.cache_read_input_tokens,
        );
        add(
            &mut self.cache_write_input_tokens,
            other.cache_write_input_tokens,
        );
    }

    /// Total tokens, falling back to input + output when the provider does not
    /// report a total directly.
    pub fn total(&self) -> Option<u64> {
        self.total_tokens
            .or_else(|| match (self.input_tokens, self.output_tokens) {
                (None, None) => None,
                (i, o) => Some(i.unwrap_or(0) + o.unwrap_or(0)),
            })
    }
}

/// Performance metrics for a model interaction.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Metrics {
    /// End-to-end latency of the model request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
    /// Latency from request to the first content chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_to_first_byte_ms: Option<u64>,
}

impl Metrics {
    /// Accumulate latency across cycles, keeping the earliest time-to-first-byte.
    pub fn accumulate(&mut self, other: &Metrics) {
        if let Some(l) = other.latency_ms {
            self.latency_ms = Some(self.latency_ms.unwrap_or(0) + l);
        }
        if let Some(ttfb) = other.time_to_first_byte_ms {
            self.time_to_first_byte_ms =
                Some(self.time_to_first_byte_ms.map_or(ttfb, |e| e.min(ttfb)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulate_sums_reported_fields() {
        let mut a = Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
            ..Default::default()
        };
        a.accumulate(&Usage {
            input_tokens: Some(3),
            output_tokens: Some(2),
            cache_read_input_tokens: Some(100),
            ..Default::default()
        });

        assert_eq!(a.input_tokens, Some(13));
        assert_eq!(a.output_tokens, Some(7));
        assert_eq!(a.cache_read_input_tokens, Some(100));
    }

    #[test]
    fn accumulate_does_not_invent_zeros_for_unreported_fields() {
        let mut a = Usage::default();
        a.accumulate(&Usage {
            input_tokens: Some(4),
            ..Default::default()
        });
        assert_eq!(a.input_tokens, Some(4));
        assert_eq!(
            a.output_tokens, None,
            "a field neither side reported must stay unreported"
        );
    }

    #[test]
    fn total_falls_back_to_input_plus_output() {
        let u = Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
            ..Default::default()
        };
        assert_eq!(u.total(), Some(15));

        let u = Usage {
            total_tokens: Some(99),
            input_tokens: Some(10),
            ..Default::default()
        };
        assert_eq!(u.total(), Some(99), "an explicit total wins");

        assert_eq!(Usage::default().total(), None);
    }

    #[test]
    fn metrics_keep_the_earliest_time_to_first_byte() {
        let mut m = Metrics {
            latency_ms: Some(100),
            time_to_first_byte_ms: Some(50),
        };
        m.accumulate(&Metrics {
            latency_ms: Some(200),
            time_to_first_byte_ms: Some(30),
        });
        assert_eq!(m.latency_ms, Some(300), "latency accumulates");
        assert_eq!(m.time_to_first_byte_ms, Some(30), "TTFB is a minimum");
    }

    #[test]
    fn only_tool_use_continues_the_loop() {
        assert!(!StopReason::ToolUse.is_terminal());
        for r in [
            StopReason::EndTurn,
            StopReason::MaxTokens,
            StopReason::Interrupt,
            StopReason::LimitTurns,
        ] {
            assert!(r.is_terminal(), "{r:?} should end the loop");
        }
    }
}
