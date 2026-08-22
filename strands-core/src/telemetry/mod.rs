//! Telemetry — metrics and spans for agent execution.
//!
//! Deliberately not bound to OpenTelemetry. The `tracing` crate is already the
//! Rust ecosystem's instrumentation seam, and an OTel exporter plugs into it
//! without this crate depending on the OTel stack — which would force a heavy
//! dependency tree on every user, including those exporting nothing.
//!
//! Ported from upstream `telemetry/`.

pub mod metrics;

pub use metrics::{AgentMetrics, CycleMetrics, MetricsCollector, ToolMetrics};

/// Span and attribute names, matching the OpenTelemetry GenAI semantic
/// conventions upstream emits.
///
/// Kept as constants so a `tracing` subscriber can map them onto OTel
/// attributes without guessing at names.
pub mod attributes {
    pub const AGENT_NAME: &str = "gen_ai.agent.name";
    pub const OPERATION_NAME: &str = "gen_ai.operation.name";
    pub const REQUEST_MODEL: &str = "gen_ai.request.model";
    pub const USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
    pub const USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
    pub const USAGE_CACHE_READ_TOKENS: &str = "gen_ai.usage.cache_read_input_tokens";
    pub const USAGE_CACHE_WRITE_TOKENS: &str = "gen_ai.usage.cache_write_input_tokens";
    pub const TOOL_NAME: &str = "gen_ai.tool.name";
    pub const TOOL_CALL_ARGUMENTS: &str = "gen_ai.tool.call.arguments";
    pub const TOOL_CALL_RESULT: &str = "gen_ai.tool.call.result";
    pub const RESPONSE_FINISH_REASON: &str = "gen_ai.response.finish_reason";
}

/// Whether span attributes carrying message or tool content should be redacted.
///
/// Tool arguments and results routinely carry secrets and personal data;
/// exporting them to a tracing backend is a decision the operator has to make
/// deliberately, so the default is to redact.
///
/// Matches upstream's `gen_ai_span_attributes_only` / span-redaction work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RedactionPolicy {
    /// Include tool call arguments in spans.
    pub include_tool_arguments: bool,
    /// Include tool results in spans.
    pub include_tool_results: bool,
    /// Include message content in spans.
    pub include_message_content: bool,
}

impl Default for RedactionPolicy {
    fn default() -> Self {
        Self {
            include_tool_arguments: false,
            include_tool_results: false,
            include_message_content: false,
        }
    }
}

impl RedactionPolicy {
    /// Redact everything. The default.
    pub fn redacted() -> Self {
        Self::default()
    }

    /// Include everything. Only for environments where the tracing backend is
    /// as trusted as the agent's own inputs.
    pub fn verbose() -> Self {
        Self {
            include_tool_arguments: true,
            include_tool_results: true,
            include_message_content: true,
        }
    }

    /// Apply the policy to a value destined for a span attribute.
    pub fn apply<'a>(&self, value: &'a str, included: bool) -> &'a str {
        if included {
            value
        } else {
            "[redacted]"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redaction_is_the_default() {
        // Tool arguments and results routinely carry secrets; exporting them
        // has to be a deliberate choice.
        let policy = RedactionPolicy::default();
        assert!(!policy.include_tool_arguments);
        assert!(!policy.include_tool_results);
        assert!(!policy.include_message_content);
    }

    #[test]
    fn verbose_includes_everything() {
        let policy = RedactionPolicy::verbose();
        assert!(policy.include_tool_arguments);
        assert!(policy.include_tool_results);
        assert!(policy.include_message_content);
    }

    #[test]
    fn apply_substitutes_a_marker_rather_than_dropping_the_field() {
        // An absent attribute is ambiguous; "[redacted]" says the field existed
        // and was withheld.
        let policy = RedactionPolicy::default();
        assert_eq!(policy.apply("secret", false), "[redacted]");
        assert_eq!(policy.apply("secret", true), "secret");
    }
}
