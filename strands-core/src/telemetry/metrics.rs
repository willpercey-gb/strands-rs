//! Metrics accumulated over an agent invocation.

use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::types::streaming::{Metrics, StopReason, Usage};

/// Per-tool execution counters.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolMetrics {
    /// How many times this tool ran.
    pub call_count: usize,
    /// How many of those calls returned an error.
    pub error_count: usize,
    /// Total time spent in this tool.
    pub total_duration: Duration,
}

impl ToolMetrics {
    fn record(&mut self, duration: Duration, is_error: bool) {
        self.call_count += 1;
        if is_error {
            self.error_count += 1;
        }
        self.total_duration += duration;
    }

    /// Mean duration per call, or `None` before any call.
    pub fn mean_duration(&self) -> Option<Duration> {
        (self.call_count > 0).then(|| self.total_duration / self.call_count as u32)
    }

    /// Fraction of calls that errored, or `None` before any call.
    ///
    /// `None` rather than `0.0`: a tool that has never run has no error rate,
    /// and reporting zero would make it look reliable.
    pub fn error_rate(&self) -> Option<f64> {
        (self.call_count > 0).then(|| self.error_count as f64 / self.call_count as f64)
    }
}

/// One cycle of the agent loop.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CycleMetrics {
    /// Tokens consumed by this cycle's model call.
    pub usage: Usage,
    /// Latency of this cycle's model call.
    pub metrics: Metrics,
    /// How many tools the model asked for in this cycle.
    pub tool_calls: usize,
}

/// Everything measured across one invocation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentMetrics {
    /// One entry per model call, in order.
    pub cycles: Vec<CycleMetrics>,
    /// Tokens summed across every cycle.
    pub total_usage: Usage,
    /// Latency summed across every cycle.
    pub total_metrics: Metrics,
    /// Per-tool counters, keyed by tool name.
    pub tools: HashMap<String, ToolMetrics>,
    /// Why the invocation ended, once it has.
    pub stop_reason: Option<StopReason>,
}

impl AgentMetrics {
    /// How many model calls were made.
    pub fn cycle_count(&self) -> usize {
        self.cycles.len()
    }

    /// Tool calls across every tool.
    pub fn total_tool_calls(&self) -> usize {
        self.tools.values().map(|t| t.call_count).sum()
    }

    /// Failed tool calls across every tool.
    pub fn total_tool_errors(&self) -> usize {
        self.tools.values().map(|t| t.error_count).sum()
    }

    /// Tools ordered by total time spent, slowest first.
    pub fn tools_by_duration(&self) -> Vec<(&str, &ToolMetrics)> {
        let mut tools: Vec<(&str, &ToolMetrics)> = self
            .tools
            .iter()
            .map(|(name, metrics)| (name.as_str(), metrics))
            .collect();
        tools.sort_by(|a, b| {
            b.1.total_duration
                .cmp(&a.1.total_duration)
                .then_with(|| a.0.cmp(b.0))
        });
        tools
    }
}

/// Accumulates metrics as an invocation proceeds.
#[derive(Debug, Default)]
pub struct MetricsCollector {
    metrics: AgentMetrics,
}

impl MetricsCollector {
    /// Create with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed model call.
    pub fn record_cycle(&mut self, usage: &Usage, metrics: &Metrics, tool_calls: usize) {
        self.metrics.total_usage.accumulate(usage);
        self.metrics.total_metrics.accumulate(metrics);
        self.metrics.cycles.push(CycleMetrics {
            usage: usage.clone(),
            metrics: metrics.clone(),
            tool_calls,
        });
    }

    /// Record a completed tool call.
    pub fn record_tool(&mut self, name: &str, duration: Duration, is_error: bool) {
        self.metrics
            .tools
            .entry(name.to_string())
            .or_default()
            .record(duration, is_error);
    }

    /// Record why the invocation ended.
    pub fn set_stop_reason(&mut self, reason: StopReason) {
        self.metrics.stop_reason = Some(reason);
    }

    /// Borrow the metrics collected so far.
    pub fn snapshot(&self) -> &AgentMetrics {
        &self.metrics
    }

    /// Consume the collector, returning the metrics.
    pub fn finish(self) -> AgentMetrics {
        self.metrics
    }
}

impl std::fmt::Display for AgentMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "{} cycle(s), {} tool call(s), {} error(s)",
            self.cycle_count(),
            self.total_tool_calls(),
            self.total_tool_errors()
        )?;

        if let Some(total) = self.total_usage.total() {
            write!(f, "  tokens: {total}")?;
            if let Some(cached) = self.total_usage.cache_read_input_tokens {
                write!(f, " ({cached} from cache)")?;
            }
            writeln!(f)?;
        }

        for (name, metrics) in self.tools_by_duration() {
            writeln!(
                f,
                "  {name}: {} call(s), {:.2}s",
                metrics.call_count,
                metrics.total_duration.as_secs_f64()
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(input: u64, output: u64) -> Usage {
        Usage {
            input_tokens: Some(input),
            output_tokens: Some(output),
            ..Default::default()
        }
    }

    #[test]
    fn cycles_accumulate_usage() {
        let mut c = MetricsCollector::new();
        c.record_cycle(&usage(10, 5), &Metrics::default(), 0);
        c.record_cycle(&usage(20, 8), &Metrics::default(), 1);

        let m = c.snapshot();
        assert_eq!(m.cycle_count(), 2);
        assert_eq!(m.total_usage.input_tokens, Some(30));
        assert_eq!(m.total_usage.output_tokens, Some(13));
    }

    #[test]
    fn cache_counters_survive_accumulation() {
        let mut c = MetricsCollector::new();
        c.record_cycle(
            &Usage {
                cache_read_input_tokens: Some(500),
                ..Default::default()
            },
            &Metrics::default(),
            0,
        );
        assert_eq!(c.snapshot().total_usage.cache_read_input_tokens, Some(500));
    }

    #[test]
    fn tool_metrics_count_calls_and_errors() {
        let mut c = MetricsCollector::new();
        c.record_tool("search", Duration::from_millis(100), false);
        c.record_tool("search", Duration::from_millis(300), true);

        let tool = &c.snapshot().tools["search"];
        assert_eq!(tool.call_count, 2);
        assert_eq!(tool.error_count, 1);
        assert_eq!(tool.total_duration, Duration::from_millis(400));
        assert_eq!(tool.mean_duration(), Some(Duration::from_millis(200)));
        assert_eq!(tool.error_rate(), Some(0.5));
    }

    #[test]
    fn an_unused_tool_has_no_error_rate_rather_than_zero() {
        // Reporting 0.0 would make a tool that never ran look reliable.
        let tool = ToolMetrics::default();
        assert_eq!(tool.error_rate(), None);
        assert_eq!(tool.mean_duration(), None);
    }

    #[test]
    fn tools_sort_slowest_first() {
        let mut c = MetricsCollector::new();
        c.record_tool("fast", Duration::from_millis(10), false);
        c.record_tool("slow", Duration::from_millis(900), false);
        c.record_tool("middle", Duration::from_millis(100), false);

        let names: Vec<&str> = c
            .snapshot()
            .tools_by_duration()
            .iter()
            .map(|(n, _)| *n)
            .collect();
        assert_eq!(names, vec!["slow", "middle", "fast"]);
    }

    #[test]
    fn totals_span_every_tool() {
        let mut c = MetricsCollector::new();
        c.record_tool("a", Duration::from_millis(1), false);
        c.record_tool("b", Duration::from_millis(1), true);
        c.record_tool("b", Duration::from_millis(1), true);

        let m = c.snapshot();
        assert_eq!(m.total_tool_calls(), 3);
        assert_eq!(m.total_tool_errors(), 2);
    }

    #[test]
    fn display_summarizes_the_run() {
        let mut c = MetricsCollector::new();
        c.record_cycle(&usage(10, 5), &Metrics::default(), 1);
        c.record_tool("search", Duration::from_millis(500), false);

        let text = c.snapshot().to_string();
        assert!(text.contains("1 cycle"));
        assert!(text.contains("tokens: 15"));
        assert!(text.contains("search"));
    }

    #[test]
    fn metrics_round_trip_through_serde() {
        let mut c = MetricsCollector::new();
        c.record_cycle(&usage(1, 2), &Metrics::default(), 0);
        c.record_tool("t", Duration::from_millis(5), false);
        c.set_stop_reason(StopReason::EndTurn);

        let json = serde_json::to_string(c.snapshot()).unwrap();
        let back: AgentMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(back.cycle_count(), 1);
        assert_eq!(back.total_tool_calls(), 1);
        assert_eq!(back.stop_reason, Some(StopReason::EndTurn));
    }
}
