use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::agent::AgentResult;
use crate::types::streaming::Usage;

/// Status of a multi-agent execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MultiAgentStatus {
    Completed,
    Failed,
    Cancelled,
    MaxStepsReached,
    TimedOut,
}

/// Status of an individual node within a multi-agent execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    Pending,
    Executing,
    Completed,
    Failed,
    Cancelled,
}

/// Result from a single node (agent) execution.
#[derive(Debug, Clone)]
pub struct NodeResult {
    pub node_id: String,
    pub status: NodeStatus,
    pub result: Option<AgentResult>,
    pub error: Option<String>,
    pub execution_time: Duration,
}

/// Result from a multi-agent orchestration (Swarm or Graph).
#[derive(Debug)]
pub struct MultiAgentResult {
    /// Overall execution status.
    pub status: MultiAgentStatus,
    /// Results from each node, keyed by node ID.
    pub results: HashMap<String, NodeResult>,
    /// Order in which nodes were executed.
    pub execution_order: Vec<String>,
    /// Total number of node executions.
    pub execution_count: usize,
    /// Total execution time.
    pub execution_time: Duration,
    /// Accumulated token usage across all nodes.
    pub accumulated_usage: Usage,
    /// Final text output from the last node.
    pub output: String,
}

impl NodeResult {
    /// The node's text output, if it produced one.
    pub fn text(&self) -> Option<String> {
        self.result.as_ref().map(|r| r.text())
    }

    pub fn is_success(&self) -> bool {
        self.status == NodeStatus::Completed
    }
}

impl std::fmt::Display for NodeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            NodeStatus::Pending => "pending",
            NodeStatus::Executing => "executing",
            NodeStatus::Completed => "completed",
            NodeStatus::Failed => "failed",
            NodeStatus::Cancelled => "cancelled",
        };
        f.write_str(s)
    }
}

impl std::fmt::Display for MultiAgentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            MultiAgentStatus::Completed => "completed",
            MultiAgentStatus::Failed => "failed",
            MultiAgentStatus::Cancelled => "cancelled",
            MultiAgentStatus::MaxStepsReached => "max_steps_reached",
            MultiAgentStatus::TimedOut => "timed_out",
        };
        f.write_str(s)
    }
}

impl std::fmt::Display for NodeResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} [{}] ({:.2}s)",
            self.node_id,
            self.status,
            self.execution_time.as_secs_f64()
        )?;
        if let Some(error) = &self.error {
            write!(f, ": {error}")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for MultiAgentResult {
    /// A short execution summary, in execution order.
    ///
    /// Node lines follow `execution_order` rather than the map's iteration
    /// order — a summary that reshuffles the run every time it is printed is
    /// useless for diagnosing one.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "MultiAgentResult [{}] {} node execution(s) in {:.2}s",
            self.status,
            self.execution_count,
            self.execution_time.as_secs_f64()
        )?;

        for node_id in &self.execution_order {
            if let Some(result) = self.results.get(node_id) {
                writeln!(f, "  - {result}")?;
            }
        }

        if let Some(total) = self.accumulated_usage.total() {
            writeln!(f, "  tokens: {total}")?;
        }

        if !self.output.is_empty() {
            write!(f, "  output: {}", self.output)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::message::Message;
    use crate::types::streaming::{Metrics, StopReason};

    fn node(id: &str, status: NodeStatus, error: Option<&str>) -> NodeResult {
        NodeResult {
            node_id: id.to_string(),
            status,
            result: Some(AgentResult {
                stop_reason: StopReason::EndTurn,
                message: Message::assistant(vec![]),
                usage: Usage::default(),
                metrics: Metrics::default(),
                cycle_count: 1,
                interrupts: Vec::new(),
            }),
            error: error.map(String::from),
            execution_time: Duration::from_millis(1500),
        }
    }

    fn result(order: Vec<&str>) -> MultiAgentResult {
        let mut results = HashMap::new();
        for id in &order {
            results.insert(id.to_string(), node(id, NodeStatus::Completed, None));
        }
        MultiAgentResult {
            status: MultiAgentStatus::Completed,
            results,
            execution_order: order.iter().map(|s| s.to_string()).collect(),
            execution_count: order.len(),
            execution_time: Duration::from_secs(3),
            accumulated_usage: Usage {
                input_tokens: Some(10),
                output_tokens: Some(5),
                ..Default::default()
            },
            output: "done".to_string(),
        }
    }

    #[test]
    fn node_display_includes_id_status_and_time() {
        let text = node("worker", NodeStatus::Completed, None).to_string();
        assert!(text.contains("worker"));
        assert!(text.contains("completed"));
        assert!(text.contains("1.50s"), "got {text}");
    }

    #[test]
    fn node_display_surfaces_the_error() {
        let text = node("worker", NodeStatus::Failed, Some("boom")).to_string();
        assert!(text.contains("failed"));
        assert!(text.contains("boom"));
    }

    #[test]
    fn result_display_follows_execution_order() {
        // The map is unordered; the summary must not be.
        let text = result(vec!["first", "second", "third"]).to_string();
        let pos = |s: &str| text.find(s).unwrap_or(usize::MAX);
        assert!(pos("first") < pos("second"));
        assert!(pos("second") < pos("third"));
    }

    #[test]
    fn result_display_reports_status_counts_and_tokens() {
        let text = result(vec!["a"]).to_string();
        assert!(text.contains("completed"));
        assert!(text.contains("1 node execution"));
        assert!(text.contains("tokens: 15"));
        assert!(text.contains("done"));
    }

    #[test]
    fn node_helpers_expose_success_and_text() {
        assert!(node("a", NodeStatus::Completed, None).is_success());
        assert!(!node("a", NodeStatus::Failed, None).is_success());
        assert_eq!(node("a", NodeStatus::Completed, None).text(), Some(String::new()));
    }
}
