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
