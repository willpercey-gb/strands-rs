use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use tracing::{debug, warn};

use crate::agent::Agent;
use crate::error::StrandsError;
use crate::types::streaming::Usage;

use super::result::*;

/// State passed to edge condition functions.
#[derive(Debug)]
pub struct GraphState {
    /// Results from completed nodes.
    pub results: HashMap<String, NodeResult>,
    /// The original task input.
    pub task: String,
}

/// A condition function that determines whether an edge should be traversed.
pub type EdgeCondition = Arc<dyn Fn(&GraphState) -> bool + Send + Sync>;

/// A directed edge in the graph.
pub struct GraphEdge {
    pub from_node: String,
    pub to_node: String,
    pub condition: Option<EdgeCondition>,
}

impl std::fmt::Debug for GraphEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphEdge")
            .field("from_node", &self.from_node)
            .field("to_node", &self.to_node)
            .field("has_condition", &self.condition.is_some())
            .finish()
    }
}

/// Configuration for graph execution.
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Maximum total node executions (important for cyclic graphs).
    pub max_node_executions: usize,
    /// Total execution timeout.
    pub execution_timeout: Duration,
    /// Per-node execution timeout.
    pub node_timeout: Duration,
    /// Whether to reset agent state when revisiting a node.
    pub reset_on_revisit: bool,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_node_executions: 50,
            execution_timeout: Duration::from_secs(900),
            node_timeout: Duration::from_secs(300),
            reset_on_revisit: false,
        }
    }
}

/// A node in the graph.
struct GraphNode {
    agent: Arc<Mutex<Agent>>,
}

/// A deterministic directed graph agent orchestration.
///
/// Agents are placed at nodes and connected by edges. Output from one node
/// propagates as input to connected nodes. Edges may have conditions that
/// control traversal.
///
/// # Example
///
/// ```ignore
/// let graph = Graph::builder()
///     .node("researcher", researcher_agent)
///     .node("writer", writer_agent)
///     .node("reviewer", reviewer_agent)
///     .edge("researcher", "writer")
///     .edge("writer", "reviewer")
///     .entry_point("researcher")
///     .build()?;
///
/// let result = graph.run("Write an article about Rust").await?;
/// ```
pub struct Graph {
    nodes: HashMap<String, GraphNode>,
    edges: Vec<GraphEdge>,
    entry_points: Vec<String>,
    config: GraphConfig,
}

impl Graph {
    pub fn builder() -> GraphBuilder {
        GraphBuilder::new()
    }

    /// Run the graph with the given task.
    pub async fn run(&self, task: &str) -> Result<MultiAgentResult, StrandsError> {
        let start = Instant::now();
        let mut results: HashMap<String, NodeResult> = HashMap::new();
        let mut execution_order: Vec<String> = Vec::new();
        let mut accumulated_usage = Usage::default();
        let mut execution_count = 0;
        let mut output = String::new();

        // Build adjacency list
        let mut adjacency: HashMap<String, Vec<&GraphEdge>> = HashMap::new();
        for edge in &self.edges {
            adjacency
                .entry(edge.from_node.clone())
                .or_default()
                .push(edge);
        }

        // Track which nodes are ready to execute (all dependencies satisfied)
        let mut pending: VecDeque<(String, String)> = VecDeque::new();
        for entry in &self.entry_points {
            pending.push_back((entry.clone(), task.to_string()));
        }

        let mut completed: HashSet<String> = HashSet::new();

        while let Some((node_id, input)) = pending.pop_front() {
            // Check limits
            if execution_count >= self.config.max_node_executions {
                return Ok(MultiAgentResult {
                    status: MultiAgentStatus::MaxStepsReached,
                    results,
                    execution_order,
                    execution_count,
                    execution_time: start.elapsed(),
                    accumulated_usage,
                    output,
                });
            }

            if start.elapsed() > self.config.execution_timeout {
                return Ok(MultiAgentResult {
                    status: MultiAgentStatus::TimedOut,
                    results,
                    execution_order,
                    execution_count,
                    execution_time: start.elapsed(),
                    accumulated_usage,
                    output,
                });
            }

            let node = self.nodes.get(&node_id).ok_or_else(|| {
                StrandsError::Other(format!("Node '{}' not found in graph", node_id))
            })?;

            // Reset agent if revisiting
            if self.config.reset_on_revisit && completed.contains(&node_id) {
                let mut agent = node.agent.lock().await;
                agent.clear_messages();
            }

            debug!(node_id = %node_id, execution = execution_count, "Executing graph node");

            // Build input with upstream context
            let enriched_input = build_node_input(&input, &node_id, &results, task);

            // Execute with timeout
            let node_start = Instant::now();
            let agent_result = {
                let mut agent = node.agent.lock().await;
                tokio::time::timeout(self.config.node_timeout, agent.prompt(&enriched_input)).await
            };

            let node_result = match agent_result {
                Ok(Ok(result)) => {
                    let text = result.text();
                    output = text.clone();

                    accumulated_usage.input_tokens = Some(
                        accumulated_usage.input_tokens.unwrap_or(0)
                            + result.usage.input_tokens.unwrap_or(0),
                    );
                    accumulated_usage.output_tokens = Some(
                        accumulated_usage.output_tokens.unwrap_or(0)
                            + result.usage.output_tokens.unwrap_or(0),
                    );

                    NodeResult {
                        node_id: node_id.clone(),
                        status: NodeStatus::Completed,
                        result: Some(result),
                        error: None,
                        execution_time: node_start.elapsed(),
                    }
                }
                Ok(Err(e)) => {
                    warn!(node_id = %node_id, error = %e, "Graph node failed");
                    NodeResult {
                        node_id: node_id.clone(),
                        status: NodeStatus::Failed,
                        result: None,
                        error: Some(e.to_string()),
                        execution_time: node_start.elapsed(),
                    }
                }
                Err(_) => NodeResult {
                    node_id: node_id.clone(),
                    status: NodeStatus::Failed,
                    result: None,
                    error: Some("Node execution timed out".to_string()),
                    execution_time: node_start.elapsed(),
                },
            };

            let node_completed = node_result.status == NodeStatus::Completed;
            results.insert(node_id.clone(), node_result);
            execution_order.push(node_id.clone());
            execution_count += 1;
            completed.insert(node_id.clone());

            if !node_completed {
                // Node failed — stop graph execution
                return Ok(MultiAgentResult {
                    status: MultiAgentStatus::Failed,
                    results,
                    execution_order,
                    execution_count,
                    execution_time: start.elapsed(),
                    accumulated_usage,
                    output,
                });
            }

            // Evaluate outgoing edges and queue downstream nodes
            let graph_state = GraphState {
                results: results.clone(),
                task: task.to_string(),
            };

            if let Some(out_edges) = adjacency.get(&node_id) {
                for edge in out_edges {
                    let should_traverse = match &edge.condition {
                        Some(cond) => cond(&graph_state),
                        None => true,
                    };

                    if should_traverse {
                        debug!(
                            from = %edge.from_node,
                            to = %edge.to_node,
                            "Traversing graph edge"
                        );
                        pending.push_back((edge.to_node.clone(), output.clone()));
                    }
                }
            }
        }

        Ok(MultiAgentResult {
            status: MultiAgentStatus::Completed,
            results,
            execution_order,
            execution_count,
            execution_time: start.elapsed(),
            accumulated_usage,
            output,
        })
    }
}

/// Build enriched input for a node including upstream context.
fn build_node_input(
    input: &str,
    _node_id: &str,
    results: &HashMap<String, NodeResult>,
    original_task: &str,
) -> String {
    let mut ctx = format!("## Original Task\n{original_task}\n\n");

    // Add results from completed upstream nodes
    if !results.is_empty() {
        ctx.push_str("## Upstream Results\n");
        for (id, result) in results {
            if result.status == NodeStatus::Completed {
                if let Some(ref agent_result) = result.result {
                    let text = agent_result.text();
                    if !text.is_empty() {
                        ctx.push_str(&format!("### {id}\n{text}\n\n"));
                    }
                }
            }
        }
    }

    ctx.push_str(&format!("## Your Task\n{input}"));
    ctx
}

/// Builder for constructing a Graph.
pub struct GraphBuilder {
    agents: Vec<(String, Agent)>,
    edges: Vec<GraphEdge>,
    entry_points: Vec<String>,
    config: GraphConfig,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            agents: Vec::new(),
            edges: Vec::new(),
            entry_points: Vec::new(),
            config: GraphConfig::default(),
        }
    }

    /// Add an agent as a node.
    pub fn node(mut self, id: impl Into<String>, agent: Agent) -> Self {
        self.agents.push((id.into(), agent));
        self
    }

    /// Add an unconditional edge between two nodes.
    pub fn edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.push(GraphEdge {
            from_node: from.into(),
            to_node: to.into(),
            condition: None,
        });
        self
    }

    /// Add a conditional edge between two nodes.
    pub fn conditional_edge(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        condition: impl Fn(&GraphState) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.edges.push(GraphEdge {
            from_node: from.into(),
            to_node: to.into(),
            condition: Some(Arc::new(condition)),
        });
        self
    }

    /// Set one or more entry points (default: auto-detect source nodes).
    pub fn entry_point(mut self, id: impl Into<String>) -> Self {
        self.entry_points.push(id.into());
        self
    }

    /// Set maximum total node executions (important for cyclic graphs).
    pub fn max_node_executions(mut self, n: usize) -> Self {
        self.config.max_node_executions = n;
        self
    }

    /// Set total execution timeout.
    pub fn execution_timeout(mut self, d: Duration) -> Self {
        self.config.execution_timeout = d;
        self
    }

    /// Set per-node timeout.
    pub fn node_timeout(mut self, d: Duration) -> Self {
        self.config.node_timeout = d;
        self
    }

    /// Whether to reset agent state when revisiting a node in cyclic graphs.
    pub fn reset_on_revisit(mut self, reset: bool) -> Self {
        self.config.reset_on_revisit = reset;
        self
    }

    /// Build the graph.
    pub fn build(self) -> Result<Graph, StrandsError> {
        if self.agents.is_empty() {
            return Err(StrandsError::Other("Graph requires at least one node".into()));
        }

        let mut nodes = HashMap::new();
        for (id, agent) in self.agents {
            nodes.insert(
                id.clone(),
                GraphNode {
                    agent: Arc::new(Mutex::new(agent)),
                },
            );
        }

        // Validate edges reference existing nodes
        for edge in &self.edges {
            if !nodes.contains_key(&edge.from_node) {
                return Err(StrandsError::Other(format!(
                    "Edge source '{}' not found in graph nodes",
                    edge.from_node
                )));
            }
            if !nodes.contains_key(&edge.to_node) {
                return Err(StrandsError::Other(format!(
                    "Edge target '{}' not found in graph nodes",
                    edge.to_node
                )));
            }
        }

        // Auto-detect entry points if none specified
        let entry_points = if self.entry_points.is_empty() {
            // Source nodes have no incoming edges
            let targets: HashSet<&String> = self.edges.iter().map(|e| &e.to_node).collect();
            let sources: Vec<String> = nodes
                .keys()
                .filter(|id| !targets.contains(id))
                .cloned()
                .collect();
            if sources.is_empty() {
                // Cyclic graph — use first node
                vec![nodes.keys().next().unwrap().clone()]
            } else {
                sources
            }
        } else {
            // Validate specified entry points
            for ep in &self.entry_points {
                if !nodes.contains_key(ep) {
                    return Err(StrandsError::Other(format!(
                        "Entry point '{}' not found in graph nodes",
                        ep
                    )));
                }
            }
            self.entry_points
        };

        Ok(Graph {
            nodes,
            edges: self.edges,
            entry_points,
            config: self.config,
        })
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}
