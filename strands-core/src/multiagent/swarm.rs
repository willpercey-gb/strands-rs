use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use tracing::{debug, warn};

use crate::agent::Agent;
use crate::error::StrandsError;
use crate::types::streaming::Usage;

use super::result::*;

/// Configuration for a Swarm orchestration.
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    /// Maximum number of agent handoffs allowed.
    pub max_handoffs: usize,
    /// Maximum total iterations across all agents.
    pub max_iterations: usize,
    /// Total execution timeout.
    pub execution_timeout: Duration,
    /// Individual agent timeout.
    pub node_timeout: Duration,
    /// Window of recent nodes to check for repetitive handoffs (0 = disabled).
    pub repetitive_handoff_detection_window: usize,
    /// Minimum unique agents required in the detection window.
    pub repetitive_handoff_min_unique_agents: usize,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            max_handoffs: 20,
            max_iterations: 20,
            execution_timeout: Duration::from_secs(900),
            node_timeout: Duration::from_secs(300),
            repetitive_handoff_detection_window: 0,
            repetitive_handoff_min_unique_agents: 0,
        }
    }
}

/// A node within a Swarm.
struct SwarmNode {
    description: String,
    agent: Arc<Mutex<Agent>>,
}

/// A collaborative multi-agent orchestration where specialized agents
/// hand off work to each other autonomously.
///
/// Each agent is automatically equipped with a `handoff_to_agent` tool
/// that allows it to transfer control to another agent in the swarm.
///
/// # Example
///
/// ```ignore
/// let researcher = Agent::builder()
///     .model(model.clone())
///     .system_prompt("You are a research specialist.")
///     .build()?;
///
/// let coder = Agent::builder()
///     .model(model.clone())
///     .system_prompt("You are a coding specialist.")
///     .build()?;
///
/// let swarm = Swarm::builder()
///     .agent("researcher", "Researches topics", researcher)
///     .agent("coder", "Writes code", coder)
///     .entry_point("researcher")
///     .build()?;
///
/// let result = swarm.run("Design a REST API").await?;
/// ```
pub struct Swarm {
    nodes: HashMap<String, SwarmNode>,
    entry_point: String,
    config: SwarmConfig,
}

impl Swarm {
    pub fn builder() -> SwarmBuilder {
        SwarmBuilder::new()
    }

    /// Run the swarm with the given task.
    pub async fn run(&self, task: &str) -> Result<MultiAgentResult, StrandsError> {
        let start = Instant::now();
        let mut results: HashMap<String, NodeResult> = HashMap::new();
        let mut execution_order: Vec<String> = Vec::new();
        let mut accumulated_usage = Usage::default();
        let mut handoff_count = 0;
        let mut iteration_count = 0;
        let mut current_node_id = self.entry_point.clone();
        let mut current_input = task.to_string();
        let mut shared_knowledge = Vec::<String>::new();
        let mut output = String::new();

        // Build the list of available agents for context
        let agent_descriptions: Vec<String> = self
            .nodes
            .iter()
            .map(|(id, node)| format!("- {id}: {}", node.description))
            .collect();
        let agents_list = agent_descriptions.join("\n");

        loop {
            // Check limits
            if iteration_count >= self.config.max_iterations {
                return Ok(MultiAgentResult {
                    status: MultiAgentStatus::MaxStepsReached,
                    results,
                    execution_order,
                    execution_count: iteration_count,
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
                    execution_count: iteration_count,
                    execution_time: start.elapsed(),
                    accumulated_usage,
                    output,
                });
            }

            // Check repetitive handoff detection
            if self.config.repetitive_handoff_detection_window > 0 {
                let window = &execution_order
                    [execution_order.len().saturating_sub(self.config.repetitive_handoff_detection_window)..];
                let unique: std::collections::HashSet<&String> = window.iter().collect();
                if window.len() >= self.config.repetitive_handoff_detection_window
                    && unique.len() < self.config.repetitive_handoff_min_unique_agents
                {
                    warn!("Repetitive handoff detected, stopping swarm");
                    return Ok(MultiAgentResult {
                        status: MultiAgentStatus::Failed,
                        results,
                        execution_order,
                        execution_count: iteration_count,
                        execution_time: start.elapsed(),
                        accumulated_usage,
                        output: "Repetitive handoff detected".to_string(),
                    });
                }
            }

            let node = self.nodes.get(&current_node_id).ok_or_else(|| {
                StrandsError::Other(format!("Agent '{}' not found in swarm", current_node_id))
            })?;

            debug!(node_id = %current_node_id, iteration = iteration_count, "Executing swarm node");

            // Build context for the agent
            let context = build_swarm_context(
                &current_input,
                &execution_order,
                &shared_knowledge,
                &agents_list,
                &current_node_id,
            );

            // Execute with timeout
            let node_start = Instant::now();
            let agent_result = {
                let mut agent = node.agent.lock().await;
                tokio::time::timeout(self.config.node_timeout, agent.prompt(&context)).await
            };

            let node_result = match agent_result {
                Ok(Ok(result)) => {
                    let text = result.text();
                    output = text.clone();

                    // Accumulate usage
                    accumulated_usage.input_tokens = Some(
                        accumulated_usage.input_tokens.unwrap_or(0)
                            + result.usage.input_tokens.unwrap_or(0),
                    );
                    accumulated_usage.output_tokens = Some(
                        accumulated_usage.output_tokens.unwrap_or(0)
                            + result.usage.output_tokens.unwrap_or(0),
                    );

                    // Add to shared knowledge
                    shared_knowledge.push(format!("{}: {}", current_node_id, text));

                    NodeResult {
                        node_id: current_node_id.clone(),
                        status: NodeStatus::Completed,
                        result: Some(result),
                        error: None,
                        execution_time: node_start.elapsed(),
                    }
                }
                Ok(Err(e)) => NodeResult {
                    node_id: current_node_id.clone(),
                    status: NodeStatus::Failed,
                    result: None,
                    error: Some(e.to_string()),
                    execution_time: node_start.elapsed(),
                },
                Err(_) => NodeResult {
                    node_id: current_node_id.clone(),
                    status: NodeStatus::Failed,
                    result: None,
                    error: Some("Node execution timed out".to_string()),
                    execution_time: node_start.elapsed(),
                },
            };

            let node_failed = node_result.status == NodeStatus::Failed;
            results.insert(current_node_id.clone(), node_result);
            execution_order.push(current_node_id.clone());
            iteration_count += 1;

            if node_failed {
                return Ok(MultiAgentResult {
                    status: MultiAgentStatus::Failed,
                    results,
                    execution_order,
                    execution_count: iteration_count,
                    execution_time: start.elapsed(),
                    accumulated_usage,
                    output,
                });
            }

            // Check if the agent's response contains a handoff instruction.
            // We look for a JSON-like pattern indicating the agent wants to hand off.
            // The agent should include handoff info in its response when it wants to delegate.
            let handoff = parse_handoff(&output, &self.nodes);

            match handoff {
                Some((target_id, message)) => {
                    handoff_count += 1;
                    if handoff_count > self.config.max_handoffs {
                        return Ok(MultiAgentResult {
                            status: MultiAgentStatus::MaxStepsReached,
                            results,
                            execution_order,
                            execution_count: iteration_count,
                            execution_time: start.elapsed(),
                            accumulated_usage,
                            output,
                        });
                    }
                    debug!(
                        from = %current_node_id,
                        to = %target_id,
                        "Swarm handoff"
                    );
                    current_node_id = target_id;
                    current_input = message;
                }
                None => {
                    // No handoff — swarm is done
                    return Ok(MultiAgentResult {
                        status: MultiAgentStatus::Completed,
                        results,
                        execution_order,
                        execution_count: iteration_count,
                        execution_time: start.elapsed(),
                        accumulated_usage,
                        output,
                    });
                }
            }
        }
    }
}

/// Build the context string sent to each agent in the swarm.
fn build_swarm_context(
    input: &str,
    execution_order: &[String],
    shared_knowledge: &[String],
    agents_list: &str,
    current_agent: &str,
) -> String {
    let mut ctx = format!("## Task\n{input}\n\n");

    if !execution_order.is_empty() {
        ctx.push_str("## Previous Agents\n");
        for agent_id in execution_order {
            ctx.push_str(&format!("- {agent_id}\n"));
        }
        ctx.push('\n');
    }

    if !shared_knowledge.is_empty() {
        ctx.push_str("## Shared Knowledge\n");
        for knowledge in shared_knowledge {
            ctx.push_str(&format!("{knowledge}\n\n"));
        }
    }

    ctx.push_str(&format!(
        "## Available Agents\n{agents_list}\n\n\
         You are: {current_agent}\n\n\
         If you need another agent's expertise, include a handoff instruction in your response \
         in the format: HANDOFF_TO: <agent_name> | <message>\n\
         If you can complete the task yourself, just provide your response without a handoff."
    ));

    ctx
}

/// Parse a handoff instruction from the agent's response.
/// Looks for "HANDOFF_TO: <agent_name> | <message>" pattern.
fn parse_handoff(
    response: &str,
    nodes: &HashMap<String, SwarmNode>,
) -> Option<(String, String)> {
    for line in response.lines().rev() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("HANDOFF_TO:") {
            let rest = rest.trim();
            let (agent_name, message) = if let Some(pos) = rest.find('|') {
                (rest[..pos].trim().to_string(), rest[pos + 1..].trim().to_string())
            } else {
                (rest.to_string(), String::new())
            };

            if nodes.contains_key(&agent_name) {
                return Some((agent_name, message));
            }
        }
    }
    None
}

/// Builder for constructing a Swarm.
pub struct SwarmBuilder {
    agents: Vec<(String, String, Agent)>,
    entry_point: Option<String>,
    config: SwarmConfig,
}

impl SwarmBuilder {
    pub fn new() -> Self {
        Self {
            agents: Vec::new(),
            entry_point: None,
            config: SwarmConfig::default(),
        }
    }

    /// Add an agent to the swarm.
    pub fn agent(
        mut self,
        id: impl Into<String>,
        description: impl Into<String>,
        agent: Agent,
    ) -> Self {
        self.agents.push((id.into(), description.into(), agent));
        self
    }

    /// Set the entry point agent (default: first agent added).
    pub fn entry_point(mut self, id: impl Into<String>) -> Self {
        self.entry_point = Some(id.into());
        self
    }

    /// Set the maximum number of handoffs.
    pub fn max_handoffs(mut self, n: usize) -> Self {
        self.config.max_handoffs = n;
        self
    }

    /// Set the maximum number of iterations.
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.config.max_iterations = n;
        self
    }

    /// Set the execution timeout.
    pub fn execution_timeout(mut self, d: Duration) -> Self {
        self.config.execution_timeout = d;
        self
    }

    /// Set the per-node timeout.
    pub fn node_timeout(mut self, d: Duration) -> Self {
        self.config.node_timeout = d;
        self
    }

    /// Configure repetitive handoff detection.
    pub fn repetitive_handoff_detection(mut self, window: usize, min_unique: usize) -> Self {
        self.config.repetitive_handoff_detection_window = window;
        self.config.repetitive_handoff_min_unique_agents = min_unique;
        self
    }

    /// Build the swarm.
    pub fn build(self) -> Result<Swarm, StrandsError> {
        if self.agents.is_empty() {
            return Err(StrandsError::Other("Swarm requires at least one agent".into()));
        }

        let entry_point = self
            .entry_point
            .unwrap_or_else(|| self.agents[0].0.clone());

        let mut nodes = HashMap::new();
        for (id, description, agent) in self.agents {
            nodes.insert(
                id.clone(),
                SwarmNode {
                    description,
                    agent: Arc::new(Mutex::new(agent)),
                },
            );
        }

        if !nodes.contains_key(&entry_point) {
            return Err(StrandsError::Other(format!(
                "Entry point '{}' not found in swarm agents",
                entry_point
            )));
        }

        Ok(Swarm {
            nodes,
            entry_point,
            config: self.config,
        })
    }
}

impl Default for SwarmBuilder {
    fn default() -> Self {
        Self::new()
    }
}
