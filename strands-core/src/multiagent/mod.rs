//! Orchestrating several agents.
//!
//! [`Swarm`] hands control between agents by autonomous handoff; [`Graph`]
//! routes output along a deterministic DAG with optional edge conditions. Both
//! report a [`MultiAgentResult`] with per-node
//! outcomes and accumulated usage.

/// Deterministic DAG orchestration.
pub mod graph;
/// Per-node and aggregate outcomes.
pub mod result;
/// Autonomous handoff between agents.
pub mod swarm;

pub use graph::{Graph, GraphBuilder, GraphEdge};
pub use result::{MultiAgentResult, MultiAgentStatus, NodeResult, NodeStatus};
pub use swarm::Swarm;
