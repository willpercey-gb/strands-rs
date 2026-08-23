//! Interventions — policy that gates what an agent is allowed to do.
//!
//! A hook can already cancel a tool call. An intervention is the structured
//! version: a named handler that inspects a proposed action and returns allow,
//! deny, or ask-a-human, with a reason attached.
//!
//! The distinction matters because "deny" and "ask" have different failure
//! modes. A denial must be final and explain itself to the model; an escalation
//! must pause the run rather than guessing on the human's behalf. Collapsing
//! them into a boolean loses that.
//!
//! Ported from upstream `interventions/` and `vended_interventions/`.

/// Applying policy to tool execution.
pub mod executor;
/// Escalating risky calls to a human.
pub mod hitl;

use async_trait::async_trait;
use serde_json::Value;
use tracing::debug;

pub use executor::InterventionExecutor;
pub use hitl::{HumanInTheLoop, RiskClassifier, RiskLevel, ToolNameClassifier};

/// What a handler decided about a proposed action.
#[derive(Debug, Clone, PartialEq)]
pub enum InterventionAction {
    /// Let it proceed.
    Allow,
    /// Refuse, with a reason the model sees.
    /// Refuse, with a reason the model sees.
    Deny {
        /// Why the call was refused.
        reason: String,
    },
    /// Pause and ask a human, under this interrupt name.
    /// Pause and ask a human.
    Escalate {
        /// Interrupt name the question is raised under.
        name: String,
        /// What the human is being asked to decide.
        reason: String,
    },
}

impl InterventionAction {
    /// Refuse with a reason.
    pub fn deny(reason: impl Into<String>) -> Self {
        InterventionAction::Deny {
            reason: reason.into(),
        }
    }

    /// Escalate to a human under `name`.
    pub fn escalate(name: impl Into<String>, reason: impl Into<String>) -> Self {
        InterventionAction::Escalate {
            name: name.into(),
            reason: reason.into(),
        }
    }

    /// Whether this decision permits the call.
    pub fn is_allow(&self) -> bool {
        matches!(self, InterventionAction::Allow)
    }
}

/// The action being judged.
#[derive(Debug, Clone)]
pub struct InterventionContext {
    /// Tool the model wants to call.
    pub tool_name: String,
    /// Arguments it supplied.
    pub input: Value,
}

/// A policy that judges proposed tool calls.
#[async_trait]
pub trait InterventionHandler: Send + Sync {
    /// Name, for diagnostics.
    fn name(&self) -> &str;

    /// Judge one proposed action.
    async fn evaluate(&self, ctx: &InterventionContext) -> InterventionAction;
}

/// An ordered set of handlers.
///
/// Evaluated in order, stopping at the first non-`Allow`. Order therefore
/// matters: a broad denial registered first shadows anything after it.
#[derive(Default)]
pub struct InterventionRegistry {
    handlers: Vec<Box<dyn InterventionHandler>>,
}

impl InterventionRegistry {
    /// Create with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a handler. Order matters — see [`evaluate`](Self::evaluate).
    pub fn register(&mut self, handler: impl InterventionHandler + 'static) {
        self.handlers.push(Box::new(handler));
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    /// Evaluate every handler, returning the first non-`Allow` decision.
    ///
    /// Fails closed: the first refusal or escalation wins, and later handlers
    /// are not consulted. A handler cannot un-deny what an earlier one denied.
    pub async fn evaluate(&self, ctx: &InterventionContext) -> InterventionAction {
        for handler in &self.handlers {
            let action = handler.evaluate(ctx).await;
            if !action.is_allow() {
                debug!(
                    handler = handler.name(),
                    tool = %ctx.tool_name,
                    ?action,
                    "Intervention returned a non-allow decision"
                );
                return action;
            }
        }
        InterventionAction::Allow
    }
}

impl std::fmt::Debug for InterventionRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InterventionRegistry")
            .field("handlers", &self.handlers.len())
            .finish()
    }
}

/// Denies any tool whose name is on a list.
pub struct DenyList {
    denied: std::collections::HashSet<String>,
}

impl DenyList {
    /// Create a new instance.
    pub fn new<I, S>(tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            denied: tools.into_iter().map(Into::into).collect(),
        }
    }
}

#[async_trait]
impl InterventionHandler for DenyList {
    /// Name, for logs and diagnostics.
    fn name(&self) -> &str {
        "deny_list"
    }

    async fn evaluate(&self, ctx: &InterventionContext) -> InterventionAction {
        if self.denied.contains(&ctx.tool_name) {
            InterventionAction::deny(format!("tool '{}' is not permitted", ctx.tool_name))
        } else {
            InterventionAction::Allow
        }
    }
}

/// Allows only tools on a list, denying everything else.
///
/// Fails closed by construction, which is what makes it usable as the outermost
/// policy: a tool added later is denied until someone permits it explicitly.
pub struct AllowList {
    allowed: std::collections::HashSet<String>,
}

impl AllowList {
    /// Create a new instance.
    pub fn new<I, S>(tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            allowed: tools.into_iter().map(Into::into).collect(),
        }
    }
}

#[async_trait]
impl InterventionHandler for AllowList {
    /// Name, for logs and diagnostics.
    fn name(&self) -> &str {
        "allow_list"
    }

    async fn evaluate(&self, ctx: &InterventionContext) -> InterventionAction {
        if self.allowed.contains(&ctx.tool_name) {
            InterventionAction::Allow
        } else {
            InterventionAction::deny(format!("tool '{}' is not on the allow list", ctx.tool_name))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ctx(tool: &str) -> InterventionContext {
        InterventionContext {
            tool_name: tool.to_string(),
            input: json!({}),
        }
    }

    struct AlwaysEscalate;

    #[async_trait]
    impl InterventionHandler for AlwaysEscalate {
        fn name(&self) -> &str {
            "escalate"
        }
        async fn evaluate(&self, _ctx: &InterventionContext) -> InterventionAction {
            InterventionAction::escalate("approve", "needs a human")
        }
    }

    #[tokio::test]
    async fn an_empty_registry_allows() {
        let registry = InterventionRegistry::new();
        assert!(registry.evaluate(&ctx("anything")).await.is_allow());
    }

    #[tokio::test]
    async fn deny_list_blocks_named_tools() {
        let mut registry = InterventionRegistry::new();
        registry.register(DenyList::new(["rm"]));

        assert!(registry.evaluate(&ctx("ls")).await.is_allow());
        assert!(matches!(
            registry.evaluate(&ctx("rm")).await,
            InterventionAction::Deny { .. }
        ));
    }

    #[tokio::test]
    async fn allow_list_fails_closed() {
        // A tool added later must be denied until someone permits it, which is
        // what makes this safe as the outermost policy.
        let mut registry = InterventionRegistry::new();
        registry.register(AllowList::new(["ls"]));

        assert!(registry.evaluate(&ctx("ls")).await.is_allow());
        assert!(matches!(
            registry.evaluate(&ctx("newly_added")).await,
            InterventionAction::Deny { .. }
        ));
    }

    #[tokio::test]
    async fn the_first_non_allow_decision_wins() {
        let mut registry = InterventionRegistry::new();
        registry.register(DenyList::new(["rm"]));
        registry.register(AlwaysEscalate);

        // The denial comes first and must not be softened into an escalation.
        assert!(matches!(
            registry.evaluate(&ctx("rm")).await,
            InterventionAction::Deny { .. }
        ));
    }

    #[tokio::test]
    async fn later_handlers_still_run_when_earlier_ones_allow() {
        let mut registry = InterventionRegistry::new();
        registry.register(DenyList::new(["rm"]));
        registry.register(AlwaysEscalate);

        assert!(matches!(
            registry.evaluate(&ctx("ls")).await,
            InterventionAction::Escalate { .. }
        ));
    }

    #[tokio::test]
    async fn a_denial_carries_a_reason_for_the_model() {
        let handler = DenyList::new(["rm"]);
        match handler.evaluate(&ctx("rm")).await {
            InterventionAction::Deny { reason } => assert!(reason.contains("rm")),
            other => panic!("expected a denial, got {other:?}"),
        }
    }
}
