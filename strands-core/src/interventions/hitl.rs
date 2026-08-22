//! Human-in-the-loop intervention.
//!
//! Classifies a proposed tool call by risk and escalates the risky ones to a
//! human via the interrupt mechanism.
//!
//! Ported from upstream `vended_interventions/hitl/`.

use async_trait::async_trait;

use super::{InterventionAction, InterventionContext, InterventionHandler};

/// How dangerous a proposed action looks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

/// Judges the risk of a proposed tool call.
pub trait RiskClassifier: Send + Sync {
    fn classify(&self, ctx: &InterventionContext) -> RiskLevel;
}

/// Classifies by tool name against a set of high-risk tools.
pub struct ToolNameClassifier {
    high_risk: std::collections::HashSet<String>,
    /// Risk assigned to a tool that is not listed.
    ///
    /// Defaults to [`RiskLevel::Low`], so listing is opt-in; set it higher when
    /// the tool set is open-ended and an unrecognised tool should be treated as
    /// suspicious rather than safe.
    default_risk: RiskLevel,
}

impl ToolNameClassifier {
    pub fn new<I, S>(high_risk: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            high_risk: high_risk.into_iter().map(Into::into).collect(),
            default_risk: RiskLevel::Low,
        }
    }

    /// Treat unlisted tools as this risky.
    pub fn with_default_risk(mut self, risk: RiskLevel) -> Self {
        self.default_risk = risk;
        self
    }
}

impl RiskClassifier for ToolNameClassifier {
    fn classify(&self, ctx: &InterventionContext) -> RiskLevel {
        if self.high_risk.contains(&ctx.tool_name) {
            RiskLevel::High
        } else {
            self.default_risk
        }
    }
}

/// Escalates anything at or above a risk threshold to a human.
pub struct HumanInTheLoop {
    classifier: Box<dyn RiskClassifier>,
    threshold: RiskLevel,
}

impl HumanInTheLoop {
    pub fn new(classifier: impl RiskClassifier + 'static) -> Self {
        Self {
            classifier: Box::new(classifier),
            threshold: RiskLevel::High,
        }
    }

    /// Escalate at or above `threshold`.
    pub fn with_threshold(mut self, threshold: RiskLevel) -> Self {
        self.threshold = threshold;
        self
    }

    /// Escalate every listed tool, using name-based classification.
    pub fn for_tools<I, S>(tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::new(ToolNameClassifier::new(tools))
    }
}

#[async_trait]
impl InterventionHandler for HumanInTheLoop {
    fn name(&self) -> &str {
        "human_in_the_loop"
    }

    async fn evaluate(&self, ctx: &InterventionContext) -> InterventionAction {
        let risk = self.classifier.classify(ctx);
        if risk < self.threshold {
            return InterventionAction::Allow;
        }

        InterventionAction::escalate(
            format!("approve_{}", ctx.tool_name),
            format!(
                "{:?}-risk call to '{}' requires approval",
                risk, ctx.tool_name
            ),
        )
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

    #[tokio::test]
    async fn high_risk_tools_escalate() {
        let hitl = HumanInTheLoop::for_tools(["delete_everything"]);

        match hitl.evaluate(&ctx("delete_everything")).await {
            InterventionAction::Escalate { name, reason } => {
                assert_eq!(name, "approve_delete_everything");
                assert!(reason.contains("delete_everything"));
            }
            other => panic!("expected an escalation, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn low_risk_tools_pass() {
        let hitl = HumanInTheLoop::for_tools(["delete_everything"]);
        assert!(hitl.evaluate(&ctx("read_file")).await.is_allow());
    }

    #[tokio::test]
    async fn the_threshold_is_configurable() {
        let hitl = HumanInTheLoop::new(
            ToolNameClassifier::new(["danger"]).with_default_risk(RiskLevel::Medium),
        )
        .with_threshold(RiskLevel::Medium);

        // Even an unlisted tool escalates once the default risk meets the bar.
        assert!(!hitl.evaluate(&ctx("anything")).await.is_allow());
    }

    #[tokio::test]
    async fn an_open_ended_tool_set_can_default_to_suspicious() {
        // With tools arriving from an MCP server, "not on my list" is a reason
        // for caution rather than confidence.
        let hitl = HumanInTheLoop::new(
            ToolNameClassifier::new(Vec::<String>::new()).with_default_risk(RiskLevel::High),
        );
        assert!(!hitl.evaluate(&ctx("unknown_mcp_tool")).await.is_allow());
    }

    #[test]
    fn risk_levels_order_correctly() {
        assert!(RiskLevel::Low < RiskLevel::Medium);
        assert!(RiskLevel::Medium < RiskLevel::High);
    }
}
