//! Applying interventions to tool execution.
//!
//! Wraps another [`ToolExecutor`](crate::tool::ToolExecutor), consulting an
//! [`InterventionRegistry`]
//! before each call. This is the async seam interventions need — hooks are
//! synchronous, and a policy that has to query a service or a policy engine
//! cannot run there.

use std::collections::HashMap;

use async_trait::async_trait;
use tracing::{debug, warn};

use crate::hooks::registry::HookRegistry;
use crate::interrupt::InterruptState;
use crate::tool::executor::{to_result_block, ToolCall, ToolExecutor};
use crate::tool::{Tool, ToolContext, ToolOutput};
use crate::types::content::ContentBlock;

use super::{InterventionAction, InterventionContext, InterventionRegistry};

/// Gates tool execution behind an intervention policy.
pub struct InterventionExecutor {
    inner: Box<dyn ToolExecutor>,
    registry: InterventionRegistry,
}

impl InterventionExecutor {
    /// Create a new instance.
    pub fn new(inner: impl ToolExecutor + 'static, registry: InterventionRegistry) -> Self {
        Self {
            inner: Box::new(inner),
            registry,
        }
    }
}

#[async_trait]
impl ToolExecutor for InterventionExecutor {
    fn name(&self) -> &'static str {
        "intervention"
    }

    async fn execute(
        &self,
        tools: &HashMap<String, Box<dyn Tool>>,
        calls: &[ToolCall<'_>],
        ctx: &ToolContext,
        hooks: &HookRegistry,
        interrupts: &mut InterruptState,
    ) -> Vec<ContentBlock> {
        // Judge every call first, so a denial does not depend on how far the
        // batch got before it was reached.
        let mut decisions = Vec::with_capacity(calls.len());
        for call in calls {
            let intervention_ctx = InterventionContext {
                tool_name: call.name.to_string(),
                input: call.input.clone(),
            };
            decisions.push(self.registry.evaluate(&intervention_ctx).await);
        }

        // Escalations become interrupts. An escalated call is withheld this
        // pass; once the human answers, the same call is re-proposed and the
        // handler sees the answer.
        let mut permitted: Vec<ToolCall<'_>> = Vec::new();
        let mut blocked: Vec<(usize, ContentBlock)> = Vec::new();

        for (index, (call, decision)) in calls.iter().zip(&decisions).enumerate() {
            match decision {
                InterventionAction::Allow => permitted.push(*call),

                InterventionAction::Deny { reason } => {
                    warn!(tool = call.name, reason, "Tool call denied by intervention");
                    blocked.push((
                        index,
                        to_result_block(call.tool_use_id, &ToolOutput::error(reason.clone())),
                    ));
                }

                InterventionAction::Escalate { name, reason } => {
                    match interrupts.interrupt(name, Some(reason.clone().into())) {
                        Some(answer) if answer.as_str() == Some("approve") => {
                            debug!(tool = call.name, "Escalation approved");
                            permitted.push(*call);
                        }
                        Some(_) => {
                            debug!(tool = call.name, "Escalation denied by human");
                            blocked.push((
                                index,
                                to_result_block(
                                    call.tool_use_id,
                                    &ToolOutput::error("approval was not granted"),
                                ),
                            ));
                        }
                        None => {
                            // Awaiting a human. Withhold the call rather than
                            // guessing — that is the whole point of escalating.
                            debug!(tool = call.name, "Awaiting human approval");
                            blocked.push((
                                index,
                                to_result_block(
                                    call.tool_use_id,
                                    &ToolOutput::error("awaiting human approval"),
                                ),
                            ));
                        }
                    }
                }
            }
        }

        let mut executed = self
            .inner
            .execute(tools, &permitted, ctx, hooks, interrupts)
            .await
            .into_iter();

        // Reassemble in the caller's order: providers match results to calls
        // positionally as well as by id.
        let mut results = Vec::with_capacity(calls.len());
        let mut blocked = blocked.into_iter().peekable();
        for index in 0..calls.len() {
            if blocked.peek().is_some_and(|(i, _)| *i == index) {
                results.push(blocked.next().expect("peeked").1);
            } else if let Some(result) = executed.next() {
                results.push(result);
            }
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interrupt::InterruptResponse;
    use crate::interventions::{DenyList, HumanInTheLoop};
    use crate::tool::SequentialToolExecutor;
    use crate::types::content::ToolResultStatus;
    use crate::types::tools::ToolSpec;
    use serde_json::{json, Value};
    use std::sync::{Arc, Mutex};

    struct RecordingTool {
        tool_name: &'static str,
        ran: Arc<Mutex<Vec<&'static str>>>,
    }

    #[async_trait]
    impl Tool for RecordingTool {
        fn name(&self) -> &str {
            self.tool_name
        }
        fn spec(&self) -> ToolSpec {
            ToolSpec::new(self.tool_name, "t", json!({"type": "object"}))
        }
        async fn invoke(
            &self,
            _input: Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, crate::error::StrandsError> {
            self.ran.lock().unwrap().push(self.tool_name);
            Ok(ToolOutput::success(json!("ok")))
        }
    }

    fn tools(ran: Arc<Mutex<Vec<&'static str>>>) -> HashMap<String, Box<dyn Tool>> {
        ["safe", "danger"]
            .into_iter()
            .map(|name| {
                (
                    name.to_string(),
                    Box::new(RecordingTool {
                        tool_name: name,
                        ran: ran.clone(),
                    }) as Box<dyn Tool>,
                )
            })
            .collect()
    }

    fn status_of(block: &ContentBlock) -> ToolResultStatus {
        match block {
            ContentBlock::ToolResult { status, .. } => status.clone(),
            other => panic!("expected a tool result, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_denied_call_never_runs_but_still_gets_a_result() {
        let ran = Arc::new(Mutex::new(Vec::new()));
        let mut registry = InterventionRegistry::new();
        registry.register(DenyList::new(["danger"]));

        let executor = InterventionExecutor::new(SequentialToolExecutor, registry);

        let input = json!({});
        let calls = vec![
            ToolCall {
                tool_use_id: "1",
                name: "safe",
                input: &input,
            },
            ToolCall {
                tool_use_id: "2",
                name: "danger",
                input: &input,
            },
        ];

        let results = executor
            .execute(
                &tools(ran.clone()),
                &calls,
                &ToolContext::default(),
                &HookRegistry::new(),
                &mut InterruptState::new(),
            )
            .await;

        assert_eq!(*ran.lock().unwrap(), vec!["safe"]);
        assert_eq!(results.len(), 2, "every call needs a result");
        assert_eq!(status_of(&results[0]), ToolResultStatus::Success);
        assert_eq!(status_of(&results[1]), ToolResultStatus::Error);
    }

    #[tokio::test]
    async fn results_stay_in_request_order_around_a_denial() {
        // Providers match results positionally as well as by id.
        let ran = Arc::new(Mutex::new(Vec::new()));
        let mut registry = InterventionRegistry::new();
        registry.register(DenyList::new(["danger"]));

        let executor = InterventionExecutor::new(SequentialToolExecutor, registry);
        let input = json!({});
        let calls = vec![
            ToolCall {
                tool_use_id: "denied",
                name: "danger",
                input: &input,
            },
            ToolCall {
                tool_use_id: "allowed",
                name: "safe",
                input: &input,
            },
        ];

        let results = executor
            .execute(
                &tools(ran),
                &calls,
                &ToolContext::default(),
                &HookRegistry::new(),
                &mut InterruptState::new(),
            )
            .await;

        let ids: Vec<&str> = results
            .iter()
            .map(|b| match b {
                ContentBlock::ToolResult { tool_use_id, .. } => tool_use_id.as_str(),
                _ => "",
            })
            .collect();
        assert_eq!(ids, vec!["denied", "allowed"]);
    }

    #[tokio::test]
    async fn an_escalation_withholds_the_call_and_raises_an_interrupt() {
        let ran = Arc::new(Mutex::new(Vec::new()));
        let mut registry = InterventionRegistry::new();
        registry.register(HumanInTheLoop::for_tools(["danger"]));

        let executor = InterventionExecutor::new(SequentialToolExecutor, registry);
        let mut interrupts = InterruptState::new();

        let input = json!({});
        let calls = vec![ToolCall {
            tool_use_id: "1",
            name: "danger",
            input: &input,
        }];

        executor
            .execute(
                &tools(ran.clone()),
                &calls,
                &ToolContext::default(),
                &HookRegistry::new(),
                &mut interrupts,
            )
            .await;

        assert!(ran.lock().unwrap().is_empty(), "the call must be withheld");
        assert!(interrupts.has_pending(), "a human must be asked");
    }

    #[tokio::test]
    async fn an_approved_escalation_lets_the_call_through() {
        let ran = Arc::new(Mutex::new(Vec::new()));
        let mut registry = InterventionRegistry::new();
        registry.register(HumanInTheLoop::for_tools(["danger"]));

        let executor = InterventionExecutor::new(SequentialToolExecutor, registry);
        let mut interrupts = InterruptState::new();
        let input = json!({});
        let calls = vec![ToolCall {
            tool_use_id: "1",
            name: "danger",
            input: &input,
        }];
        let all_tools = tools(ran.clone());

        // First pass raises the interrupt.
        executor
            .execute(
                &all_tools,
                &calls,
                &ToolContext::default(),
                &HookRegistry::new(),
                &mut interrupts,
            )
            .await;

        let id = interrupts.pending()[0].id.clone();
        interrupts.respond(&[InterruptResponse::new(id, "approve")]);

        // Second pass sees the approval.
        executor
            .execute(
                &all_tools,
                &calls,
                &ToolContext::default(),
                &HookRegistry::new(),
                &mut interrupts,
            )
            .await;

        assert_eq!(*ran.lock().unwrap(), vec!["danger"]);
    }

    #[tokio::test]
    async fn a_refused_escalation_keeps_the_call_blocked() {
        let ran = Arc::new(Mutex::new(Vec::new()));
        let mut registry = InterventionRegistry::new();
        registry.register(HumanInTheLoop::for_tools(["danger"]));

        let executor = InterventionExecutor::new(SequentialToolExecutor, registry);
        let mut interrupts = InterruptState::new();
        let input = json!({});
        let calls = vec![ToolCall {
            tool_use_id: "1",
            name: "danger",
            input: &input,
        }];
        let all_tools = tools(ran.clone());

        executor
            .execute(
                &all_tools,
                &calls,
                &ToolContext::default(),
                &HookRegistry::new(),
                &mut interrupts,
            )
            .await;

        let id = interrupts.pending()[0].id.clone();
        interrupts.respond(&[InterruptResponse::new(id, "deny")]);

        executor
            .execute(
                &all_tools,
                &calls,
                &ToolContext::default(),
                &HookRegistry::new(),
                &mut interrupts,
            )
            .await;

        assert!(ran.lock().unwrap().is_empty(), "a refusal must hold");
    }

    #[tokio::test]
    async fn an_empty_policy_changes_nothing() {
        let ran = Arc::new(Mutex::new(Vec::new()));
        let executor =
            InterventionExecutor::new(SequentialToolExecutor, InterventionRegistry::new());

        let input = json!({});
        let calls = vec![ToolCall {
            tool_use_id: "1",
            name: "safe",
            input: &input,
        }];

        executor
            .execute(
                &tools(ran.clone()),
                &calls,
                &ToolContext::default(),
                &HookRegistry::new(),
                &mut InterruptState::new(),
            )
            .await;

        assert_eq!(*ran.lock().unwrap(), vec!["safe"]);
    }
}
