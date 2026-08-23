//! Pluggable tool execution strategies.
//!
//! The agent decides *what* to call; a [`ToolExecutor`] decides *how* — one at
//! a time, all at once, or something bespoke (a bounded pool, a rate limiter, a
//! remote dispatcher). Extracting this from the event loop is also what lets a
//! middleware stage wrap tool execution later without touching the loop itself.
//!
//! Ported from upstream `tools/executors/`.

use std::collections::HashMap;
use std::time::Instant;

use async_trait::async_trait;
use serde_json::Value;
use tracing::{debug, warn};

use crate::hooks::events::{AfterToolCallEvent, BeforeToolCallEvent, HookEvent};
use crate::hooks::registry::HookRegistry;
use crate::interrupt::InterruptState;
use crate::types::content::{ContentBlock, ToolResultContent, ToolResultStatus};

use super::{Tool, ToolContext, ToolOutput};

/// One tool call requested by the model.
#[derive(Debug, Clone, Copy)]
pub struct ToolCall<'a> {
    /// Identifier the result must echo back.
    pub tool_use_id: &'a str,
    /// Tool to invoke.
    pub name: &'a str,
    /// Arguments from the model.
    pub input: &'a Value,
}

/// Strategy for running a batch of tool calls.
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Execute every call in `calls`, returning one `ToolResult` block per
    /// call, **in the same order**.
    ///
    /// Order matters: providers match results to calls positionally as well as
    /// by id, and a reordered batch is rejected by some of them.
    async fn execute(
        &self,
        tools: &HashMap<String, Box<dyn Tool>>,
        calls: &[ToolCall<'_>],
        ctx: &ToolContext,
        hooks: &HookRegistry,
        interrupts: &mut InterruptState,
    ) -> Vec<ContentBlock>;

    /// Human-readable name, for logs and diagnostics.
    fn name(&self) -> &'static str;
}

/// Run one tool, mapping a not-found or failing tool onto an error result
/// rather than aborting the batch.
///
/// A tool failure is information the model can act on — retry with different
/// arguments, or explain the failure — so it is surfaced as a result, not an
/// error that kills the invocation.
async fn invoke_one(
    tools: &HashMap<String, Box<dyn Tool>>,
    name: &str,
    input: &Value,
    ctx: &ToolContext,
) -> ToolOutput {
    match tools.get(name) {
        Some(tool) => {
            debug!(tool_name = name, "Invoking tool");
            match tool.invoke(input.clone(), ctx).await {
                Ok(output) => output,
                Err(e) => {
                    warn!(tool_name = name, error = %e, "Tool execution failed");
                    ToolOutput::error(e.to_string())
                }
            }
        }
        None => {
            warn!(tool_name = name, "Tool not found");
            ToolOutput::error(format!("Tool not found: {name}"))
        }
    }
}

/// Fire `BeforeToolCall` and report whether the hook cancelled the call.
///
/// The interrupt state is moved into the event and back out again, so a hook
/// that pauses for human input records the request where the agent loop can
/// see it.
fn dispatch_before(
    hooks: &HookRegistry,
    call: &ToolCall<'_>,
    interrupts: &mut InterruptState,
) -> bool {
    let mut event = HookEvent::BeforeToolCall(BeforeToolCallEvent {
        tool_name: call.name.to_string(),
        input: call.input.clone(),
        cancel: false,
        interrupts: std::mem::take(interrupts),
    });
    hooks.dispatch(&mut event);

    let HookEvent::BeforeToolCall(event) = event else {
        unreachable!("dispatch must not change the event variant")
    };
    *interrupts = event.interrupts;
    event.cancel
}

/// Fire `AfterToolCall` and report whether the hook asked for a retry.
fn dispatch_after(
    hooks: &HookRegistry,
    name: &str,
    output: &ToolOutput,
    duration: std::time::Duration,
) -> bool {
    let mut event = HookEvent::AfterToolCall(AfterToolCallEvent {
        tool_name: name.to_string(),
        is_error: output.is_error,
        duration,
        retry: false,
    });
    hooks.dispatch(&mut event);
    matches!(
        event,
        HookEvent::AfterToolCall(AfterToolCallEvent { retry: true, .. })
    )
}

/// Convert a tool output into the `ToolResult` block the model expects.
pub fn to_result_block(tool_use_id: &str, output: &ToolOutput) -> ContentBlock {
    let text = match &output.content {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    };

    ContentBlock::ToolResult {
        tool_use_id: tool_use_id.to_string(),
        status: if output.is_error {
            ToolResultStatus::Error
        } else {
            ToolResultStatus::Success
        },
        content: vec![ToolResultContent::Text { text }],
    }
}

/// Run one call end to end, including its hooks and a possible retry.
async fn run_with_hooks(
    tools: &HashMap<String, Box<dyn Tool>>,
    call: &ToolCall<'_>,
    ctx: &ToolContext,
    hooks: &HookRegistry,
    cancelled: bool,
) -> ToolOutput {
    let started = Instant::now();
    let output = if cancelled {
        debug!(tool_name = call.name, "Tool call cancelled by hook");
        ToolOutput::error("Tool call cancelled")
    } else {
        invoke_one(tools, call.name, call.input, ctx).await
    };
    let duration = started.elapsed();

    if dispatch_after(hooks, call.name, &output, duration) {
        debug!(tool_name = call.name, "Hook requested tool retry");
        // Retry with the *original* input: re-invoking with anything else runs
        // a different call than the one the hook asked to retry.
        invoke_one(tools, call.name, call.input, ctx).await
    } else {
        output
    }
}

/// Runs tools one at a time, in request order.
///
/// The safe default: tools that share mutable state, or whose side effects are
/// order-dependent, behave predictably here.
#[derive(Debug, Default, Clone, Copy)]
pub struct SequentialToolExecutor;

#[async_trait]
impl ToolExecutor for SequentialToolExecutor {
    fn name(&self) -> &'static str {
        "sequential"
    }

    async fn execute(
        &self,
        tools: &HashMap<String, Box<dyn Tool>>,
        calls: &[ToolCall<'_>],
        ctx: &ToolContext,
        hooks: &HookRegistry,
        interrupts: &mut InterruptState,
    ) -> Vec<ContentBlock> {
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            let cancelled = dispatch_before(hooks, call, interrupts);
            let output = run_with_hooks(tools, call, ctx, hooks, cancelled).await;
            results.push(to_result_block(call.tool_use_id, &output));
        }
        results
    }
}

/// Runs tools concurrently.
///
/// `BeforeToolCall` hooks still fire sequentially and in order, so a hook that
/// cancels based on what it has already seen behaves deterministically; only
/// the tool bodies overlap.
#[derive(Debug, Default, Clone, Copy)]
pub struct ConcurrentToolExecutor;

#[async_trait]
impl ToolExecutor for ConcurrentToolExecutor {
    fn name(&self) -> &'static str {
        "concurrent"
    }

    async fn execute(
        &self,
        tools: &HashMap<String, Box<dyn Tool>>,
        calls: &[ToolCall<'_>],
        ctx: &ToolContext,
        hooks: &HookRegistry,
        interrupts: &mut InterruptState,
    ) -> Vec<ContentBlock> {
        // Decide cancellation up front and in order — hooks are synchronous,
        // may depend on the sequence they observe, and share one interrupt
        // state that cannot be borrowed across concurrent futures.
        let cancelled: Vec<bool> = calls
            .iter()
            .map(|c| dispatch_before(hooks, c, interrupts))
            .collect();

        let futures = calls.iter().zip(&cancelled).map(|(call, &cancelled)| {
            let started = Instant::now();
            async move {
                let output = if cancelled {
                    ToolOutput::error("Tool call cancelled")
                } else {
                    invoke_one(tools, call.name, call.input, ctx).await
                };
                (call, output, started.elapsed())
            }
        });

        // join_all preserves input order, which the trait contract requires.
        let outputs = futures::future::join_all(futures).await;

        let mut results = Vec::with_capacity(outputs.len());
        for (call, output, duration) in outputs {
            let final_output = if dispatch_after(hooks, call.name, &output, duration) {
                debug!(tool_name = call.name, "Hook requested tool retry");
                invoke_one(tools, call.name, call.input, ctx).await
            } else {
                output
            };
            results.push(to_result_block(call.tool_use_id, &final_output));
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::tools::ToolSpec;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    struct EchoTool {
        tool_name: &'static str,
        /// Order in which invocations started, shared across tools.
        order: Arc<Mutex<Vec<&'static str>>>,
        delay_ms: u64,
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            self.tool_name
        }
        fn spec(&self) -> ToolSpec {
            ToolSpec::new(self.tool_name, "echo", json!({"type": "object"}))
        }
        async fn invoke(
            &self,
            input: Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, crate::error::StrandsError> {
            if self.delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            }
            self.order.lock().unwrap().push(self.tool_name);
            Ok(ToolOutput::success(input))
        }
    }

    fn registry(tools: Vec<Box<dyn Tool>>) -> HashMap<String, Box<dyn Tool>> {
        tools
            .into_iter()
            .map(|t| (t.name().to_string(), t))
            .collect()
    }

    fn result_texts(blocks: &[ContentBlock]) -> Vec<String> {
        blocks
            .iter()
            .map(|b| match b {
                ContentBlock::ToolResult { content, .. } => match &content[0] {
                    ToolResultContent::Text { text } => text.clone(),
                    _ => String::new(),
                },
                _ => String::new(),
            })
            .collect()
    }

    async fn run<E: ToolExecutor>(executor: E, delays: [u64; 2]) -> Vec<ContentBlock> {
        let order = Arc::new(Mutex::new(Vec::new()));
        let tools = registry(vec![
            Box::new(EchoTool {
                tool_name: "slow",
                order: order.clone(),
                delay_ms: delays[0],
            }),
            Box::new(EchoTool {
                tool_name: "fast",
                order: order.clone(),
                delay_ms: delays[1],
            }),
        ]);

        let slow_input = json!({"v": "slow"});
        let fast_input = json!({"v": "fast"});
        let calls = vec![
            ToolCall {
                tool_use_id: "1",
                name: "slow",
                input: &slow_input,
            },
            ToolCall {
                tool_use_id: "2",
                name: "fast",
                input: &fast_input,
            },
        ];

        executor
            .execute(
                &tools,
                &calls,
                &ToolContext::default(),
                &HookRegistry::new(),
                &mut InterruptState::new(),
            )
            .await
    }

    #[tokio::test]
    async fn sequential_preserves_request_order() {
        let blocks = run(SequentialToolExecutor, [20, 0]).await;
        let texts = result_texts(&blocks);
        assert!(texts[0].contains("slow"));
        assert!(texts[1].contains("fast"));
    }

    #[tokio::test]
    async fn concurrent_preserves_request_order_even_when_completion_differs() {
        // "fast" finishes first, but results must still come back in the order
        // the model asked for them.
        let blocks = run(ConcurrentToolExecutor, [30, 0]).await;
        let texts = result_texts(&blocks);
        assert!(texts[0].contains("slow"), "got {texts:?}");
        assert!(texts[1].contains("fast"), "got {texts:?}");
    }

    #[tokio::test]
    async fn concurrent_actually_overlaps() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let tools = registry(vec![
            Box::new(EchoTool {
                tool_name: "slow",
                order: order.clone(),
                delay_ms: 40,
            }),
            Box::new(EchoTool {
                tool_name: "fast",
                order: order.clone(),
                delay_ms: 0,
            }),
        ]);
        let a = json!({});
        let calls = vec![
            ToolCall {
                tool_use_id: "1",
                name: "slow",
                input: &a,
            },
            ToolCall {
                tool_use_id: "2",
                name: "fast",
                input: &a,
            },
        ];

        ConcurrentToolExecutor
            .execute(
                &tools,
                &calls,
                &ToolContext::default(),
                &HookRegistry::new(),
                &mut InterruptState::new(),
            )
            .await;

        assert_eq!(
            *order.lock().unwrap(),
            vec!["fast", "slow"],
            "the fast tool should complete first, proving overlap"
        );
    }

    #[tokio::test]
    async fn missing_tool_becomes_an_error_result_not_a_failure() {
        let tools: HashMap<String, Box<dyn Tool>> = HashMap::new();
        let input = json!({});
        let calls = vec![ToolCall {
            tool_use_id: "1",
            name: "nope",
            input: &input,
        }];

        for blocks in [
            SequentialToolExecutor
                .execute(
                    &tools,
                    &calls,
                    &ToolContext::default(),
                    &HookRegistry::new(),
                    &mut InterruptState::new(),
                )
                .await,
            ConcurrentToolExecutor
                .execute(
                    &tools,
                    &calls,
                    &ToolContext::default(),
                    &HookRegistry::new(),
                    &mut InterruptState::new(),
                )
                .await,
        ] {
            match &blocks[0] {
                ContentBlock::ToolResult { status, .. } => {
                    assert_eq!(*status, ToolResultStatus::Error);
                }
                other => panic!("expected a tool result, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn both_executors_retry_with_the_original_input() {
        for concurrent in [false, true] {
            let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));

            struct Recorder {
                seen: Arc<Mutex<Vec<Value>>>,
            }

            #[async_trait]
            impl Tool for Recorder {
                fn name(&self) -> &str {
                    "rec"
                }
                fn spec(&self) -> ToolSpec {
                    ToolSpec::new("rec", "rec", json!({"type": "object"}))
                }
                async fn invoke(
                    &self,
                    input: Value,
                    _ctx: &ToolContext,
                ) -> Result<ToolOutput, crate::error::StrandsError> {
                    self.seen.lock().unwrap().push(input);
                    Ok(ToolOutput::success(json!("ok")))
                }
            }

            let tools = registry(vec![Box::new(Recorder { seen: seen.clone() })]);
            let input = json!({"k": "v"});
            let calls = vec![ToolCall {
                tool_use_id: "1",
                name: "rec",
                input: &input,
            }];

            let retries = Arc::new(AtomicUsize::new(0));
            let mut hooks = HookRegistry::new();
            hooks.register(move |event: &mut HookEvent| {
                if let HookEvent::AfterToolCall(AfterToolCallEvent { retry, .. }) = event {
                    if retries.fetch_add(1, Ordering::SeqCst) == 0 {
                        *retry = true;
                    }
                }
            });

            if concurrent {
                ConcurrentToolExecutor
                    .execute(
                        &tools,
                        &calls,
                        &ToolContext::default(),
                        &hooks,
                        &mut InterruptState::new(),
                    )
                    .await;
            } else {
                SequentialToolExecutor
                    .execute(
                        &tools,
                        &calls,
                        &ToolContext::default(),
                        &hooks,
                        &mut InterruptState::new(),
                    )
                    .await;
            }

            let seen = seen.lock().unwrap();
            assert_eq!(seen.len(), 2, "concurrent={concurrent}");
            assert_eq!(seen[0], seen[1], "concurrent={concurrent}");
            assert_eq!(seen[1]["k"], "v");
        }
    }

    #[tokio::test]
    async fn a_cancelling_hook_skips_the_tool_in_both_executors() {
        for concurrent in [false, true] {
            let order = Arc::new(Mutex::new(Vec::new()));
            let tools = registry(vec![Box::new(EchoTool {
                tool_name: "t",
                order: order.clone(),
                delay_ms: 0,
            })]);
            let input = json!({});
            let calls = vec![ToolCall {
                tool_use_id: "1",
                name: "t",
                input: &input,
            }];

            let mut hooks = HookRegistry::new();
            hooks.register(|event: &mut HookEvent| {
                if let HookEvent::BeforeToolCall(BeforeToolCallEvent { cancel, .. }) = event {
                    *cancel = true;
                }
            });

            if concurrent {
                ConcurrentToolExecutor
                    .execute(
                        &tools,
                        &calls,
                        &ToolContext::default(),
                        &hooks,
                        &mut InterruptState::new(),
                    )
                    .await;
            } else {
                SequentialToolExecutor
                    .execute(
                        &tools,
                        &calls,
                        &ToolContext::default(),
                        &hooks,
                        &mut InterruptState::new(),
                    )
                    .await;
            }

            assert!(
                order.lock().unwrap().is_empty(),
                "cancelled tool must not run (concurrent={concurrent})"
            );
        }
    }

    #[tokio::test]
    async fn empty_batch_returns_no_results() {
        let tools: HashMap<String, Box<dyn Tool>> = HashMap::new();
        let blocks = SequentialToolExecutor
            .execute(
                &tools,
                &[],
                &ToolContext::default(),
                &HookRegistry::new(),
                &mut InterruptState::new(),
            )
            .await;
        assert!(blocks.is_empty());
    }
}
