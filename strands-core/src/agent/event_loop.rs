use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use futures::StreamExt;
use serde_json::Value;
use tracing::{debug, warn};

use crate::agent::callback::CallbackHandler;
use crate::conversation::{ConversationManager, ReduceContext};
use crate::error::StrandsError;
use crate::hooks::events::*;
use crate::hooks::registry::HookRegistry;
use crate::interrupt::InterruptState;
use crate::model::Model;
use crate::tool::{Tool, ToolCall, ToolContext, ToolExecutor, ToolOutput};
use crate::types::content::{ContentBlock, ReasoningContent, SystemPrompt, ToolResultStatus};
use crate::types::message::{Message, Role};
use crate::types::streaming::{
    ContentBlockType, DeltaContent, Metrics, StopReason, StreamEvent, Usage,
};

use super::limits::Limits;
use super::result::AgentResult;

/// Accumulates streaming events into complete content blocks.
struct StreamAccumulator {
    blocks: Vec<ContentBlock>,
    active_text: Option<String>,
    active_tool: Option<PartialToolUse>,
    active_reasoning: Option<ReasoningContent>,
}

struct PartialToolUse {
    tool_use_id: String,
    name: String,
    input_json: String,
    /// Signature tying the model's reasoning to this call. Providers that
    /// issue one reject the tool call unless it is echoed back, so it must be
    /// preserved even when no reasoning text accompanied it.
    reasoning_signature: Option<String>,
}

impl PartialToolUse {
    /// Parse the accumulated input fragments into a JSON value.
    ///
    /// Providers occasionally emit truncated or malformed tool input — most
    /// often when a response is cut short mid-stream. Defaulting to an empty
    /// object keeps the loop alive, but doing so silently loses the model's
    /// intent, so log the raw fragment (truncated) to make it diagnosable.
    fn parse_input(&self) -> Value {
        match serde_json::from_str(&self.input_json) {
            Ok(value) => value,
            Err(e) => {
                const MAX_LOGGED: usize = 200;
                let raw: String = self.input_json.chars().take(MAX_LOGGED).collect();
                warn!(
                    tool_name = %self.name,
                    tool_use_id = %self.tool_use_id,
                    raw_input = %raw,
                    error = %e,
                    "Failed to parse tool input JSON, defaulting to empty object"
                );
                Value::Object(serde_json::Map::new())
            }
        }
    }

    /// Flush this partial tool use into content blocks.
    ///
    /// When the provider supplied a reasoning signature, it is emitted as its
    /// own `Reasoning` block ahead of the tool use — a reasoning block with
    /// empty text but a non-empty signature is valid and must not be dropped.
    fn into_content_blocks(self) -> Vec<ContentBlock> {
        let input = self.parse_input();
        let mut blocks = Vec::new();

        if let Some(signature) = self.reasoning_signature {
            blocks.push(ContentBlock::Reasoning(ReasoningContent {
                text: None,
                signature: Some(signature),
                redacted_content: None,
            }));
        }

        blocks.push(ContentBlock::ToolUse {
            tool_use_id: self.tool_use_id,
            name: self.name,
            input,
        });
        blocks
    }
}

impl StreamAccumulator {
    fn new() -> Self {
        Self {
            blocks: Vec::new(),
            active_text: None,
            active_tool: None,
            active_reasoning: None,
        }
    }

    fn handle_event(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::ContentBlockStart { content_type, .. } => match content_type {
                ContentBlockType::Text => {
                    self.active_text = Some(String::new());
                }
                ContentBlockType::ToolUse {
                    tool_use_id,
                    name,
                    reasoning_signature,
                } => {
                    self.active_tool = Some(PartialToolUse {
                        tool_use_id: tool_use_id.clone(),
                        name: name.clone(),
                        input_json: String::new(),
                        reasoning_signature: reasoning_signature.clone(),
                    });
                }
                ContentBlockType::Reasoning => {
                    self.active_reasoning = Some(ReasoningContent::default());
                }
            },
            StreamEvent::ContentBlockDelta { delta, .. } => match delta {
                DeltaContent::TextDelta(text) => {
                    if let Some(ref mut buf) = self.active_text {
                        buf.push_str(text);
                    }
                }
                DeltaContent::ToolInputDelta(fragment) => {
                    if let Some(ref mut tool) = self.active_tool {
                        tool.input_json.push_str(fragment);
                    }
                }
                DeltaContent::ReasoningDelta(text) => {
                    let reasoning = self.active_reasoning.get_or_insert_with(Default::default);
                    reasoning.text.get_or_insert_with(String::new).push_str(text);
                }
                DeltaContent::ReasoningSignature(signature) => {
                    // Some providers deliver the signature as its own delta
                    // rather than on the block start.
                    let reasoning = self.active_reasoning.get_or_insert_with(Default::default);
                    reasoning.signature = Some(signature.clone());
                }
            },
            StreamEvent::ContentBlockStop { .. } => {
                self.flush_active();
            }
            _ => {}
        }
    }

    /// Close out whatever block is currently open.
    fn flush_active(&mut self) {
        if let Some(text) = self.active_text.take() {
            if !text.is_empty() {
                self.blocks.push(ContentBlock::Text { text });
            }
        }
        if let Some(reasoning) = self.active_reasoning.take() {
            // Keep a reasoning block whose text is empty but whose signature
            // is set: dropping it invalidates the tool call it belongs to.
            let has_content = reasoning.text.as_deref().is_some_and(|t| !t.is_empty())
                || reasoning.signature.is_some()
                || reasoning.redacted_content.is_some();
            if has_content {
                self.blocks.push(ContentBlock::Reasoning(reasoning));
            }
        }
        if let Some(tool) = self.active_tool.take() {
            self.blocks.extend(tool.into_content_blocks());
        }
    }

    fn finalize(mut self) -> Vec<ContentBlock> {
        self.flush_active();
        self.blocks
    }
}

/// Configuration for model call retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retries per model call.
    pub max_retries: usize,
    /// Initial backoff delay in milliseconds.
    pub initial_backoff_ms: u64,
    /// Backoff multiplier per retry.
    pub backoff_multiplier: f64,
    /// Maximum backoff delay in milliseconds.
    pub max_backoff_ms: u64,
    /// Maximum number of consecutive hook-requested model retries.
    ///
    /// Hook retries do not advance the cycle counter, so without a separate
    /// bound a hook that always sets `retry` would loop forever.
    pub max_hook_retries: usize,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 500,
            backoff_multiplier: 2.0,
            max_backoff_ms: 30_000,
            max_hook_retries: 3,
        }
    }
}

/// Run the ReAct agent loop.
///
/// Takes individual fields to avoid borrow conflicts on Agent.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_loop(
    model: std::sync::Arc<dyn Model>,
    tools: &HashMap<String, Box<dyn Tool>>,
    messages: &mut Vec<Message>,
    system_prompt: Option<&SystemPrompt>,
    conversation_manager: &dyn ConversationManager,
    hooks: &HookRegistry,
    callback_handler: Option<&dyn CallbackHandler>,
    cancel: &Arc<AtomicBool>,
    max_cycles: usize,
    retry_config: &RetryConfig,
    invocation_state: &mut Value,
    tool_executor: &dyn ToolExecutor,
    limits: &Limits,
    interrupts: &mut InterruptState,
    model_middleware: &crate::middleware::MiddlewareChain<
        crate::middleware::InvokeModelContext,
        crate::middleware::stages::InvokeModelResult,
    >,
) -> crate::Result<AgentResult> {
    let tool_specs: Vec<_> = tools.values().map(|t| t.spec()).collect();
    let tool_ctx = ToolContext {
        state: invocation_state.clone(),
    };

    let mut total_usage = Usage::default();
    let mut total_metrics = Metrics::default();
    let mut collector = crate::telemetry::MetricsCollector::new();
    #[allow(unused_assignments)]
    let mut stop_reason = StopReason::EndTurn;
    #[allow(unused_assignments)]
    let mut last_assistant_message = None::<Message>;
    let mut cycle = 0;
    let mut hook_retries = 0usize;
    let mut overflow_recovered = false;

    // BeforeInvocation — hooks can override messages
    let mut before_event = HookEvent::BeforeInvocation(BeforeInvocationEvent {
        messages: messages.clone(),
        override_messages: None,
    });
    hooks.dispatch(&mut before_event);
    if let HookEvent::BeforeInvocation(ref evt) = before_event {
        if let Some(ref override_msgs) = evt.override_messages {
            *messages = override_msgs.clone();
        }
    }

    loop {
        if cycle >= max_cycles {
            return Err(StrandsError::MaxCycles(max_cycles));
        }

        // Budget caps are checked here, at a turn boundary, so any tools the
        // previous turn requested have already produced results and the history
        // stays re-invokable.
        if let Some(reason) = limits.exceeded(cycle, &total_usage) {
            debug!(?reason, cycle, "Invocation limit reached");
            stop_reason = reason;
            break;
        }

        if cancel.load(Ordering::Relaxed) {
            return Err(StrandsError::Cancelled);
        }

        // Reduce context before calling the model.
        //
        // When the manager wants proactive compression, measure how full the
        // window actually is first. `estimate_utilization` returns None when
        // the model's limit is unknown, and that stays None all the way to the
        // manager — "unknown" must not read as headroom.
        let utilization = match conversation_manager.proactive_compression() {
            Some(_) => {
                let counted = model
                    .count_tokens(messages, system_prompt, &tool_specs)
                    .await
                    .unwrap_or_else(|e| {
                        // Token counting is an optimisation; a provider that
                        // refuses should not fail the invocation.
                        debug!(error = %e, "Token counting failed; skipping proactive compression");
                        0
                    });
                if counted == 0 {
                    None
                } else {
                    model.estimate_utilization(counted)
                }
            }
            None => None,
        };

        conversation_manager
            .reduce_context(
                messages,
                ReduceContext::routine(system_prompt).with_utilization(utilization),
            )
            .await?;

        // Model call with retry loop
        // A context-window overflow is recoverable: reduce and try once more.
        // Without this the manager's overflow path is unreachable, and the
        // invocation fails on something trimming would have fixed.
        let outcome = match call_model_with_retry(
            &model,
            messages,
            system_prompt,
            &tool_specs,
            hooks,
            callback_handler,
            cancel,
            cycle,
            retry_config,
            model_middleware,
        )
        .await
        {
            Err(e) if e.is_context_overflow() && !overflow_recovered => {
                debug!(error = %e, "Context overflow; reducing and retrying once");
                overflow_recovered = true;
                conversation_manager
                    .reduce_context(messages, ReduceContext::overflow(system_prompt))
                    .await?;
                call_model_with_retry(
                    &model,
                    messages,
                    system_prompt,
                    &tool_specs,
                    hooks,
                    callback_handler,
                    cancel,
                    cycle,
                    retry_config,
                    model_middleware,
                )
                .await?
            }
            other => other?,
        };

        // A successful call means the reduced history fits; allow one more
        // recovery if the conversation grows past the limit again later.
        overflow_recovered = false;

        let crate::middleware::ModelCallOutcome {
            content: content_blocks,
            stop_reason: model_stop_reason,
            usage: cycle_usage,
            metrics: cycle_metrics,
        } = outcome;

        // Accumulate usage and metrics across cycles.
        total_usage.accumulate(&cycle_usage);
        total_metrics.accumulate(&cycle_metrics);
        collector.record_cycle(
            &cycle_usage,
            &cycle_metrics,
            content_blocks.iter().filter(|b| b.is_tool_use()).count(),
        );
        stop_reason = model_stop_reason;

        // Build and append assistant message, stamped with a durable id and
        // the usage/metrics of the call that produced it.
        let mut assistant_msg = Message::assistant(content_blocks);
        assistant_msg.ensure_tracking_id();
        assistant_msg.metadata = Some(crate::types::message::MessageMetadata {
            usage: Some(cycle_usage.clone()),
            metrics: Some(cycle_metrics.clone()),
            ..Default::default()
        });
        messages.push(assistant_msg.clone());
        last_assistant_message = Some(assistant_msg.clone());

        // AfterModelCall — hooks can request retry
        let mut after_model = HookEvent::AfterModelCall(AfterModelCallEvent {
            stop_reason,
            cycle,
            retry: false,
        });
        hooks.dispatch(&mut after_model);
        if let HookEvent::AfterModelCall(ref evt) = after_model {
            if evt.retry {
                // A hook-requested retry does not advance `cycle`, so it is
                // not bounded by `max_cycles`. Count them separately or a
                // hook that unconditionally sets `retry` spins forever.
                hook_retries += 1;
                if hook_retries > retry_config.max_hook_retries {
                    warn!(
                        cycle,
                        hook_retries,
                        max = retry_config.max_hook_retries,
                        "Hook retry limit exceeded; proceeding with the last response"
                    );
                } else {
                    // Remove the assistant message we just added and retry
                    messages.pop();
                    debug!(cycle, hook_retries, "Hook requested model retry");
                    continue;
                }
            }
        }
        hook_retries = 0;

        hooks.dispatch(&mut HookEvent::MessageAdded {
            message: assistant_msg.clone(),
        });

        cycle += 1;

        // Check if we should stop or execute tools
        // Every reason other than ToolUse ends the loop. Branching on
        // `is_terminal` rather than enumerating variants means a newly added
        // StopReason stops the agent rather than silently looping forever.
        if stop_reason.is_terminal() {
            break;
        }

        let tool_uses = assistant_msg.tool_uses();

        // BeforeTools — one decision point for the whole batch, ahead of any
        // per-tool hook.
        let mut before_tools = HookEvent::BeforeTools(BeforeToolsEvent {
            tool_calls: tool_uses
                .iter()
                .map(|(id, name, _)| (id.to_string(), name.to_string()))
                .collect(),
            cancel: false,
        });
        hooks.dispatch(&mut before_tools);
        let batch_cancelled = matches!(
            before_tools,
            HookEvent::BeforeTools(BeforeToolsEvent { cancel: true, .. })
        );

        let tool_results = if batch_cancelled {
            debug!("Tool batch cancelled by hook");
            tool_uses
                .iter()
                .map(|(id, _, _)| {
                    crate::tool::executor::to_result_block(
                        id,
                        &ToolOutput::error("Tool batch cancelled"),
                    )
                })
                .collect()
        } else {
            let calls: Vec<ToolCall<'_>> = tool_uses
                .iter()
                .map(|(id, name, input)| ToolCall {
                    tool_use_id: id,
                    name,
                    input,
                })
                .collect();
            tool_executor
                .execute(tools, &calls, &tool_ctx, hooks, interrupts)
                .await
        };

        // AfterTools — dispatched in reverse hook order.
        let mut after_tools = HookEvent::AfterTools(AfterToolsEvent {
            results: tool_results
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolResult {
                        tool_use_id,
                        status,
                        ..
                    } => Some((
                        tool_use_id.clone(),
                        *status == ToolResultStatus::Error,
                    )),
                    _ => None,
                })
                .collect(),
            end_turn: false,
        });
        hooks.dispatch(&mut after_tools);
        let end_turn = matches!(
            after_tools,
            HookEvent::AfterTools(AfterToolsEvent { end_turn: true, .. })
        );

        let mut tool_result_msg = Message::new(Role::User, tool_results);
        tool_result_msg.ensure_tracking_id();
        messages.push(tool_result_msg.clone());

        hooks.dispatch(&mut HookEvent::MessageAdded {
            message: tool_result_msg,
        });

        if end_turn {
            debug!("Hook requested end of turn after tool batch");
            stop_reason = StopReason::EndTurn;
            break;
        }

        // A hook that paused for human input stops the loop here, after the
        // tool results are already in the history — so resuming continues from
        // a valid conversation rather than a severed tool pair.
        if interrupts.has_pending() {
            debug!(
                pending = interrupts.pending().len(),
                "Pausing for human input"
            );
            stop_reason = StopReason::Interrupt;
            break;
        }
    }

    // AfterInvocation — hooks can request resume
    let mut after_event = HookEvent::AfterInvocation(AfterInvocationEvent {
        stop_reason,
        cycle_count: cycle,
        resume: false,
    });
    hooks.dispatch(&mut after_event);

    // Update invocation state from tool context
    *invocation_state = tool_ctx.state;

    collector.set_stop_reason(stop_reason);

    Ok(AgentResult {
        telemetry: collector.finish(),
        interrupts: interrupts.pending(),
        stop_reason,
        message: last_assistant_message.unwrap_or_else(|| Message::assistant(vec![])),
        usage: total_usage,
        metrics: total_metrics,
        cycle_count: cycle,
    })
}

/// The operation at the end of the model middleware chain: the real call.
struct ModelCallTerminal<'a> {
    callback_handler: Option<&'a dyn CallbackHandler>,
    cancel: &'a Arc<AtomicBool>,
}

impl crate::middleware::Terminal<
        crate::middleware::InvokeModelContext,
        crate::middleware::stages::InvokeModelResult,
    > for ModelCallTerminal<'_>
{
    fn call<'a>(
        &'a self,
        ctx: crate::middleware::InvokeModelContext,
    ) -> futures::future::BoxFuture<'a, crate::middleware::stages::InvokeModelResult> {
        Box::pin(async move {
            try_model_call(
                ctx.model.as_ref(),
                &ctx.messages,
                ctx.system_prompt.as_ref(),
                &ctx.tool_specs,
                self.callback_handler,
                self.cancel,
            )
            .await
        })
    }
}

/// Call the model with exponential backoff retry on error.
#[allow(clippy::too_many_arguments)]
async fn call_model_with_retry(
    model: &std::sync::Arc<dyn Model>,
    messages: &[Message],
    system_prompt: Option<&SystemPrompt>,
    tool_specs: &[crate::types::tools::ToolSpec],
    hooks: &HookRegistry,
    callback_handler: Option<&dyn CallbackHandler>,
    cancel: &Arc<AtomicBool>,
    cycle: usize,
    retry_config: &RetryConfig,
    model_middleware: &crate::middleware::MiddlewareChain<
        crate::middleware::InvokeModelContext,
        crate::middleware::stages::InvokeModelResult,
    >,
) -> crate::Result<crate::middleware::ModelCallOutcome> {
    let mut attempt = 0;
    let mut backoff_ms = retry_config.initial_backoff_ms;

    loop {
        hooks.dispatch(&mut HookEvent::BeforeModelCall { cycle });
        debug!(cycle, attempt, "Calling model");

        let ctx = crate::middleware::InvokeModelContext {
            messages: crate::types::message::messages_for_model(messages),
            system_prompt: system_prompt.cloned(),
            tool_specs: tool_specs.to_vec(),
            model: model.clone(),
            cycle,
        };
        let terminal = ModelCallTerminal {
            callback_handler,
            cancel,
        };

        match model_middleware.run(ctx, &terminal).await {
            Ok(result) => return Ok(result),
            Err(e) => {
                // Quota / auth failures are guaranteed to fail again
                // — short-circuit so the user pays once, not 4×.
                if matches!(e, crate::error::StrandsError::Quota(_)) {
                    warn!(cycle, error = %e, "Non-retryable provider error; surfacing immediately");
                    return Err(e);
                }
                attempt += 1;
                if attempt > retry_config.max_retries {
                    return Err(e);
                }
                warn!(
                    cycle,
                    attempt,
                    max_retries = retry_config.max_retries,
                    backoff_ms,
                    error = %e,
                    "Model call failed, retrying"
                );
                tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms as f64 * retry_config.backoff_multiplier) as u64;
                backoff_ms = backoff_ms.min(retry_config.max_backoff_ms);
            }
        }
    }
}

/// Single attempt to call the model and consume the stream.
async fn try_model_call(
    model: &dyn Model,
    messages: &[Message],
    system_prompt: Option<&SystemPrompt>,
    tool_specs: &[crate::types::tools::ToolSpec],
    callback_handler: Option<&dyn CallbackHandler>,
    cancel: &Arc<AtomicBool>,
) -> crate::Result<crate::middleware::ModelCallOutcome> {
    let mut stream = model.stream(messages, system_prompt, tool_specs).await?;
    let mut accumulator = StreamAccumulator::new();
    let mut stop_reason = StopReason::EndTurn;
    let mut usage = Usage::default();
    let mut metrics = Metrics::default();

    while let Some(event_result) = stream.next().await {
        if cancel.load(Ordering::Relaxed) {
            return Err(StrandsError::Cancelled);
        }

        let event = event_result?;

        // Fire callback handler for real-time streaming
        if let Some(handler) = callback_handler {
            handler.on_stream_event(&event);
        }

        match &event {
            StreamEvent::MessageStop { stop_reason: sr } => {
                stop_reason = *sr;
            }
            StreamEvent::Metadata {
                usage: u,
                metrics: m,
            } => {
                usage = u.clone();
                metrics = m.clone();
            }
            _ => {}
        }

        accumulator.handle_event(&event);
    }

    Ok(crate::middleware::ModelCallOutcome {
        content: accumulator.finalize(),
        stop_reason,
        usage,
        metrics,
    })
}

