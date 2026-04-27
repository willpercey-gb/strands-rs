use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use futures::StreamExt;
use serde_json::Value;
use tracing::{debug, warn};

use crate::agent::callback::CallbackHandler;
use crate::conversation::ConversationManager;
use crate::error::StrandsError;
use crate::hooks::events::*;
use crate::hooks::registry::HookRegistry;
use crate::model::Model;
use crate::tool::{Tool, ToolContext, ToolOutput};
use crate::types::content::{ContentBlock, ToolResultContent, ToolResultStatus};
use crate::types::message::{Message, Role};
use crate::types::streaming::{
    ContentBlockType, DeltaContent, StopReason, StreamEvent, Usage,
};

use super::result::AgentResult;

/// Accumulates streaming events into complete content blocks.
struct StreamAccumulator {
    blocks: Vec<ContentBlock>,
    active_text: Option<String>,
    active_tool: Option<PartialToolUse>,
}

struct PartialToolUse {
    tool_use_id: String,
    name: String,
    input_json: String,
}

impl StreamAccumulator {
    fn new() -> Self {
        Self {
            blocks: Vec::new(),
            active_text: None,
            active_tool: None,
        }
    }

    fn handle_event(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::ContentBlockStart { content_type, .. } => match content_type {
                ContentBlockType::Text => {
                    self.active_text = Some(String::new());
                }
                ContentBlockType::ToolUse { tool_use_id, name } => {
                    self.active_tool = Some(PartialToolUse {
                        tool_use_id: tool_use_id.clone(),
                        name: name.clone(),
                        input_json: String::new(),
                    });
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
            },
            StreamEvent::ContentBlockStop { .. } => {
                if let Some(text) = self.active_text.take() {
                    if !text.is_empty() {
                        self.blocks.push(ContentBlock::Text { text });
                    }
                }
                if let Some(tool) = self.active_tool.take() {
                    let input = serde_json::from_str(&tool.input_json)
                        .unwrap_or(Value::Object(serde_json::Map::new()));
                    self.blocks.push(ContentBlock::ToolUse {
                        tool_use_id: tool.tool_use_id,
                        name: tool.name,
                        input,
                    });
                }
            }
            _ => {}
        }
    }

    fn finalize(self) -> Vec<ContentBlock> {
        let mut blocks = self.blocks;
        if let Some(text) = self.active_text {
            if !text.is_empty() {
                blocks.push(ContentBlock::Text { text });
            }
        }
        if let Some(tool) = self.active_tool {
            let input = serde_json::from_str(&tool.input_json)
                .unwrap_or(Value::Object(serde_json::Map::new()));
            blocks.push(ContentBlock::ToolUse {
                tool_use_id: tool.tool_use_id,
                name: tool.name,
                input,
            });
        }
        blocks
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
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 500,
            backoff_multiplier: 2.0,
            max_backoff_ms: 30_000,
        }
    }
}

/// Run the ReAct agent loop.
///
/// Takes individual fields to avoid borrow conflicts on Agent.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_loop(
    model: &dyn Model,
    tools: &HashMap<String, Box<dyn Tool>>,
    messages: &mut Vec<Message>,
    system_prompt: Option<&str>,
    conversation_manager: &dyn ConversationManager,
    hooks: &HookRegistry,
    callback_handler: Option<&dyn CallbackHandler>,
    cancel: &Arc<AtomicBool>,
    max_cycles: usize,
    retry_config: &RetryConfig,
    invocation_state: &mut Value,
    concurrent_tools: bool,
) -> crate::Result<AgentResult> {
    let tool_specs: Vec<_> = tools.values().map(|t| t.spec()).collect();
    let tool_ctx = ToolContext {
        state: invocation_state.clone(),
    };

    let mut total_usage = Usage::default();
    #[allow(unused_assignments)]
    let mut stop_reason = StopReason::EndTurn;
    #[allow(unused_assignments)]
    let mut last_assistant_message = None::<Message>;
    let mut cycle = 0;

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

        if cancel.load(Ordering::Relaxed) {
            return Err(StrandsError::Cancelled);
        }

        // Reduce context before calling model
        conversation_manager.reduce_context(messages, system_prompt).await?;

        // Model call with retry loop
        let (content_blocks, model_stop_reason, cycle_usage) = call_model_with_retry(
            model, messages, system_prompt, &tool_specs, hooks, callback_handler, cancel, cycle,
            retry_config,
        )
        .await?;

        // Accumulate usage
        total_usage.input_tokens = Some(
            total_usage.input_tokens.unwrap_or(0) + cycle_usage.input_tokens.unwrap_or(0),
        );
        total_usage.output_tokens = Some(
            total_usage.output_tokens.unwrap_or(0) + cycle_usage.output_tokens.unwrap_or(0),
        );
        stop_reason = model_stop_reason;

        // Build and append assistant message
        let assistant_msg = Message::assistant(content_blocks);
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
                // Remove the assistant message we just added and retry
                messages.pop();
                debug!(cycle, "Hook requested model retry");
                continue;
            }
        }

        hooks.dispatch(&mut HookEvent::MessageAdded {
            message: assistant_msg.clone(),
        });

        cycle += 1;

        // Check if we should stop or execute tools
        match stop_reason {
            StopReason::EndTurn
            | StopReason::MaxTokens
            | StopReason::Cancelled
            | StopReason::ContentFiltered
            | StopReason::GuardrailIntervention => {
                break;
            }
            StopReason::ToolUse => {
                let tool_uses = assistant_msg.tool_uses();
                let tool_results = if concurrent_tools {
                    execute_tools_concurrent(tools, &tool_uses, &tool_ctx, hooks).await
                } else {
                    execute_tools_sequential(tools, &tool_uses, &tool_ctx, hooks).await
                };

                let tool_result_msg = Message {
                    role: Role::User,
                    content: tool_results,
                };
                messages.push(tool_result_msg.clone());

                hooks.dispatch(&mut HookEvent::MessageAdded {
                    message: tool_result_msg,
                });
            }
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

    Ok(AgentResult {
        stop_reason,
        message: last_assistant_message.unwrap_or_else(|| Message::assistant(vec![])),
        usage: total_usage,
        cycle_count: cycle,
    })
}

/// Call the model with exponential backoff retry on error.
#[allow(clippy::too_many_arguments)]
async fn call_model_with_retry(
    model: &dyn Model,
    messages: &[Message],
    system_prompt: Option<&str>,
    tool_specs: &[crate::types::tools::ToolSpec],
    hooks: &HookRegistry,
    callback_handler: Option<&dyn CallbackHandler>,
    cancel: &Arc<AtomicBool>,
    cycle: usize,
    retry_config: &RetryConfig,
) -> crate::Result<(Vec<ContentBlock>, StopReason, Usage)> {
    let mut attempt = 0;
    let mut backoff_ms = retry_config.initial_backoff_ms;

    loop {
        hooks.dispatch(&mut HookEvent::BeforeModelCall { cycle });
        debug!(cycle, attempt, "Calling model");

        match try_model_call(model, messages, system_prompt, tool_specs, callback_handler, cancel)
            .await
        {
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
    system_prompt: Option<&str>,
    tool_specs: &[crate::types::tools::ToolSpec],
    callback_handler: Option<&dyn CallbackHandler>,
    cancel: &Arc<AtomicBool>,
) -> crate::Result<(Vec<ContentBlock>, StopReason, Usage)> {
    let mut stream = model.stream(messages, system_prompt, tool_specs).await?;
    let mut accumulator = StreamAccumulator::new();
    let mut stop_reason = StopReason::EndTurn;
    let mut usage = Usage::default();

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
            StreamEvent::Metadata { usage: u } => {
                usage = u.clone();
            }
            _ => {}
        }

        accumulator.handle_event(&event);
    }

    Ok((accumulator.finalize(), stop_reason, usage))
}

/// Execute a single tool, handling not-found gracefully.
async fn execute_tool(
    tools: &HashMap<String, Box<dyn Tool>>,
    tool_name: &str,
    input: &Value,
    tool_ctx: &ToolContext,
) -> ToolOutput {
    match tools.get(tool_name) {
        Some(tool) => {
            debug!(tool_name, "Invoking tool");
            match tool.invoke(input.clone(), tool_ctx).await {
                Ok(output) => output,
                Err(e) => {
                    warn!(tool_name, error = %e, "Tool execution failed");
                    ToolOutput::error(e.to_string())
                }
            }
        }
        None => {
            warn!(tool_name, "Tool not found");
            ToolOutput::error(format!("Tool not found: {tool_name}"))
        }
    }
}

/// Execute tool calls sequentially with hook support.
async fn execute_tools_sequential(
    tools: &HashMap<String, Box<dyn Tool>>,
    tool_uses: &[(&str, &str, &Value)],
    tool_ctx: &ToolContext,
    hooks: &HookRegistry,
) -> Vec<ContentBlock> {
    let mut results = Vec::new();

    for (tool_use_id, tool_name, input) in tool_uses {
        let mut before_tool = HookEvent::BeforeToolCall(BeforeToolCallEvent {
            tool_name: tool_name.to_string(),
            input: (*input).clone(),
            cancel: false,
        });
        hooks.dispatch(&mut before_tool);

        let cancelled = matches!(
            before_tool,
            HookEvent::BeforeToolCall(BeforeToolCallEvent { cancel: true, .. })
        );

        let output = if cancelled {
            debug!(tool_name, "Tool call cancelled by hook");
            ToolOutput::error("Tool call cancelled")
        } else {
            execute_tool(tools, tool_name, input, tool_ctx).await
        };

        let mut after_tool = HookEvent::AfterToolCall(AfterToolCallEvent {
            tool_name: tool_name.to_string(),
            is_error: output.is_error,
            retry: false,
        });
        hooks.dispatch(&mut after_tool);

        let final_output =
            if let HookEvent::AfterToolCall(AfterToolCallEvent { retry: true, .. }) = after_tool {
                debug!(tool_name, "Hook requested tool retry");
                execute_tool(tools, tool_name, input, tool_ctx).await
            } else {
                output
            };

        results.push(tool_output_to_content_block(tool_use_id, &final_output));
    }

    results
}

/// Execute tool calls concurrently with hook support.
async fn execute_tools_concurrent(
    tools: &HashMap<String, Box<dyn Tool>>,
    tool_uses: &[(&str, &str, &Value)],
    tool_ctx: &ToolContext,
    hooks: &HookRegistry,
) -> Vec<ContentBlock> {
    // Fire BeforeToolCall hooks sequentially (they may cancel)
    let mut tasks: Vec<(&str, &str, &Value, bool)> = Vec::new();
    for (tool_use_id, tool_name, input) in tool_uses {
        let mut before_tool = HookEvent::BeforeToolCall(BeforeToolCallEvent {
            tool_name: tool_name.to_string(),
            input: (*input).clone(),
            cancel: false,
        });
        hooks.dispatch(&mut before_tool);

        let cancelled = matches!(
            before_tool,
            HookEvent::BeforeToolCall(BeforeToolCallEvent { cancel: true, .. })
        );
        tasks.push((tool_use_id, tool_name, input, cancelled));
    }

    // Execute non-cancelled tools concurrently
    let futures: Vec<_> = tasks
        .iter()
        .map(|(tool_use_id, tool_name, input, cancelled)| async move {
            let output = if *cancelled {
                ToolOutput::error("Tool call cancelled")
            } else {
                execute_tool(tools, tool_name, input, tool_ctx).await
            };
            (*tool_use_id, tool_name.to_string(), output)
        })
        .collect();

    let outputs = futures::future::join_all(futures).await;

    // Fire AfterToolCall hooks and build results
    let mut results = Vec::new();
    for (tool_use_id, tool_name, output) in outputs {
        let mut after_tool = HookEvent::AfterToolCall(AfterToolCallEvent {
            tool_name: tool_name.clone(),
            is_error: output.is_error,
            retry: false,
        });
        hooks.dispatch(&mut after_tool);

        let final_output =
            if let HookEvent::AfterToolCall(AfterToolCallEvent { retry: true, .. }) = after_tool {
                execute_tool(tools, &tool_name, &Value::Null, tool_ctx).await
            } else {
                output
            };

        results.push(tool_output_to_content_block(tool_use_id, &final_output));
    }

    results
}

/// Convert a tool output to a ContentBlock::ToolResult.
fn tool_output_to_content_block(tool_use_id: &str, output: &ToolOutput) -> ContentBlock {
    let content_text = match &output.content {
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
        content: vec![ToolResultContent::Text { text: content_text }],
    }
}
