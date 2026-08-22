use strands_core::types::content::SystemPrompt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;
use serde_json::json;
use strands_core::model::{Model, ModelStream};
use strands_core::types::message::Message;
use strands_core::types::streaming::*;
use strands_core::types::tools::ToolSpec;
use strands_core::*;

// ---------------------------------------------------------------------------
// Mock model that returns a simple text response
// ---------------------------------------------------------------------------

struct MockTextModel {
    response: String,
}

#[async_trait]
impl Model for MockTextModel {
    async fn stream(
        &self,
        _messages: &[Message],
        _system_prompt: Option<&SystemPrompt>,
        _tool_specs: &[ToolSpec],
    ) -> Result<ModelStream> {
        let events = vec![
            Ok(StreamEvent::MessageStart {
                role: Role::Assistant,
            }),
            Ok(StreamEvent::ContentBlockStart {
                index: 0,
                content_type: ContentBlockType::Text,
            }),
            Ok(StreamEvent::ContentBlockDelta {
                index: 0,
                delta: DeltaContent::TextDelta(self.response.clone()),
            }),
            Ok(StreamEvent::ContentBlockStop { index: 0 }),
            Ok(StreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
            }),
        ];
        Ok(Box::pin(stream::iter(events)))
    }
}

// ---------------------------------------------------------------------------
// Mock model that calls a tool then returns text
// ---------------------------------------------------------------------------

struct MockToolModel {
    call_count: Arc<AtomicUsize>,
}

#[async_trait]
impl Model for MockToolModel {
    async fn stream(
        &self,
        _messages: &[Message],
        _system_prompt: Option<&SystemPrompt>,
        _tool_specs: &[ToolSpec],
    ) -> Result<ModelStream> {
        let count = self.call_count.fetch_add(1, Ordering::SeqCst);

        if count == 0 {
            let events = vec![
                Ok(StreamEvent::MessageStart {
                    role: Role::Assistant,
                }),
                Ok(StreamEvent::ContentBlockStart {
                    index: 0,
                    content_type: ContentBlockType::ToolUse {
                        tool_use_id: "call_1".to_string(),
                        name: "greet".to_string(),
                        reasoning_signature: None,
                    },
                }),
                Ok(StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: DeltaContent::ToolInputDelta(r#"{"name":"World"}"#.to_string()),
                }),
                Ok(StreamEvent::ContentBlockStop { index: 0 }),
                Ok(StreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                }),
            ];
            Ok(Box::pin(stream::iter(events)))
        } else {
            let events = vec![
                Ok(StreamEvent::MessageStart {
                    role: Role::Assistant,
                }),
                Ok(StreamEvent::ContentBlockStart {
                    index: 0,
                    content_type: ContentBlockType::Text,
                }),
                Ok(StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: DeltaContent::TextDelta("The greeting is: Hello, World!".to_string()),
                }),
                Ok(StreamEvent::ContentBlockStop { index: 0 }),
                Ok(StreamEvent::MessageStop {
                    stop_reason: StopReason::EndTurn,
                }),
            ];
            Ok(Box::pin(stream::iter(events)))
        }
    }
}

// ---------------------------------------------------------------------------
// Simple tool
// ---------------------------------------------------------------------------

struct GreetTool;

#[async_trait]
impl Tool for GreetTool {
    fn name(&self) -> &str {
        "greet"
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec::new("greet".to_string(), "Greet someone by name".to_string(), json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" }
                },
                "required": ["name"]
            }))
    }

    async fn invoke(
        &self,
        input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput> {
        let name = input["name"].as_str().unwrap_or("stranger");
        Ok(ToolOutput::success(json!(format!("Hello, {name}!"))))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_simple_text_response() {
    let mut agent = Agent::builder()
        .model(MockTextModel {
            response: "Hello from the agent!".to_string(),
        })
        .system_prompt("You are helpful.")
        .build()
        .unwrap();

    let result = agent.prompt("Hi").await.unwrap();

    assert_eq!(result.text(), "Hello from the agent!");
    assert_eq!(result.stop_reason, StopReason::EndTurn);
    assert_eq!(result.cycle_count, 1);
}

#[tokio::test]
async fn test_tool_execution() {
    let call_count = Arc::new(AtomicUsize::new(0));

    let mut agent = Agent::builder()
        .model(MockToolModel {
            call_count: call_count.clone(),
        })
        .tool(GreetTool)
        .build()
        .unwrap();

    let result = agent.prompt("Greet the world").await.unwrap();

    assert_eq!(result.text(), "The greeting is: Hello, World!");
    assert_eq!(result.cycle_count, 2);
    assert_eq!(call_count.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn test_conversation_history() {
    let mut agent = Agent::builder()
        .model(MockTextModel {
            response: "Response 1".to_string(),
        })
        .build()
        .unwrap();

    agent.prompt("Message 1").await.unwrap();

    assert_eq!(agent.messages().len(), 2);
    assert_eq!(agent.messages()[0].role, Role::User);
    assert_eq!(agent.messages()[1].role, Role::Assistant);
}

#[tokio::test]
async fn test_fn_tool() {
    let tool = FnTool::new(
        "add",
        "Add two numbers",
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer" },
                "b": { "type": "integer" }
            },
            "required": ["a", "b"]
        }),
        |input: serde_json::Value, _ctx: &ToolContext| async move {
            let a = input["a"].as_i64().unwrap_or(0);
            let b = input["b"].as_i64().unwrap_or(0);
            Ok(ToolOutput::success(json!(a + b)))
        },
    );

    assert_eq!(tool.name(), "add");

    let result = tool
        .invoke(json!({"a": 3, "b": 4}), &ToolContext::default())
        .await
        .unwrap();
    assert_eq!(result.content, json!(7));
    assert!(!result.is_error);
}

#[tokio::test]
async fn test_sliding_window() {
    use strands_core::conversation::{ReduceContext, SlidingWindowConversationManager};

    let cm = SlidingWindowConversationManager::new(3);
    let mut messages = vec![
        Message::user("msg 1"),
        Message::assistant(vec![]),
        Message::user("msg 2"),
        Message::assistant(vec![]),
        Message::user("msg 3"),
    ];

    cm.reduce_context(&mut messages, ReduceContext::default())
        .await
        .unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].text(), "msg 2");
}

#[tokio::test]
async fn test_hooks_are_called() {
    use std::sync::Mutex;
    use strands_core::hooks::HookEvent;

    let events_log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let log_clone = events_log.clone();

    let mut agent = Agent::builder()
        .model(MockTextModel {
            response: "ok".to_string(),
        })
        .hook(move |event: &mut HookEvent| {
            let name = match event {
                HookEvent::BeforeInvocation(_) => "before_invocation",
                HookEvent::AfterInvocation(_) => "after_invocation",
                HookEvent::BeforeModelCall { .. } => "before_model_call",
                HookEvent::AfterModelCall(_) => "after_model_call",
                HookEvent::MessageAdded { .. } => "message_added",
                _ => "other",
            };
            log_clone.lock().unwrap().push(name.to_string());
        })
        .build()
        .unwrap();

    agent.prompt("test").await.unwrap();

    let log = events_log.lock().unwrap();
    assert!(log.contains(&"before_invocation".to_string()));
    assert!(log.contains(&"before_model_call".to_string()));
    assert!(log.contains(&"after_model_call".to_string()));
    assert!(log.contains(&"after_invocation".to_string()));
}

#[tokio::test]
async fn test_hook_cancel_tool() {
    use strands_core::hooks::HookEvent;
    use strands_core::hooks::events::BeforeToolCallEvent;

    // Model always requests the tool
    struct AlwaysToolModel;

    #[async_trait]
    impl Model for AlwaysToolModel {
        async fn stream(
            &self,
            messages: &[Message],
            _system_prompt: Option<&SystemPrompt>,
            _tool_specs: &[ToolSpec],
        ) -> Result<ModelStream> {
            // If we already have a tool result, just return text
            let has_tool_result = messages.iter().any(|m| {
                m.content.iter().any(|c| matches!(c, ContentBlock::ToolResult { .. }))
            });

            if has_tool_result {
                let events = vec![
                    Ok(StreamEvent::MessageStart { role: Role::Assistant }),
                    Ok(StreamEvent::ContentBlockStart {
                        index: 0,
                        content_type: ContentBlockType::Text,
                    }),
                    Ok(StreamEvent::ContentBlockDelta {
                        index: 0,
                        delta: DeltaContent::TextDelta("done".to_string()),
                    }),
                    Ok(StreamEvent::ContentBlockStop { index: 0 }),
                    Ok(StreamEvent::MessageStop { stop_reason: StopReason::EndTurn }),
                ];
                Ok(Box::pin(stream::iter(events)))
            } else {
                let events = vec![
                    Ok(StreamEvent::MessageStart { role: Role::Assistant }),
                    Ok(StreamEvent::ContentBlockStart {
                        index: 0,
                        content_type: ContentBlockType::ToolUse {
                            tool_use_id: "call_1".to_string(),
                            name: "greet".to_string(),
                        reasoning_signature: None,
                        },
                    }),
                    Ok(StreamEvent::ContentBlockDelta {
                        index: 0,
                        delta: DeltaContent::ToolInputDelta(r#"{"name":"test"}"#.to_string()),
                    }),
                    Ok(StreamEvent::ContentBlockStop { index: 0 }),
                    Ok(StreamEvent::MessageStop { stop_reason: StopReason::ToolUse }),
                ];
                Ok(Box::pin(stream::iter(events)))
            }
        }
    }

    let mut agent = Agent::builder()
        .model(AlwaysToolModel)
        .tool(GreetTool)
        .hook(|event: &mut HookEvent| {
            // Cancel all tool calls
            if let HookEvent::BeforeToolCall(BeforeToolCallEvent { cancel, .. }) = event {
                *cancel = true;
            }
        })
        .build()
        .unwrap();

    let result = agent.prompt("test").await.unwrap();
    assert_eq!(result.text(), "done");
}

#[tokio::test]
async fn test_max_cycles_limit() {
    struct InfiniteToolModel;

    #[async_trait]
    impl Model for InfiniteToolModel {
        async fn stream(
            &self,
            _messages: &[Message],
            _system_prompt: Option<&SystemPrompt>,
            _tool_specs: &[ToolSpec],
        ) -> Result<ModelStream> {
            let events = vec![
                Ok(StreamEvent::MessageStart {
                    role: Role::Assistant,
                }),
                Ok(StreamEvent::ContentBlockStart {
                    index: 0,
                    content_type: ContentBlockType::ToolUse {
                        tool_use_id: "call_loop".to_string(),
                        name: "greet".to_string(),
                        reasoning_signature: None,
                    },
                }),
                Ok(StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: DeltaContent::ToolInputDelta(r#"{"name":"loop"}"#.to_string()),
                }),
                Ok(StreamEvent::ContentBlockStop { index: 0 }),
                Ok(StreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                }),
            ];
            Ok(Box::pin(stream::iter(events)))
        }
    }

    let mut agent = Agent::builder()
        .model(InfiniteToolModel)
        .tool(GreetTool)
        .max_cycles(3)
        .build()
        .unwrap();

    let result = agent.prompt("loop forever").await;
    assert!(matches!(result, Err(StrandsError::MaxCycles(3))));
}

#[tokio::test]
async fn test_message_serialization() {
    let msg = Message::user("hello");
    let json = serde_json::to_string(&msg).unwrap();
    let deserialized: Message = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.text(), "hello");
    assert_eq!(deserialized.role, Role::User);
}

// ---------------------------------------------------------------------------
// Regression: concurrent tool retry must re-invoke with the ORIGINAL input.
//
// The concurrent execution path previously retried with `Value::Null`, so a
// hook-requested retry ran a different call than the one it asked to retry.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_concurrent_tool_retry_reuses_original_input() {
    use std::sync::Mutex;
    use strands_core::hooks::events::AfterToolCallEvent;
    use strands_core::hooks::HookEvent;

    /// Records every input it is invoked with.
    struct RecordingTool {
        seen: Arc<Mutex<Vec<serde_json::Value>>>,
    }

    #[async_trait]
    impl Tool for RecordingTool {
        fn name(&self) -> &str {
            "greet"
        }

        fn spec(&self) -> ToolSpec {
            ToolSpec::new("greet".to_string(), "Greet someone by name".to_string(), json!({
                    "type": "object",
                    "properties": { "name": { "type": "string" } },
                    "required": ["name"]
                }))
        }

        async fn invoke(
            &self,
            input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput> {
            self.seen.lock().unwrap().push(input.clone());
            Ok(ToolOutput::success(json!("ok")))
        }
    }

    let seen: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let retries = Arc::new(AtomicUsize::new(0));
    let retries_hook = retries.clone();

    let mut agent = Agent::builder()
        .model(MockToolModel {
            call_count: Arc::new(AtomicUsize::new(0)),
        })
        .tool(RecordingTool { seen: seen.clone() })
        .concurrent_tools(true)
        .hook(move |event: &mut HookEvent| {
            if let HookEvent::AfterToolCall(AfterToolCallEvent { retry, .. }) = event {
                // Ask for exactly one retry.
                if retries_hook.fetch_add(1, Ordering::SeqCst) == 0 {
                    *retry = true;
                }
            }
        })
        .build()
        .unwrap();

    agent.prompt("greet the world").await.unwrap();

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2, "expected an initial call plus one retry");
    assert_eq!(
        seen[0], seen[1],
        "the retry must re-invoke with the original input, not a substitute"
    );
    assert_eq!(seen[1]["name"], "World");
}

// ---------------------------------------------------------------------------
// Regression: a hook that always requests a model retry must terminate.
//
// Hook retries do not advance the cycle counter, so `max_cycles` never bounds
// them. Without a separate limit this loops forever.
// ---------------------------------------------------------------------------

// A multi-thread runtime is required: the retry loop only awaits
// already-ready futures, so on a current-thread runtime the timer below never
// gets polled and a regression hangs the suite instead of failing it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_hook_model_retry_is_bounded() {
    use strands_core::hooks::events::AfterModelCallEvent;
    use strands_core::hooks::HookEvent;

    let calls = Arc::new(AtomicUsize::new(0));
    let calls_model = calls.clone();

    struct CountingModel {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Model for CountingModel {
        async fn stream(
            &self,
            _messages: &[Message],
            _system_prompt: Option<&SystemPrompt>,
            _tool_specs: &[ToolSpec],
        ) -> Result<ModelStream> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let events = vec![
                Ok(StreamEvent::ContentBlockStart {
                    index: 0,
                    content_type: ContentBlockType::Text,
                }),
                Ok(StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: DeltaContent::TextDelta("hi".to_string()),
                }),
                Ok(StreamEvent::ContentBlockStop { index: 0 }),
                Ok(StreamEvent::MessageStop {
                    stop_reason: StopReason::EndTurn,
                }),
            ];
            Ok(Box::pin(stream::iter(events)))
        }
    }

    let mut agent = Agent::builder()
        .model(CountingModel {
            calls: calls_model,
        })
        .retry_config(RetryConfig {
            max_hook_retries: 2,
            ..Default::default()
        })
        .hook(|event: &mut HookEvent| {
            if let HookEvent::AfterModelCall(AfterModelCallEvent { retry, .. }) = event {
                // Always ask to retry — the loop must still terminate.
                *retry = true;
            }
        })
        .build()
        .unwrap();

    // Bounded by the runtime, not by the test: if the guard regresses this
    // never returns.
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        agent.prompt("hello"),
    )
    .await
    .expect("agent loop did not terminate: hook retries are unbounded");

    result.unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "expected the initial call plus max_hook_retries (2) retries"
    );
}

// ---------------------------------------------------------------------------
// Phase 0 — type foundations
// ---------------------------------------------------------------------------

/// Captures exactly what the agent hands the model, so we can assert on what
/// does and does not cross the boundary.
struct CapturingModel {
    seen_messages: Arc<std::sync::Mutex<Vec<Message>>>,
    seen_system: Arc<std::sync::Mutex<Option<SystemPrompt>>>,
}

#[async_trait]
impl Model for CapturingModel {
    async fn stream(
        &self,
        messages: &[Message],
        system_prompt: Option<&SystemPrompt>,
        _tool_specs: &[ToolSpec],
    ) -> Result<ModelStream> {
        *self.seen_messages.lock().unwrap() = messages.to_vec();
        *self.seen_system.lock().unwrap() = system_prompt.cloned();

        let events = vec![
            Ok(StreamEvent::ContentBlockStart {
                index: 0,
                content_type: ContentBlockType::Text,
            }),
            Ok(StreamEvent::ContentBlockDelta {
                index: 0,
                delta: DeltaContent::TextDelta("ok".to_string()),
            }),
            Ok(StreamEvent::ContentBlockStop { index: 0 }),
            Ok(StreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
            }),
        ];
        Ok(Box::pin(stream::iter(events)))
    }
}

#[tokio::test]
async fn test_tracking_ids_assigned_but_not_sent_to_the_model() {
    let seen_messages = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen_system = Arc::new(std::sync::Mutex::new(None));

    let mut agent = Agent::builder()
        .model(CapturingModel {
            seen_messages: seen_messages.clone(),
            seen_system: seen_system.clone(),
        })
        .build()
        .unwrap();

    agent.prompt("hello").await.unwrap();

    // The agent's own history carries durable ids...
    for msg in agent.messages() {
        assert!(
            msg.tracking_id.is_some(),
            "every stored message should be stamped: {msg:?}"
        );
    }
    let ids: Vec<_> = agent
        .messages()
        .iter()
        .map(|m| m.tracking_id.clone().unwrap())
        .collect();
    assert_eq!(
        ids.iter().collect::<std::collections::HashSet<_>>().len(),
        ids.len(),
        "ids must be unique within a conversation"
    );

    // ...but the model never sees them.
    for msg in seen_messages.lock().unwrap().iter() {
        assert!(
            msg.tracking_id.is_none() && msg.metadata.is_none(),
            "SDK bookkeeping must be stripped before the model call: {msg:?}"
        );
    }
}

#[tokio::test]
async fn test_assistant_message_records_its_own_usage() {
    let mut agent = Agent::builder()
        .model(MockTextModel {
            response: "hi".to_string(),
        })
        .build()
        .unwrap();

    agent.prompt("hello").await.unwrap();

    let assistant = agent
        .messages()
        .iter()
        .find(|m| m.role == Role::Assistant)
        .expect("assistant message present");
    assert!(
        assistant.metadata.as_ref().is_some_and(|m| m.usage.is_some()),
        "the assistant message should carry the usage of the call that produced it"
    );
}

#[tokio::test]
async fn test_structured_system_prompt_reaches_the_model_with_cache_points() {
    use strands_core::types::content::{CachePoint, SystemContentBlock};

    let seen_messages = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen_system = Arc::new(std::sync::Mutex::new(None));

    let mut agent = Agent::builder()
        .model(CapturingModel {
            seen_messages: seen_messages.clone(),
            seen_system: seen_system.clone(),
        })
        .system_prompt(vec![
            SystemContentBlock::Text {
                text: "long standing instructions".into(),
            },
            SystemContentBlock::CachePoint(CachePoint::new().with_ttl("1h")),
        ])
        .build()
        .unwrap();

    agent.prompt("hello").await.unwrap();

    let system = seen_system.lock().unwrap().clone().expect("system prompt");
    let (text, blocks) = system.split();

    assert_eq!(text.as_deref(), Some("long standing instructions"));
    assert_eq!(blocks.len(), 2, "the cache point must survive to the adapter");
    assert!(matches!(blocks[1], SystemContentBlock::CachePoint(_)));
}

#[tokio::test]
async fn test_plain_string_system_prompt_still_works() {
    let seen_system = Arc::new(std::sync::Mutex::new(None));

    let mut agent = Agent::builder()
        .model(CapturingModel {
            seen_messages: Arc::new(std::sync::Mutex::new(Vec::new())),
            seen_system: seen_system.clone(),
        })
        .system_prompt("be helpful")
        .build()
        .unwrap();

    agent.prompt("hello").await.unwrap();

    let system = seen_system.lock().unwrap().clone().expect("system prompt");
    assert_eq!(system.as_text().as_deref(), Some("be helpful"));
}

#[tokio::test]
async fn test_reasoning_signature_survives_as_its_own_block() {
    // A provider that ties reasoning to a tool call via a signature, with no
    // reasoning text. Dropping the signature invalidates the tool call.
    struct SignedToolModel;

    #[async_trait]
    impl Model for SignedToolModel {
        async fn stream(
            &self,
            _messages: &[Message],
            _system_prompt: Option<&SystemPrompt>,
            _tool_specs: &[ToolSpec],
        ) -> Result<ModelStream> {
            let events = vec![
                Ok(StreamEvent::ContentBlockStart {
                    index: 0,
                    content_type: ContentBlockType::ToolUse {
                        tool_use_id: "call_1".to_string(),
                        name: "greet".to_string(),
                        reasoning_signature: Some("sig-abc".to_string()),
                    },
                }),
                Ok(StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: DeltaContent::ToolInputDelta(r#"{"name":"World"}"#.to_string()),
                }),
                Ok(StreamEvent::ContentBlockStop { index: 0 }),
                Ok(StreamEvent::MessageStop {
                    stop_reason: StopReason::EndTurn,
                }),
            ];
            Ok(Box::pin(stream::iter(events)))
        }
    }

    let mut agent = Agent::builder()
        .model(SignedToolModel)
        .tool(GreetTool)
        .build()
        .unwrap();

    let result = agent.prompt("greet").await.unwrap();

    let has_signature = result.message.content.iter().any(|b| {
        matches!(
            b,
            ContentBlock::Reasoning(r) if r.signature.as_deref() == Some("sig-abc")
        )
    });
    assert!(
        has_signature,
        "a reasoning block with empty text but a signature must be preserved: {:?}",
        result.message.content
    );
}

// ---------------------------------------------------------------------------
// Phase 1 — batch tool hooks, ordering, duration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_before_tools_can_cancel_the_whole_batch() {
    use std::sync::Mutex;
    use strands_core::hooks::events::BeforeToolsEvent;
    use strands_core::hooks::HookEvent;

    let invoked = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));

    struct RecordingTool {
        seen: Arc<Mutex<Vec<serde_json::Value>>>,
    }

    #[async_trait]
    impl Tool for RecordingTool {
        fn name(&self) -> &str {
            "greet"
        }
        fn spec(&self) -> ToolSpec {
            ToolSpec::new("greet", "greet", json!({"type": "object"}))
        }
        async fn invoke(
            &self,
            input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput> {
            self.seen.lock().unwrap().push(input);
            Ok(ToolOutput::success(json!("ran")))
        }
    }

    let mut agent = Agent::builder()
        .model(MockToolModel {
            call_count: Arc::new(AtomicUsize::new(0)),
        })
        .tool(RecordingTool {
            seen: invoked.clone(),
        })
        .hook(|event: &mut HookEvent| {
            if let HookEvent::BeforeTools(BeforeToolsEvent { cancel, .. }) = event {
                *cancel = true;
            }
        })
        .build()
        .unwrap();

    agent.prompt("greet the world").await.unwrap();

    assert!(
        invoked.lock().unwrap().is_empty(),
        "cancelling the batch must stop every tool, not just the first"
    );
}

#[tokio::test]
async fn test_before_tools_sees_the_requested_calls() {
    use std::sync::Mutex;
    use strands_core::hooks::events::BeforeToolsEvent;
    use strands_core::hooks::HookEvent;

    let seen = Arc::new(Mutex::new(Vec::<(String, String)>::new()));
    let seen_hook = seen.clone();

    let mut agent = Agent::builder()
        .model(MockToolModel {
            call_count: Arc::new(AtomicUsize::new(0)),
        })
        .tool(GreetTool)
        .hook(move |event: &mut HookEvent| {
            if let HookEvent::BeforeTools(BeforeToolsEvent { tool_calls, .. }) = event {
                *seen_hook.lock().unwrap() = tool_calls.clone();
            }
        })
        .build()
        .unwrap();

    agent.prompt("greet").await.unwrap();

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0], ("call_1".to_string(), "greet".to_string()));
}

#[tokio::test]
async fn test_after_tools_can_end_the_turn() {
    use strands_core::hooks::events::AfterToolsEvent;
    use strands_core::hooks::HookEvent;

    let call_count = Arc::new(AtomicUsize::new(0));

    let mut agent = Agent::builder()
        .model(MockToolModel {
            call_count: call_count.clone(),
        })
        .tool(GreetTool)
        .hook(|event: &mut HookEvent| {
            if let HookEvent::AfterTools(AfterToolsEvent { end_turn, .. }) = event {
                *end_turn = true;
            }
        })
        .build()
        .unwrap();

    let result = agent.prompt("greet").await.unwrap();

    assert_eq!(result.stop_reason, StopReason::EndTurn);
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "ending the turn must skip the follow-up model call"
    );
}

#[tokio::test]
async fn test_after_tool_call_reports_a_duration() {
    use std::sync::Mutex;
    use strands_core::hooks::events::AfterToolCallEvent;
    use strands_core::hooks::HookEvent;

    let duration = Arc::new(Mutex::new(None::<std::time::Duration>));
    let duration_hook = duration.clone();

    struct SlowTool;

    #[async_trait]
    impl Tool for SlowTool {
        fn name(&self) -> &str {
            "greet"
        }
        fn spec(&self) -> ToolSpec {
            ToolSpec::new("greet", "greet", json!({"type": "object"}))
        }
        async fn invoke(
            &self,
            _input: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput> {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            Ok(ToolOutput::success(json!("ok")))
        }
    }

    let mut agent = Agent::builder()
        .model(MockToolModel {
            call_count: Arc::new(AtomicUsize::new(0)),
        })
        .tool(SlowTool)
        .hook(move |event: &mut HookEvent| {
            if let HookEvent::AfterToolCall(AfterToolCallEvent { duration, .. }) = event {
                *duration_hook.lock().unwrap() = Some(*duration);
            }
        })
        .build()
        .unwrap();

    agent.prompt("greet").await.unwrap();

    let observed = duration.lock().unwrap().expect("duration reported");
    assert!(
        observed >= std::time::Duration::from_millis(15),
        "expected the measured duration to reflect the tool's work, got {observed:?}"
    );
}

#[tokio::test]
async fn test_hook_order_controls_dispatch_sequence() {
    use std::sync::Mutex;
    use strands_core::hooks::registry::order;
    use strands_core::hooks::HookEvent;

    let log: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let (a, b) = (log.clone(), log.clone());

    let mut agent = Agent::builder()
        .model(MockTextModel {
            response: "hi".to_string(),
        })
        .hook(move |event: &mut HookEvent| {
            if matches!(event, HookEvent::BeforeModelCall { .. }) {
                a.lock().unwrap().push("default");
            }
        })
        .hook_with_order(
            move |event: &mut HookEvent| {
                if matches!(event, HookEvent::BeforeModelCall { .. }) {
                    b.lock().unwrap().push("first");
                }
            },
            order::SDK_FIRST,
        )
        .build()
        .unwrap();

    agent.prompt("hello").await.unwrap();

    assert_eq!(
        *log.lock().unwrap(),
        vec!["first", "default"],
        "a lower order must run first regardless of registration order"
    );
}

// ---------------------------------------------------------------------------
// Phase 1 — invocation limits
// ---------------------------------------------------------------------------

/// Always requests a tool, so the loop keeps turning until something stops it.
struct AlwaysToolModel {
    calls: Arc<AtomicUsize>,
    output_tokens: u64,
}

#[async_trait]
impl Model for AlwaysToolModel {
    async fn stream(
        &self,
        _messages: &[Message],
        _system_prompt: Option<&SystemPrompt>,
        _tool_specs: &[ToolSpec],
    ) -> Result<ModelStream> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let events = vec![
            Ok(StreamEvent::ContentBlockStart {
                index: 0,
                content_type: ContentBlockType::ToolUse {
                    tool_use_id: "call_1".to_string(),
                    name: "greet".to_string(),
                    reasoning_signature: None,
                },
            }),
            Ok(StreamEvent::ContentBlockDelta {
                index: 0,
                delta: DeltaContent::ToolInputDelta(r#"{"name":"World"}"#.to_string()),
            }),
            Ok(StreamEvent::ContentBlockStop { index: 0 }),
            Ok(StreamEvent::Metadata {
                usage: Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(self.output_tokens),
                    ..Default::default()
                },
                metrics: Metrics::default(),
            }),
            Ok(StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
            }),
        ];
        Ok(Box::pin(stream::iter(events)))
    }
}

#[tokio::test]
async fn test_turn_limit_stops_cleanly_rather_than_erroring() {
    use strands_core::agent::Limits;

    let calls = Arc::new(AtomicUsize::new(0));
    let mut agent = Agent::builder()
        .model(AlwaysToolModel {
            calls: calls.clone(),
            output_tokens: 1,
        })
        .tool(GreetTool)
        .limits(Limits::turns(3))
        .max_cycles(100)
        .build()
        .unwrap();

    let result = agent.prompt("go").await.unwrap();

    assert_eq!(result.stop_reason, StopReason::LimitTurns);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn test_limit_leaves_history_reinvokable() {
    use strands_core::agent::Limits;

    let mut agent = Agent::builder()
        .model(AlwaysToolModel {
            calls: Arc::new(AtomicUsize::new(0)),
            output_tokens: 1,
        })
        .tool(GreetTool)
        .limits(Limits::turns(2))
        .max_cycles(100)
        .build()
        .unwrap();

    agent.prompt("go").await.unwrap();

    // Every ToolUse must have been answered — stopping mid-pair would leave a
    // history the provider rejects on the next call.
    let last = agent.messages().last().expect("history non-empty");
    assert!(
        !last.has_tool_use(),
        "history must not end on an unanswered tool call: {last:?}"
    );
}

#[tokio::test]
async fn test_total_token_limit_stops_the_loop() {
    use strands_core::agent::Limits;

    let mut agent = Agent::builder()
        .model(AlwaysToolModel {
            calls: Arc::new(AtomicUsize::new(0)),
            output_tokens: 40,
        })
        .tool(GreetTool)
        .limits(Limits::default().with_total_tokens(100))
        .max_cycles(100)
        .build()
        .unwrap();

    let result = agent.prompt("go").await.unwrap();
    assert_eq!(result.stop_reason, StopReason::LimitTotalTokens);
}

#[tokio::test]
async fn test_output_token_limit_stops_the_loop() {
    use strands_core::agent::Limits;

    let mut agent = Agent::builder()
        .model(AlwaysToolModel {
            calls: Arc::new(AtomicUsize::new(0)),
            output_tokens: 60,
        })
        .tool(GreetTool)
        .limits(Limits::default().with_output_tokens(100))
        .max_cycles(100)
        .build()
        .unwrap();

    let result = agent.prompt("go").await.unwrap();
    assert_eq!(result.stop_reason, StopReason::LimitOutputTokens);
}

#[tokio::test]
async fn test_custom_tool_executor_is_used() {
    use std::sync::Mutex;
    use strands_core::tool::executor::to_result_block;
    use strands_core::tool::{ToolCall, ToolExecutor};

    /// Bypasses the tools entirely and answers every call itself.
    struct StubExecutor {
        used: Arc<Mutex<bool>>,
    }

    #[async_trait]
    impl ToolExecutor for StubExecutor {
        fn name(&self) -> &'static str {
            "stub"
        }
        async fn execute(
            &self,
            _tools: &std::collections::HashMap<String, Box<dyn Tool>>,
            calls: &[ToolCall<'_>],
            _ctx: &ToolContext,
            _hooks: &strands_core::hooks::HookRegistry,
            _interrupts: &mut strands_core::InterruptState,
        ) -> Vec<ContentBlock> {
            *self.used.lock().unwrap() = true;
            calls
                .iter()
                .map(|c| to_result_block(c.tool_use_id, &ToolOutput::success(json!("stubbed"))))
                .collect()
        }
    }

    let used = Arc::new(Mutex::new(false));
    let mut agent = Agent::builder()
        .model(MockToolModel {
            call_count: Arc::new(AtomicUsize::new(0)),
        })
        .tool(GreetTool)
        .tool_executor(StubExecutor { used: used.clone() })
        .build()
        .unwrap();

    agent.prompt("greet").await.unwrap();
    assert!(*used.lock().unwrap(), "the custom executor should have run");
}

// ---------------------------------------------------------------------------
// Phase 1 — structured output
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Deserialize, PartialEq)]
struct Person {
    name: String,
    age: u32,
}

fn person_spec() -> strands_core::tool::StructuredOutputSpec {
    strands_core::tool::StructuredOutputSpec::new(
        "Person",
        "A person record",
        json!({
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"]
        }),
    )
}

/// Emits a single tool call with the supplied JSON as its arguments.
struct StructuredModel {
    payloads: Arc<std::sync::Mutex<Vec<String>>>,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Model for StructuredModel {
    async fn stream(
        &self,
        _messages: &[Message],
        _system_prompt: Option<&SystemPrompt>,
        _tool_specs: &[ToolSpec],
    ) -> Result<ModelStream> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        let payload = {
            let p = self.payloads.lock().unwrap();
            p.get(n).cloned()
        };

        let Some(payload) = payload else {
            // Nothing left to say — finish the turn without a tool call.
            return Ok(Box::pin(stream::iter(vec![Ok(StreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
            })])));
        };

        let events = vec![
            Ok(StreamEvent::ContentBlockStart {
                index: 0,
                content_type: ContentBlockType::ToolUse {
                    tool_use_id: format!("call_{n}"),
                    name: "Person".to_string(),
                    reasoning_signature: None,
                },
            }),
            Ok(StreamEvent::ContentBlockDelta {
                index: 0,
                delta: DeltaContent::ToolInputDelta(payload),
            }),
            Ok(StreamEvent::ContentBlockStop { index: 0 }),
            Ok(StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
            }),
        ];
        Ok(Box::pin(stream::iter(events)))
    }
}

fn structured_agent(payloads: Vec<&str>) -> Agent {
    Agent::builder()
        .model(StructuredModel {
            payloads: Arc::new(std::sync::Mutex::new(
                payloads.into_iter().map(String::from).collect(),
            )),
            calls: Arc::new(AtomicUsize::new(0)),
        })
        .max_cycles(10)
        .build()
        .unwrap()
}

#[tokio::test]
async fn test_structured_output_returns_a_typed_value() {
    let mut agent = structured_agent(vec![r#"{"name":"Ada","age":36}"#]);

    let person: Person = agent
        .prompt_structured("describe Ada", person_spec())
        .await
        .unwrap();

    assert_eq!(
        person,
        Person {
            name: "Ada".into(),
            age: 36
        }
    );
}

#[tokio::test]
async fn test_structured_output_lets_the_model_correct_itself() {
    // First attempt omits a required field; the validation error goes back to
    // the model, which fixes it on the next turn.
    let mut agent = structured_agent(vec![
        r#"{"name":"Ada"}"#,
        r#"{"name":"Ada","age":36}"#,
    ]);

    let person: Person = agent
        .prompt_structured("describe Ada", person_spec())
        .await
        .unwrap();

    assert_eq!(person.age, 36);
}

#[tokio::test]
async fn test_structured_output_errors_when_never_produced() {
    let mut agent = structured_agent(vec![]);

    let result: Result<Person> = agent.prompt_structured("describe Ada", person_spec()).await;

    assert!(
        result.is_err(),
        "a run that never produced the structure must fail, not return a default"
    );
}

#[tokio::test]
async fn test_structured_output_tool_is_not_advertised_afterwards() {
    let mut agent = structured_agent(vec![r#"{"name":"Ada","age":36}"#]);

    let _: Person = agent
        .prompt_structured("describe Ada", person_spec())
        .await
        .unwrap();

    // Leaving the synthetic tool registered would let later, unrelated turns
    // call it.
    assert!(
        !agent.tool_names().any(|n| n == "Person"),
        "the synthetic tool must be removed once the call completes"
    );
}

#[tokio::test]
async fn test_structured_output_tool_is_removed_after_a_failure_too() {
    let mut agent = structured_agent(vec![]);
    let _: Result<Person> = agent.prompt_structured("describe Ada", person_spec()).await;

    assert!(!agent.tool_names().any(|n| n == "Person"));
}

// ---------------------------------------------------------------------------
// Phase 2 — human-in-the-loop interrupts
// ---------------------------------------------------------------------------

/// A destructive tool that records whether it was ever allowed to run.
struct DeleteTool {
    ran: Arc<std::sync::Mutex<bool>>,
}

#[async_trait]
impl Tool for DeleteTool {
    fn name(&self) -> &str {
        "greet"
    }
    fn spec(&self) -> ToolSpec {
        ToolSpec::new("greet", "deletes something", json!({"type": "object"}))
    }
    async fn invoke(&self, _input: serde_json::Value, _ctx: &ToolContext) -> Result<ToolOutput> {
        *self.ran.lock().unwrap() = true;
        Ok(ToolOutput::success(json!("deleted")))
    }
}

/// Approval hook: pauses the first time, then acts on the human's answer.
fn approval_hook() -> impl strands_core::hooks::Hook {
    use strands_core::hooks::HookEvent;

    |event: &mut HookEvent| {
        if let HookEvent::BeforeToolCall(e) = event {
            match e.interrupt("approve_delete", Some(json!("Delete this?"))) {
                // No answer yet — refuse to act. This is the safety property.
                None => e.cancel = true,
                Some(answer) => {
                    if answer.as_str() != Some("yes") {
                        e.cancel = true;
                    }
                }
            }
        }
    }
}

/// Requests the tool until the conversation contains a tool result, then stops.
///
/// Decides from the history rather than a call counter, so it behaves the same
/// on a resumed invocation as on the first one.
struct ToolUntilAnsweredModel;

#[async_trait]
impl Model for ToolUntilAnsweredModel {
    async fn stream(
        &self,
        messages: &[Message],
        _system_prompt: Option<&SystemPrompt>,
        _tool_specs: &[ToolSpec],
    ) -> Result<ModelStream> {
        let already_ran = messages.iter().any(|m| {
            m.content.iter().any(|b| {
                matches!(
                    b,
                    ContentBlock::ToolResult { status, .. }
                        if *status == strands_core::types::content::ToolResultStatus::Success
                )
            })
        });

        if already_ran {
            return Ok(Box::pin(stream::iter(vec![Ok(StreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
            })])));
        }

        let events = vec![
            Ok(StreamEvent::ContentBlockStart {
                index: 0,
                content_type: ContentBlockType::ToolUse {
                    tool_use_id: "call_1".to_string(),
                    name: "greet".to_string(),
                    reasoning_signature: None,
                },
            }),
            Ok(StreamEvent::ContentBlockDelta {
                index: 0,
                delta: DeltaContent::ToolInputDelta(r#"{"key":"X"}"#.to_string()),
            }),
            Ok(StreamEvent::ContentBlockStop { index: 0 }),
            Ok(StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
            }),
        ];
        Ok(Box::pin(stream::iter(events)))
    }
}

fn interrupt_agent(ran: Arc<std::sync::Mutex<bool>>) -> Agent {
    Agent::builder()
        .model(ToolUntilAnsweredModel)
        .tool(DeleteTool { ran })
        .hook(approval_hook())
        .max_cycles(10)
        .build()
        .unwrap()
}

#[tokio::test]
async fn test_interrupt_pauses_the_run_without_acting() {
    let ran = Arc::new(std::sync::Mutex::new(false));
    let mut agent = interrupt_agent(ran.clone());

    let result = agent.prompt("delete key X").await.unwrap();

    assert_eq!(result.stop_reason, StopReason::Interrupt);
    assert!(result.is_interrupted());
    assert_eq!(result.interrupts.len(), 1);
    assert_eq!(result.interrupts[0].name, "approve_delete");
    assert_eq!(result.interrupts[0].reason, Some(json!("Delete this?")));
    assert!(
        !*ran.lock().unwrap(),
        "the tool must not run while approval is outstanding"
    );
}

#[tokio::test]
async fn test_interrupt_resumes_and_acts_on_approval() {
    use strands_core::InterruptResponse;

    let ran = Arc::new(std::sync::Mutex::new(false));
    let mut agent = interrupt_agent(ran.clone());

    let paused = agent.prompt("delete key X").await.unwrap();
    let id = paused.interrupts[0].id.clone();

    assert_eq!(agent.respond(&[InterruptResponse::new(id, "yes")]), 1);

    let resumed = agent.prompt("continue").await.unwrap();

    assert_ne!(resumed.stop_reason, StopReason::Interrupt);
    assert!(
        *ran.lock().unwrap(),
        "the tool should run once approval is granted"
    );
}

#[tokio::test]
async fn test_interrupt_denial_keeps_the_tool_cancelled() {
    use strands_core::InterruptResponse;

    let ran = Arc::new(std::sync::Mutex::new(false));
    let mut agent = interrupt_agent(ran.clone());

    let paused = agent.prompt("delete key X").await.unwrap();
    let id = paused.interrupts[0].id.clone();

    agent.respond(&[InterruptResponse::new(id, "no")]);

    // A denied tool keeps this model retrying, so the run ends on the cycle
    // cap rather than cleanly. What matters is that the denial holds for every
    // one of those attempts.
    let outcome = agent.prompt("continue").await;
    assert!(
        matches!(outcome, Err(StrandsError::MaxCycles(_))) || outcome.is_ok(),
        "unexpected outcome: {outcome:?}"
    );
    assert!(
        !*ran.lock().unwrap(),
        "a denied approval must block the tool on every retry"
    );
}

#[tokio::test]
async fn test_interrupt_leaves_history_valid_for_resume() {
    let ran = Arc::new(std::sync::Mutex::new(false));
    let mut agent = interrupt_agent(ran);

    agent.prompt("delete key X").await.unwrap();

    // Pausing mid-batch would leave a ToolUse with no ToolResult, which the
    // provider rejects on the resuming call.
    let last = agent.messages().last().expect("history non-empty");
    assert!(
        !last.has_tool_use(),
        "history must not end on an unanswered tool call: {last:?}"
    );
}

#[tokio::test]
async fn test_completed_run_does_not_leave_a_stale_interrupt() {
    let mut agent = Agent::builder()
        .model(MockTextModel {
            response: "hi".to_string(),
        })
        .build()
        .unwrap();

    agent.prompt("hello").await.unwrap();
    assert!(agent.pending_interrupts().is_empty());
}

// ---------------------------------------------------------------------------
// Phase 2 — model middleware
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_model_middleware_can_short_circuit_with_a_cached_response() {
    use futures::future::BoxFuture;
    use strands_core::middleware::stages::InvokeModelResult;
    use strands_core::middleware::{InvokeModelContext, Middleware, ModelCallOutcome, Next};

    /// Answers without ever reaching the model.
    struct CacheHit {
        hits: Arc<AtomicUsize>,
    }

    impl Middleware<InvokeModelContext, InvokeModelResult> for CacheHit {
        fn handle<'a>(
            &'a self,
            _ctx: InvokeModelContext,
            _next: Next<'a, InvokeModelContext, InvokeModelResult>,
        ) -> BoxFuture<'a, InvokeModelResult> {
            self.hits.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                Ok(ModelCallOutcome {
                    content: vec![ContentBlock::Text {
                        text: "from cache".to_string(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                    metrics: Metrics::default(),
                })
            })
        }
    }

    let model_calls = Arc::new(AtomicUsize::new(0));
    let hits = Arc::new(AtomicUsize::new(0));

    let mut agent = Agent::builder()
        .model(AlwaysToolModel {
            calls: model_calls.clone(),
            output_tokens: 1,
        })
        .tool(GreetTool)
        .model_middleware(CacheHit { hits: hits.clone() })
        .build()
        .unwrap();

    let result = agent.prompt("hello").await.unwrap();

    assert_eq!(result.text(), "from cache");
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    assert_eq!(
        model_calls.load(Ordering::SeqCst),
        0,
        "short-circuiting must skip the model entirely"
    );
}

#[tokio::test]
async fn test_model_middleware_can_rewrite_the_request_and_the_result() {
    use futures::future::BoxFuture;
    use std::sync::Mutex;
    use strands_core::middleware::stages::InvokeModelResult;
    use strands_core::middleware::{InvokeModelContext, Middleware, Next};

    /// Injects a system prompt on the way in, uppercases text on the way out.
    struct Rewriter {
        seen_prompt: Arc<Mutex<Option<String>>>,
    }

    impl Middleware<InvokeModelContext, InvokeModelResult> for Rewriter {
        fn handle<'a>(
            &'a self,
            mut ctx: InvokeModelContext,
            next: Next<'a, InvokeModelContext, InvokeModelResult>,
        ) -> BoxFuture<'a, InvokeModelResult> {
            *self.seen_prompt.lock().unwrap() =
                ctx.system_prompt.as_ref().and_then(|p| p.as_text());
            ctx.system_prompt = Some(SystemPrompt::from("injected by middleware"));

            Box::pin(async move {
                let mut outcome = next.run(ctx).await?;
                for block in &mut outcome.content {
                    if let ContentBlock::Text { text } = block {
                        *text = text.to_uppercase();
                    }
                }
                Ok(outcome)
            })
        }
    }

    /// Reports back whatever system prompt it received.
    struct EchoPromptModel;

    #[async_trait]
    impl Model for EchoPromptModel {
        async fn stream(
            &self,
            _messages: &[Message],
            system_prompt: Option<&SystemPrompt>,
            _tool_specs: &[ToolSpec],
        ) -> Result<ModelStream> {
            let text = system_prompt
                .and_then(|p| p.as_text())
                .unwrap_or_else(|| "none".to_string());
            let events = vec![
                Ok(StreamEvent::ContentBlockStart {
                    index: 0,
                    content_type: ContentBlockType::Text,
                }),
                Ok(StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: DeltaContent::TextDelta(text),
                }),
                Ok(StreamEvent::ContentBlockStop { index: 0 }),
                Ok(StreamEvent::MessageStop {
                    stop_reason: StopReason::EndTurn,
                }),
            ];
            Ok(Box::pin(stream::iter(events)))
        }
    }

    let seen_prompt = Arc::new(Mutex::new(None));
    let mut agent = Agent::builder()
        .model(EchoPromptModel)
        .system_prompt("original")
        .model_middleware(Rewriter {
            seen_prompt: seen_prompt.clone(),
        })
        .build()
        .unwrap();

    let result = agent.prompt("hi").await.unwrap();

    assert_eq!(
        seen_prompt.lock().unwrap().as_deref(),
        Some("original"),
        "middleware should observe the original request"
    );
    assert_eq!(
        result.text(),
        "INJECTED BY MIDDLEWARE",
        "the rewritten prompt should reach the model and the result be transformed"
    );
}

#[tokio::test]
async fn test_model_middleware_can_swap_the_model_for_one_call() {
    use futures::future::BoxFuture;
    use strands_core::middleware::stages::InvokeModelResult;
    use strands_core::middleware::{InvokeModelContext, Middleware, Next};

    /// Routes the call to a different model. This is the mechanism model
    /// routing and fallback are built on.
    struct Router {
        replacement: Arc<dyn Model>,
    }

    impl Middleware<InvokeModelContext, InvokeModelResult> for Router {
        fn handle<'a>(
            &'a self,
            mut ctx: InvokeModelContext,
            next: Next<'a, InvokeModelContext, InvokeModelResult>,
        ) -> BoxFuture<'a, InvokeModelResult> {
            ctx.model = self.replacement.clone();
            Box::pin(async move { next.run(ctx).await })
        }
    }

    let primary_calls = Arc::new(AtomicUsize::new(0));
    let mut agent = Agent::builder()
        .model(AlwaysToolModel {
            calls: primary_calls.clone(),
            output_tokens: 1,
        })
        .tool(GreetTool)
        .model_middleware(Router {
            replacement: Arc::new(MockTextModel {
                response: "from the replacement".to_string(),
            }),
        })
        .build()
        .unwrap();

    let result = agent.prompt("hi").await.unwrap();

    assert_eq!(result.text(), "from the replacement");
    assert_eq!(
        primary_calls.load(Ordering::SeqCst),
        0,
        "the original model must not be called"
    );
}
