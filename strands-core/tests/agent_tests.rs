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
    use strands_core::conversation::SlidingWindowConversationManager;

    let cm = SlidingWindowConversationManager::new(3);
    let mut messages = vec![
        Message::user("msg 1"),
        Message::assistant(vec![]),
        Message::user("msg 2"),
        Message::assistant(vec![]),
        Message::user("msg 3"),
    ];

    cm.reduce_context(&mut messages, None).await.unwrap();
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
