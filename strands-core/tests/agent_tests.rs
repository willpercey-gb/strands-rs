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
        _system_prompt: Option<&str>,
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
        _system_prompt: Option<&str>,
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
        ToolSpec {
            name: "greet".to_string(),
            description: "Greet someone by name".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" }
                },
                "required": ["name"]
            }),
        }
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
            _system_prompt: Option<&str>,
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
            _system_prompt: Option<&str>,
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
