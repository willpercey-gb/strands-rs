use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use reqwest::Client;
use strands_core::model::{Model, ModelStream};
use strands_core::types::message::{Message, Role};
use strands_core::types::streaming::StreamEvent;
use strands_core::types::tools::ToolSpec;
use strands_core::{ContentBlock, StrandsError};

use crate::stream::OpenAiStreamState;
use crate::types::*;

const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Adapter for OpenRouter's OpenAI-compatible `/chat/completions` endpoint.
pub struct OpenRouterModel {
    base_url: String,
    api_key: String,
    model: String,
    referrer: Option<String>,
    app_title: Option<String>,
    client: Client,
}

impl OpenRouterModel {
    /// Create a new OpenRouter adapter for `model` using `api_key`.
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            model: model.into(),
            referrer: None,
            app_title: None,
            client: Client::new(),
        }
    }

    /// Override the base URL (defaults to `https://openrouter.ai/api/v1`).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the `HTTP-Referer` header. OpenRouter uses it for analytics
    /// and ranking on the openrouter.ai/rankings page.
    pub fn with_referrer(mut self, r: impl Into<String>) -> Self {
        self.referrer = Some(r.into());
        self
    }

    /// Set the `X-Title` header (the app name shown on rankings).
    pub fn with_app_title(mut self, t: impl Into<String>) -> Self {
        self.app_title = Some(t.into());
        self
    }
}

#[async_trait]
impl Model for OpenRouterModel {
    async fn stream(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
        tool_specs: &[ToolSpec],
    ) -> Result<ModelStream, StrandsError> {
        let body = build_request(self, messages, system_prompt, tool_specs);
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let mut req = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body);
        if let Some(r) = &self.referrer {
            req = req.header("HTTP-Referer", r);
        }
        if let Some(t) = &self.app_title {
            req = req.header("X-Title", t);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| StrandsError::Other(format!("openrouter request: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(StrandsError::Other(format!(
                "openrouter {status}: {body}"
            )));
        }

        let byte_stream = resp.bytes_stream();
        // Parse SSE line-by-line, then run each `data:` JSON through
        // OpenAiStreamState to produce the strands StreamEvent flow.
        let parsed = sse_to_events(byte_stream);
        Ok(Box::pin(parsed))
    }
}

fn build_request<'a>(
    m: &'a OpenRouterModel,
    messages: &[Message],
    system_prompt: Option<&str>,
    tool_specs: &[ToolSpec],
) -> ChatCompletionRequest<'a> {
    let mut converted = Vec::new();
    if let Some(sys) = system_prompt {
        if !sys.is_empty() {
            converted.push(OaiMessage {
                role: "system".into(),
                content: OaiContent::Text(sys.into()),
                tool_call_id: None,
                tool_calls: Vec::new(),
            });
        }
    }
    for msg in messages {
        match msg.role {
            Role::System => {
                let text = msg.text();
                if !text.is_empty() {
                    converted.push(OaiMessage {
                        role: "system".into(),
                        content: OaiContent::Text(text),
                        tool_call_id: None,
                        tool_calls: Vec::new(),
                    });
                }
            }
            Role::User => {
                // Tool results are folded into a synthetic "tool" role
                // message per OpenAI conventions; plain text/images go
                // under the user role.
                let mut user_parts = Vec::new();
                let mut tool_results = Vec::new();
                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text } => {
                            user_parts.push(OaiContentPart::Text { text: text.clone() });
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            tool_results.push(OaiMessage {
                                role: "tool".into(),
                                content: OaiContent::Text(
                                    serde_json::to_string(content).unwrap_or_default(),
                                ),
                                tool_call_id: Some(tool_use_id.clone()),
                                tool_calls: Vec::new(),
                            });
                        }
                        _ => {}
                    }
                }
                if !user_parts.is_empty() {
                    converted.push(OaiMessage {
                        role: "user".into(),
                        content: if user_parts.len() == 1
                            && matches!(user_parts[0], OaiContentPart::Text { .. })
                        {
                            let OaiContentPart::Text { text } = user_parts.remove(0) else {
                                unreachable!()
                            };
                            OaiContent::Text(text)
                        } else {
                            OaiContent::Parts(user_parts)
                        },
                        tool_call_id: None,
                        tool_calls: Vec::new(),
                    });
                }
                converted.extend(tool_results);
            }
            Role::Assistant => {
                let mut text_parts = String::new();
                let mut echoed_tool_calls = Vec::new();
                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text } => text_parts.push_str(text),
                        ContentBlock::ToolUse {
                            tool_use_id,
                            name,
                            input,
                        } => {
                            echoed_tool_calls.push(OaiToolCallEcho {
                                id: tool_use_id.clone(),
                                kind: "function",
                                function: OaiToolFunctionEcho {
                                    name: name.clone(),
                                    arguments: input.to_string(),
                                },
                            });
                        }
                        _ => {}
                    }
                }
                converted.push(OaiMessage {
                    role: "assistant".into(),
                    content: OaiContent::Text(text_parts),
                    tool_call_id: None,
                    tool_calls: echoed_tool_calls,
                });
            }
        }
    }

    let tools = tool_specs
        .iter()
        .map(|t| OaiTool {
            kind: "function",
            function: OaiToolFunction {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            },
        })
        .collect();

    ChatCompletionRequest {
        model: &m.model,
        messages: converted,
        stream: true,
        tools,
    }
}

/// Convert a byte stream of SSE-formatted bytes into a stream of strands
/// `StreamEvent`s. Emits one event per `data:` JSON chunk.
fn sse_to_events(
    byte_stream: impl futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>>
        + Send
        + 'static,
) -> impl futures::Stream<Item = Result<StreamEvent, StrandsError>> + Send + 'static {
    let mut buf = String::new();
    let mut state = OpenAiStreamState::new();

    byte_stream
        .map(move |item| {
            let bytes = item.map_err(|e| StrandsError::Other(format!("sse read: {e}")))?;
            let chunk = std::str::from_utf8(&bytes)
                .map_err(|e| StrandsError::Other(format!("sse utf8: {e}")))?;
            buf.push_str(chunk);

            let mut events: Vec<Result<StreamEvent, StrandsError>> = Vec::new();
            // SSE messages are separated by blank lines (\n\n).
            while let Some(end) = buf.find("\n\n") {
                let frame = buf[..end].to_string();
                buf.drain(..end + 2);
                for line in frame.lines() {
                    let line = line.trim();
                    if let Some(payload) = line.strip_prefix("data:") {
                        let payload = payload.trim();
                        if payload.is_empty() || payload == "[DONE]" {
                            continue;
                        }
                        match serde_json::from_str::<StreamChunk>(payload) {
                            Ok(chunk) => {
                                for e in state.ingest(chunk) {
                                    events.push(Ok(e));
                                }
                            }
                            Err(e) => {
                                tracing::warn!("openrouter chunk decode: {e} payload={payload}");
                            }
                        }
                    }
                }
            }
            Ok::<Vec<Result<StreamEvent, StrandsError>>, StrandsError>(events)
        })
        .flat_map(|item| match item {
            Ok(events) => stream::iter(events),
            Err(e) => stream::iter(vec![Err(e)]),
        })
}
