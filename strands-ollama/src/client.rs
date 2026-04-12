use async_trait::async_trait;
use futures::stream;
use futures::TryStreamExt;
use reqwest::Client;
use strands_core::model::{Model, ModelStream};
use strands_core::types::message::{Message, Role};
use strands_core::types::streaming::{
    ContentBlockType, DeltaContent, StopReason, StreamEvent, Usage,
};
use strands_core::types::tools::ToolSpec;
use strands_core::{ContentBlock, StrandsError};
use tracing::debug;

use crate::types::*;

/// Model adapter for Ollama's `/api/chat` endpoint.
pub struct OllamaModel {
    host: String,
    model_name: String,
    client: Client,
    options: Option<OllamaRequestOptions>,
}

impl OllamaModel {
    /// Create a new Ollama model adapter with the given model name.
    /// Defaults to `http://localhost:11434`.
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            host: "http://localhost:11434".to_string(),
            model_name: model_name.into(),
            client: Client::new(),
            options: None,
        }
    }

    /// Set the Ollama host URL.
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Set model options (temperature, top_p, etc.).
    pub fn with_options(mut self, opts: OllamaRequestOptions) -> Self {
        self.options = Some(opts);
        self
    }
}

impl OllamaModel {
    /// Convert strands Messages to Ollama format.
    fn convert_messages(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
    ) -> Vec<OllamaMessage> {
        let mut ollama_msgs = Vec::new();

        // Add system prompt as first message
        if let Some(prompt) = system_prompt {
            ollama_msgs.push(OllamaMessage {
                role: "system".to_string(),
                content: prompt.to_string(),
                tool_calls: None,
            });
        }

        for msg in messages {
            match msg.role {
                Role::System => {
                    let text = msg.text();
                    if !text.is_empty() {
                        ollama_msgs.push(OllamaMessage {
                            role: "system".to_string(),
                            content: text,
                            tool_calls: None,
                        });
                    }
                }
                Role::User => {
                    // User messages may contain text or tool results
                    let mut text_parts = Vec::new();
                    let mut tool_results = Vec::new();

                    for block in &msg.content {
                        match block {
                            ContentBlock::Text { text } => {
                                text_parts.push(text.clone());
                            }
                            ContentBlock::ToolResult {
                                tool_use_id: _,
                                status: _,
                                content,
                            } => {
                                // Find the tool name from the preceding assistant message
                                // For simplicity, emit as a tool role message
                                let result_text: String = content
                                    .iter()
                                    .map(|c| match c {
                                        strands_core::types::content::ToolResultContent::Text {
                                            text,
                                        } => text.clone(),
                                        strands_core::types::content::ToolResultContent::Json {
                                            value,
                                        } => serde_json::to_string(value).unwrap_or_default(),
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n");
                                tool_results.push(result_text);
                            }
                            _ => {}
                        }
                    }

                    // Emit tool results as tool role messages
                    for result in tool_results {
                        ollama_msgs.push(OllamaMessage {
                            role: "tool".to_string(),
                            content: result,
                            tool_calls: None,
                        });
                    }

                    // Emit text content as user message
                    let text = text_parts.join("\n");
                    if !text.is_empty() {
                        ollama_msgs.push(OllamaMessage {
                            role: "user".to_string(),
                            content: text,
                            tool_calls: None,
                        });
                    }
                }
                Role::Assistant => {
                    let text = msg.text();
                    let tool_calls: Vec<OllamaToolCall> = msg
                        .content
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::ToolUse { name, input, .. } => {
                                Some(OllamaToolCall {
                                    function: OllamaFunctionCall {
                                        name: name.clone(),
                                        arguments: input.clone(),
                                    },
                                })
                            }
                            _ => None,
                        })
                        .collect();

                    ollama_msgs.push(OllamaMessage {
                        role: "assistant".to_string(),
                        content: text,
                        tool_calls: if tool_calls.is_empty() {
                            None
                        } else {
                            Some(tool_calls)
                        },
                    });
                }
            }
        }

        ollama_msgs
    }

    /// Convert strands ToolSpecs to Ollama format.
    fn convert_tools(&self, specs: &[ToolSpec]) -> Vec<OllamaTool> {
        specs
            .iter()
            .map(|spec| OllamaTool {
                tool_type: "function".to_string(),
                function: OllamaFunctionDef {
                    name: spec.name.clone(),
                    description: spec.description.clone(),
                    parameters: spec.input_schema.clone(),
                },
            })
            .collect()
    }

}

#[async_trait]
impl Model for OllamaModel {
    async fn stream(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
        tool_specs: &[ToolSpec],
    ) -> Result<ModelStream, StrandsError> {
        let ollama_messages = self.convert_messages(messages, system_prompt);
        let ollama_tools = self.convert_tools(tool_specs);

        let request = ChatRequest {
            model: self.model_name.clone(),
            messages: ollama_messages,
            tools: ollama_tools,
            stream: true,
            options: self.options.clone(),
        };

        let url = format!("{}/api/chat", self.host);
        debug!(url, model = %self.model_name, "Sending request to Ollama");

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| StrandsError::Model(format!("HTTP error: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            return Err(StrandsError::Model(format!(
                "Ollama returned {status}: {body}"
            )));
        }

        // Ollama streams newline-delimited JSON chunks.
        // We read all chunks, accumulating text deltas. The final chunk
        // (done: true) carries tool calls and usage metadata.
        // We collect everything into StreamEvents and return them as a stream.
        let mut bytes_stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut all_events: Vec<Result<StreamEvent, StrandsError>> = Vec::new();
        let mut accumulated_text = String::new();
        let mut final_chunk: Option<ChatResponse> = None;

        loop {
            match bytes_stream.try_next().await {
                Ok(Some(bytes)) => {
                    buffer.push_str(&String::from_utf8_lossy(&bytes));

                    // Process complete lines
                    while let Some(newline_pos) = buffer.find('\n') {
                        let line = buffer[..newline_pos].trim().to_string();
                        buffer = buffer[newline_pos + 1..].to_string();

                        if line.is_empty() {
                            continue;
                        }

                        match serde_json::from_str::<ChatResponse>(&line) {
                            Ok(chunk) => {
                                if !chunk.message.content.is_empty() {
                                    accumulated_text.push_str(&chunk.message.content);
                                }
                                if chunk.done {
                                    final_chunk = Some(chunk);
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Failed to parse Ollama chunk: {e}");
                            }
                        }
                    }
                }
                Ok(None) => {
                    // Process remaining buffer
                    let remaining = buffer.trim().to_string();
                    if !remaining.is_empty() {
                        if let Ok(chunk) = serde_json::from_str::<ChatResponse>(&remaining) {
                            if !chunk.message.content.is_empty() {
                                accumulated_text.push_str(&chunk.message.content);
                            }
                            if chunk.done {
                                final_chunk = Some(chunk);
                            }
                        }
                    }
                    break;
                }
                Err(e) => {
                    all_events.push(Err(StrandsError::Model(format!("Stream error: {e}"))));
                    break;
                }
            }
        }

        // Build events from accumulated data
        all_events.push(Ok(StreamEvent::MessageStart {
            role: Role::Assistant,
        }));

        let mut block_index = 0;

        // Text content
        if !accumulated_text.is_empty() {
            all_events.push(Ok(StreamEvent::ContentBlockStart {
                index: block_index,
                content_type: ContentBlockType::Text,
            }));
            all_events.push(Ok(StreamEvent::ContentBlockDelta {
                index: block_index,
                delta: DeltaContent::TextDelta(accumulated_text),
            }));
            all_events.push(Ok(StreamEvent::ContentBlockStop {
                index: block_index,
            }));
            block_index += 1;
        }

        // Tool calls from final chunk
        let mut has_tool_calls = false;
        if let Some(ref chunk) = final_chunk {
            if let Some(ref tool_calls) = chunk.message.tool_calls {
                for tc in tool_calls {
                    has_tool_calls = true;
                    let tool_use_id = format!("call_{}", uuid::Uuid::new_v4());
                    all_events.push(Ok(StreamEvent::ContentBlockStart {
                        index: block_index,
                        content_type: ContentBlockType::ToolUse {
                            tool_use_id,
                            name: tc.function.name.clone(),
                        },
                    }));
                    let input_json =
                        serde_json::to_string(&tc.function.arguments).unwrap_or_default();
                    all_events.push(Ok(StreamEvent::ContentBlockDelta {
                        index: block_index,
                        delta: DeltaContent::ToolInputDelta(input_json),
                    }));
                    all_events.push(Ok(StreamEvent::ContentBlockStop {
                        index: block_index,
                    }));
                    block_index += 1;
                }
            }
        }

        let stop_reason = if has_tool_calls {
            StopReason::ToolUse
        } else {
            StopReason::EndTurn
        };
        all_events.push(Ok(StreamEvent::MessageStop { stop_reason }));

        // Usage metadata
        if let Some(chunk) = final_chunk {
            all_events.push(Ok(StreamEvent::Metadata {
                usage: Usage {
                    input_tokens: chunk.prompt_eval_count,
                    output_tokens: chunk.eval_count,
                    total_duration_ns: chunk.total_duration,
                },
            }));
        }

        Ok(Box::pin(stream::iter(all_events)))
    }
}
