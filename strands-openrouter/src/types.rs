use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ChatCompletionRequest<'a> {
    pub model: &'a str,
    pub messages: Vec<OaiMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<OaiTool>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum OaiContent {
    Text(String),
    Parts(Vec<OaiContentPart>),
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum OaiContentPart {
    Text {
        text: String,
    },
    #[allow(dead_code)]
    ImageUrl {
        image_url: OaiImageUrl,
    },
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct OaiImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct OaiMessage {
    pub role: String, // "system" | "user" | "assistant" | "tool"
    pub content: OaiContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<OaiToolCallEcho>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct OaiToolCallEcho {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str, // always "function"
    pub function: OaiToolFunctionEcho,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct OaiToolFunctionEcho {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct OaiTool {
    #[serde(rename = "type")]
    pub kind: &'static str, // always "function"
    pub function: OaiToolFunction,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct OaiToolFunction {
    pub name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub description: String,
    pub parameters: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Streaming-response wire types (SSE chunks)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StreamChunk {
    pub choices: Vec<StreamChoice>,
    #[serde(default)]
    pub usage: Option<StreamUsage>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StreamChoice {
    #[serde(default)]
    pub index: usize,
    pub delta: StreamDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct StreamDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<StreamDeltaToolCall>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StreamDeltaToolCall {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<StreamDeltaToolFn>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct StreamDeltaToolFn {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StreamUsage {
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub completion_tokens: Option<u64>,
}
