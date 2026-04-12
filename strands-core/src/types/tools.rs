use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Schema describing a tool's capabilities and input shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// Configuration for how the model should select tools.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolConfig {
    pub tool_choice: ToolChoice,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ToolChoice {
    #[default]
    Auto,
    Any,
    None,
    Specific {
        name: String,
    },
}
