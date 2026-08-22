use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Schema describing a tool's capabilities and input shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    /// Optional JSON Schema for the tool's output.
    ///
    /// Not every provider accepts this; adapters that do not support it must
    /// filter it out rather than forwarding it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
    /// Behavioural hints from the tool provider, e.g. MCP's `readOnlyHint`
    /// or `destructiveHint`.
    ///
    /// These are **untrusted hints, not guarantees** — a permission layer must
    /// never treat them as a security boundary. A missing key means unknown,
    /// not false: per the MCP spec `destructiveHint` and `openWorldHint`
    /// default to true when absent, while `readOnlyHint` and `idempotentHint`
    /// default to false. Absent entirely for non-MCP tools. Never sent to
    /// provider APIs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<ToolAnnotations>,
}

impl ToolSpec {
    /// Build a spec with no output schema or annotations.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            output_schema: None,
            annotations: None,
        }
    }

    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    pub fn with_annotations(mut self, annotations: ToolAnnotations) -> Self {
        self.annotations = Some(annotations);
        self
    }
}

/// MCP-style behavioural hints about a tool.
///
/// Every field is `Option` because absence is meaningful — see
/// [`ToolSpec::annotations`] for the defaults each key implies.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolAnnotations {
    /// Human-readable title for the tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Hint: the tool does not modify its environment. Defaults to false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_only_hint: Option<bool>,
    /// Hint: the tool may perform destructive updates. Defaults to **true**.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub destructive_hint: Option<bool>,
    /// Hint: repeated calls with the same arguments have no additional effect.
    /// Defaults to false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idempotent_hint: Option<bool>,
    /// Hint: the tool interacts with an open world (e.g. the internet).
    /// Defaults to **true**.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub open_world_hint: Option<bool>,
}

impl ToolAnnotations {
    /// Resolve `readOnlyHint`, applying the MCP default (false) when absent.
    pub fn is_read_only(&self) -> bool {
        self.read_only_hint.unwrap_or(false)
    }

    /// Resolve `destructiveHint`, applying the MCP default (**true**) when absent.
    pub fn is_destructive(&self) -> bool {
        self.destructive_hint.unwrap_or(true)
    }

    /// Resolve `idempotentHint`, applying the MCP default (false) when absent.
    pub fn is_idempotent(&self) -> bool {
        self.idempotent_hint.unwrap_or(false)
    }

    /// Resolve `openWorldHint`, applying the MCP default (**true**) when absent.
    pub fn is_open_world(&self) -> bool {
        self.open_world_hint.unwrap_or(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_annotations_use_mcp_defaults_not_false() {
        let a = ToolAnnotations::default();
        assert!(!a.is_read_only());
        assert!(!a.is_idempotent());
        // The permissive-looking fields default to true, which is the
        // conservative reading: assume a tool is dangerous unless told otherwise.
        assert!(a.is_destructive());
        assert!(a.is_open_world());
    }

    #[test]
    fn explicit_annotations_override_defaults() {
        let a = ToolAnnotations {
            destructive_hint: Some(false),
            read_only_hint: Some(true),
            ..Default::default()
        };
        assert!(!a.is_destructive());
        assert!(a.is_read_only());
    }

    #[test]
    fn spec_omits_optional_fields_when_unset() {
        let spec = ToolSpec::new("t", "d", serde_json::json!({}));
        let json = serde_json::to_string(&spec).unwrap();
        assert!(!json.contains("annotations"), "{json}");
        assert!(!json.contains("output_schema"), "{json}");
    }
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
