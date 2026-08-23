use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Schema describing a tool's capabilities and input shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Unique name the model calls this tool by.
    pub name: String,
    /// What the tool does. The model chooses from this, so it matters.
    pub description: String,
    /// JSON Schema for the arguments.
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

    /// Declare a schema for the tool's output.
    /// Set the output schema.
    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Attach behavioural hints.
    /// Set the annotations.
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
    /// How the model should pick among the available tools.
    pub tool_choice: ToolChoice,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
/// How the model should select a tool.
pub enum ToolChoice {
    /// The model decides whether and which tool to call.
    #[default]
    Auto,
    /// The model must call some tool.
    Any,
    /// The model must not call a tool.
    None,
    /// The model must call this specific tool.
    Specific {
        /// Name of the required tool.
        name: String,
    },
}

/// Maximum nesting depth accepted in a tool input schema.
///
/// Schemas can arrive from untrusted sources (an MCP server, a remote tool
/// registry). A schema nested this deep is far more likely to be malformed or
/// hostile than genuinely needed, and rejecting it early keeps downstream
/// consumers — provider request builders, schema walkers — off inputs they may
/// well handle recursively.
///
/// Note this is not the only line of defence: `serde_json`'s parser already
/// caps nesting at 128 levels by default, so a schema *parsed* from untrusted
/// text cannot reach arbitrary depth in the first place. This check covers
/// programmatically constructed schemas and lowers the bound to something
/// deliberate.
pub const MAX_SCHEMA_DEPTH: usize = 32;

/// Whether a JSON schema nests no deeper than `max_depth`.
///
/// Iterative rather than recursive, so the check adds no stack depth of its own
/// on top of whatever the value already required.
pub fn schema_within_depth(schema: &Value, max_depth: usize) -> bool {
    let mut stack = vec![(schema, 1usize)];

    while let Some((node, depth)) = stack.pop() {
        if depth > max_depth {
            return false;
        }
        match node {
            Value::Object(map) => {
                for value in map.values() {
                    stack.push((value, depth + 1));
                }
            }
            Value::Array(items) => {
                for value in items {
                    stack.push((value, depth + 1));
                }
            }
            _ => {}
        }
    }

    true
}

impl ToolSpec {
    /// Whether this spec's input schema is within [`MAX_SCHEMA_DEPTH`].
    pub fn has_valid_schema_depth(&self) -> bool {
        schema_within_depth(&self.input_schema, MAX_SCHEMA_DEPTH)
    }
}

#[cfg(test)]
mod schema_depth_tests {
    use super::*;
    use serde_json::json;

    /// Build an object nested `depth` levels deep.
    fn nested(depth: usize) -> Value {
        let mut value = json!({"type": "string"});
        for _ in 0..depth {
            value = json!({"type": "object", "properties": {"inner": value}});
        }
        value
    }

    #[test]
    fn shallow_schemas_pass() {
        assert!(schema_within_depth(&json!({"type": "object"}), 32));
        assert!(schema_within_depth(&nested(3), 32));
    }

    #[test]
    fn deeply_nested_schemas_are_rejected() {
        assert!(!schema_within_depth(&nested(200), MAX_SCHEMA_DEPTH));
    }

    #[test]
    fn the_check_adds_no_stack_depth_of_its_own() {
        // Deep enough to be well past the limit, but within what serde_json
        // itself can hold: `Value`'s own Drop recurses, so a truly enormous
        // fixture would overflow while being torn down rather than testing
        // anything about this function.
        assert!(!schema_within_depth(&nested(400), MAX_SCHEMA_DEPTH));
    }

    #[test]
    fn depth_is_measured_from_the_root() {
        // The root itself counts as level 1, so a bare object fits in 1.
        assert!(schema_within_depth(&json!({}), 1));
        assert!(!schema_within_depth(&json!({"a": {"b": 1}}), 2));
        assert!(schema_within_depth(&json!({"a": {"b": 1}}), 3));
    }

    #[test]
    fn arrays_count_toward_depth() {
        let mut value = json!("leaf");
        for _ in 0..50 {
            value = json!([value]);
        }
        assert!(!schema_within_depth(&value, MAX_SCHEMA_DEPTH));
    }

    #[test]
    fn scalars_are_trivially_within_depth() {
        assert!(schema_within_depth(&json!(1), 1));
        assert!(schema_within_depth(&Value::Null, 1));
    }

    #[test]
    fn tool_spec_exposes_the_check() {
        assert!(ToolSpec::new("t", "d", json!({"type": "object"})).has_valid_schema_depth());
        assert!(!ToolSpec::new("t", "d", nested(200)).has_valid_schema_depth());
    }
}
