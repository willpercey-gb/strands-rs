//! Structured output — constraining a model's final answer to a schema.
//!
//! The mechanism is a synthetic tool: the schema is advertised as a tool the
//! model must call, and the "arguments" it produces are the structured answer.
//! This reuses the provider's own constrained-decoding path, which is far more
//! reliable than asking for JSON in a prompt and parsing whatever comes back.
//!
//! Validation failures are returned to the model as tool **errors**, not raised
//! to the caller. A model that gets its own schema wrong can usually fix it on
//! the next turn if it is told how — failing the invocation outright throws that
//! away.
//!
//! Upstream builds the schema from a pydantic model. Rust has no equivalent, so
//! the schema is supplied explicitly and the result is validated by
//! deserializing into `T`.
//!
//! Ported from upstream `tools/structured_output/`.

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tracing::debug;

use crate::error::StrandsError;
use crate::types::tools::ToolSpec;

use super::{Tool, ToolContext, ToolOutput};

/// Prompt used to force a final structured answer when the model finished
/// without producing one.
pub const DEFAULT_STRUCTURED_OUTPUT_PROMPT: &str =
    "You must format the previous response as structured output.";

/// Describes the shape a structured answer must take.
#[derive(Debug, Clone)]
pub struct StructuredOutputSpec {
    /// Tool name presented to the model. Conventionally the type's name.
    pub name: String,
    /// What the structure represents. Shown to the model.
    pub description: String,
    /// JSON Schema for the structure.
    pub schema: Value,
}

impl StructuredOutputSpec {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            schema,
        }
    }

    /// The tool spec advertised to the model.
    ///
    /// The description is prefixed to steer the model to call this last —
    /// without it, models routinely emit the final answer partway through and
    /// then keep working.
    pub fn tool_spec(&self) -> ToolSpec {
        ToolSpec::new(
            self.name.clone(),
            format!(
                "IMPORTANT: Call this only as the final step, once you have the \
                 completed result to return to the caller. <description>{}</description>",
                self.description
            ),
            self.schema.clone(),
        )
    }
}

/// Where a validated structured result is deposited.
///
/// Shared with the tool so the agent loop can collect the answer after the
/// invocation without the tool needing a reference back to the agent.
pub struct StructuredOutputSlot<T> {
    value: Arc<Mutex<Option<T>>>,
}

impl<T> Clone for StructuredOutputSlot<T> {
    fn clone(&self) -> Self {
        Self {
            value: Arc::clone(&self.value),
        }
    }
}

impl<T> Default for StructuredOutputSlot<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> StructuredOutputSlot<T> {
    pub fn new() -> Self {
        Self {
            value: Arc::new(Mutex::new(None)),
        }
    }

    /// Take the stored value, leaving the slot empty.
    pub fn take(&self) -> Option<T> {
        self.value.lock().ok()?.take()
    }

    /// Whether a value has been produced.
    pub fn is_filled(&self) -> bool {
        self.value.lock().map(|v| v.is_some()).unwrap_or(false)
    }

    fn store(&self, value: T) {
        if let Ok(mut slot) = self.value.lock() {
            *slot = Some(value);
        }
    }
}

/// A tool whose "arguments" are the model's structured final answer.
pub struct StructuredOutputTool<T> {
    spec: StructuredOutputSpec,
    slot: StructuredOutputSlot<T>,
    _marker: PhantomData<fn() -> T>,
}

impl<T> StructuredOutputTool<T> {
    pub fn new(spec: StructuredOutputSpec, slot: StructuredOutputSlot<T>) -> Self {
        Self {
            spec,
            slot,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<T> Tool for StructuredOutputTool<T>
where
    T: DeserializeOwned + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.spec.name
    }

    fn spec(&self) -> ToolSpec {
        self.spec.tool_spec()
    }

    async fn invoke(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, StrandsError> {
        match serde_json::from_value::<T>(input) {
            Ok(value) => {
                debug!(tool_name = %self.spec.name, "Structured output validated");
                self.slot.store(value);
                Ok(ToolOutput::success(format!(
                    "Successfully validated {} structured output",
                    self.spec.name
                )))
            }
            Err(e) => {
                // Handed back to the model, not raised: it can usually correct
                // its own schema mistake once told what was wrong.
                let message = format!(
                    "Validation failed for {}. Please fix the following and call the tool again:\n- {e}",
                    self.spec.name
                );
                debug!(tool_name = %self.spec.name, error = %e, "Structured output validation failed");
                Ok(ToolOutput::error(message))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde_json::json;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Person {
        name: String,
        age: u32,
    }

    fn spec() -> StructuredOutputSpec {
        StructuredOutputSpec::new(
            "Person",
            "A person record",
            json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }),
        )
    }

    #[tokio::test]
    async fn valid_input_fills_the_slot() {
        let slot = StructuredOutputSlot::<Person>::new();
        let tool = StructuredOutputTool::new(spec(), slot.clone());

        let out = tool
            .invoke(json!({"name": "Ada", "age": 36}), &ToolContext::default())
            .await
            .unwrap();

        assert!(!out.is_error);
        assert_eq!(
            slot.take(),
            Some(Person {
                name: "Ada".into(),
                age: 36
            })
        );
    }

    #[tokio::test]
    async fn invalid_input_is_an_error_result_not_a_failure() {
        // The model must get a chance to correct itself; raising here would
        // discard the whole invocation over a fixable mistake.
        let slot = StructuredOutputSlot::<Person>::new();
        let tool = StructuredOutputTool::new(spec(), slot.clone());

        let out = tool
            .invoke(json!({"name": "Ada"}), &ToolContext::default())
            .await
            .expect("validation failure must not surface as an Err");

        assert!(out.is_error);
        assert!(!slot.is_filled());

        let text = out.content.as_str().unwrap_or_default();
        assert!(
            text.contains("Person") && text.contains("age"),
            "the error should name the type and the offending field: {text}"
        );
    }

    #[tokio::test]
    async fn wrong_type_is_also_recoverable() {
        let slot = StructuredOutputSlot::<Person>::new();
        let tool = StructuredOutputTool::new(spec(), slot.clone());

        let out = tool
            .invoke(
                json!({"name": "Ada", "age": "thirty-six"}),
                &ToolContext::default(),
            )
            .await
            .unwrap();

        assert!(out.is_error);
        assert!(!slot.is_filled());
    }

    #[test]
    fn tool_spec_steers_the_model_to_call_it_last() {
        let spec = spec().tool_spec();
        assert_eq!(spec.name, "Person");
        assert!(
            spec.description.contains("final step"),
            "without this steer, models emit the answer mid-run and keep going"
        );
        assert!(spec.description.contains("A person record"));
    }

    #[test]
    fn slot_take_empties_it() {
        let slot = StructuredOutputSlot::<Person>::new();
        slot.store(Person {
            name: "a".into(),
            age: 1,
        });
        assert!(slot.is_filled());
        assert!(slot.take().is_some());
        assert!(!slot.is_filled());
        assert!(slot.take().is_none());
    }

    #[test]
    fn clones_share_one_slot() {
        let slot = StructuredOutputSlot::<Person>::new();
        let other = slot.clone();
        other.store(Person {
            name: "a".into(),
            age: 1,
        });
        assert!(slot.is_filled(), "a clone must write through to the original");
    }
}
