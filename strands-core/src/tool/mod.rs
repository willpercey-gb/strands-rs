pub mod executor;
/// Schema-constrained final answers.
pub mod structured_output;

use async_trait::async_trait;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::StrandsError;
use crate::types::tools::ToolSpec;

/// Context available to tools during execution.
#[derive(Debug, Clone, Default)]
pub struct ToolContext {
    /// Shared state accessible across tools within a single invocation.
    pub state: Value,
}

/// The result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    /// What the tool produced.
    pub content: Value,
    /// Whether this represents a failure. Surfaced to the model either way.
    pub is_error: bool,
}

impl ToolOutput {
    /// A successful result.
    pub fn success(content: impl Into<Value>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    /// A failed result the model can read and react to.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: Value::String(message.into()),
            is_error: true,
        }
    }
}

pub use executor::{ConcurrentToolExecutor, SequentialToolExecutor, ToolCall, ToolExecutor};
pub use structured_output::{
    StructuredOutputSlot, StructuredOutputSpec, StructuredOutputTool,
    DEFAULT_STRUCTURED_OUTPUT_PROMPT,
};

/// Implement this trait to define a tool the agent can invoke.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Unique name matching the tool spec.
    fn name(&self) -> &str;

    /// The tool's specification (name, description, input schema).
    fn spec(&self) -> ToolSpec;

    /// Execute the tool with the given JSON input.
    async fn invoke(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, StrandsError>;
}

// ---------------------------------------------------------------------------
// FnTool — closure-based tool for quick definitions without a struct
// ---------------------------------------------------------------------------

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
type InvokeFn = Arc<
    dyn Fn(Value, &ToolContext) -> BoxFuture<'_, Result<ToolOutput, StrandsError>> + Send + Sync,
>;

/// A tool backed by a closure. Convenient for simple tools that don't need
/// their own struct.
pub struct FnTool {
    tool_spec: ToolSpec,
    invoke_fn: InvokeFn,
}

impl FnTool {
    /// Create a new instance.
    pub fn new<F, Fut>(name: &str, description: &str, input_schema: Value, f: F) -> Self
    where
        F: Fn(Value, &ToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolOutput, StrandsError>> + Send + 'static,
    {
        let f = Arc::new(f);
        Self {
            tool_spec: ToolSpec::new(name, description, input_schema),
            invoke_fn: Arc::new(move |input, ctx| {
                let f = Arc::clone(&f);
                Box::pin(async move { f(input, ctx).await })
            }),
        }
    }
}

impl std::fmt::Debug for FnTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnTool")
            .field("spec", &self.tool_spec)
            .finish()
    }
}

#[async_trait]
impl Tool for FnTool {
    /// Name, for logs and diagnostics.
    fn name(&self) -> &str {
        &self.tool_spec.name
    }

    fn spec(&self) -> ToolSpec {
        self.tool_spec.clone()
    }

    async fn invoke(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, StrandsError> {
        (self.invoke_fn)(input, ctx).await
    }
}
