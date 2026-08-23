//! Exercises the `#[tool]` proc macro.
//!
//! The macro had been silently broken since the type changes in the v1.53 sync:
//! it emitted a `ToolSpec` struct literal, which stopped compiling once the
//! struct gained fields. Nothing in the workspace used it, so the whole test
//! suite passed against a macro that could not expand.
//!
//! These tests exist so that cannot recur — the macro is now compiled and
//! invoked by the suite itself.

#![cfg(feature = "macros")]

use strands_core::tool::{Tool, ToolContext};
use strands_core::tool;
use strands_core::StrandsError;

/// Get the current weather for a city.
///
/// # Arguments
///
/// * `city` - The city to check weather for
/// * `unit` - Temperature unit (celsius or fahrenheit)
#[tool]
async fn get_weather(
    city: String,
    unit: Option<String>,
) -> Result<String, StrandsError> {
    let unit = unit.unwrap_or_else(|| "celsius".into());
    Ok(format!("22 degrees {unit} in {city}"))
}

#[tool]
async fn add(a: i64, b: i64) -> Result<String, StrandsError> {
    Ok((a + b).to_string())
}

#[test]
fn the_macro_derives_a_usable_spec() {
    let spec = GetWeather.spec();

    assert_eq!(spec.name, "get_weather");
    assert!(spec.description.contains("weather"));
    assert_eq!(spec.input_schema["type"], "object");
}

#[test]
fn the_schema_describes_the_parameters() {
    let spec = GetWeather.spec();
    let properties = &spec.input_schema["properties"];

    assert!(properties.get("city").is_some(), "{properties}");
    assert!(properties.get("unit").is_some(), "{properties}");
}

#[test]
fn argument_docs_become_parameter_descriptions() {
    // Rust rejects doc comments in parameter position, so these can only come
    // from the function's `# Arguments` section.
    let spec = GetWeather.spec();
    let city = &spec.input_schema["properties"]["city"];

    assert_eq!(
        city["description"], "The city to check weather for",
        "the # Arguments entry should reach the schema: {city}"
    );
}

#[test]
fn the_description_excludes_the_arguments_section() {
    let spec = GetWeather.spec();
    assert_eq!(spec.description, "Get the current weather for a city.");
    assert!(
        !spec.description.contains("Arguments"),
        "the parameter list is not part of the tool description: {}",
        spec.description
    );
}

#[test]
fn optional_parameters_are_not_required() {
    let spec = GetWeather.spec();
    let required = spec.input_schema["required"]
        .as_array()
        .expect("required list");

    let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
    assert!(names.contains(&"city"));
    assert!(
        !names.contains(&"unit"),
        "an Option<T> parameter must not be required: {names:?}"
    );
}

#[test]
fn new_toolspec_fields_default_rather_than_breaking_the_macro() {
    // The regression that motivated this file: a struct literal in the macro
    // stops compiling the moment ToolSpec grows a field.
    let spec = GetWeather.spec();
    assert!(spec.annotations.is_none());
    assert!(spec.output_schema.is_none());
}

#[tokio::test]
async fn the_generated_tool_invokes() {
    let out = GetWeather
        .invoke(
            serde_json::json!({"city": "London"}),
            &ToolContext::default(),
        )
        .await
        .unwrap();

    assert!(!out.is_error);
    assert!(out.content.as_str().unwrap().contains("London"));
}

#[tokio::test]
async fn a_generated_tool_works_on_an_agent() {
    // The macro's whole point is dropping into a builder, so check that path
    // rather than only the trait impl.
    let names: Vec<String> = strands_core::Agent::builder()
        .model(NoopModel)
        .tool(GetWeather)
        .tool(Add)
        .build()
        .unwrap()
        .tool_names()
        .map(str::to_string)
        .collect();

    assert!(names.contains(&"get_weather".to_string()));
    assert!(names.contains(&"add".to_string()));
}

struct NoopModel;

#[async_trait::async_trait]
impl strands_core::model::Model for NoopModel {
    async fn stream(
        &self,
        _messages: &[strands_core::types::message::Message],
        _system_prompt: Option<&strands_core::types::content::SystemPrompt>,
        _tool_specs: &[strands_core::types::tools::ToolSpec],
    ) -> Result<strands_core::model::ModelStream, StrandsError> {
        Ok(Box::pin(futures::stream::iter(vec![])))
    }
}
