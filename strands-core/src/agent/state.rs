//! Durable per-agent key/value state.
//!
//! Distinct from the per-invocation `invocation_state` the tool context
//! carries: this survives across `prompt()` calls and is persisted with the
//! session, so it is where an agent keeps things it needs to remember between
//! turns.
//!
//! Ported from upstream `agent/state.py`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A JSON-valued store attached to an agent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AgentState {
    values: HashMap<String, Value>,
}

impl AgentState {
    /// Create with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build state from an existing map.
    pub fn from_map(values: HashMap<String, Value>) -> Self {
        Self { values }
    }

    /// Read a value by key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.values.get(key)
    }

    /// Read a value, deserialized into `T`.
    ///
    /// Returns `None` both when the key is absent and when the stored value
    /// does not fit `T` — callers wanting to tell those apart should use
    /// [`get`](Self::get).
    pub fn get_as<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        serde_json::from_value(self.values.get(key)?.clone()).ok()
    }

    /// Insert or replace a value, returning the previous one.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<Value>) -> Option<Value> {
        self.values.insert(key.into(), value.into())
    }

    /// Insert or replace a value by serializing `T`.
    ///
    /// Fails when `T` cannot be represented as JSON, rather than storing a
    /// partial value — the store is persisted verbatim, so an unserializable
    /// entry would surface as a corrupt session later rather than here.
    pub fn set_json<T: Serialize>(
        &mut self,
        key: impl Into<String>,
        value: &T,
    ) -> Result<Option<Value>, serde_json::Error> {
        Ok(self.values.insert(key.into(), serde_json::to_value(value)?))
    }

    /// Remove a value, returning it.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.values.remove(key)
    }

    /// Whether a value exists for this key.
    pub fn contains_key(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    /// Iterate the keys.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.values.keys()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Remove every entry.
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Borrow the underlying map.
    pub fn as_map(&self) -> &HashMap<String, Value> {
        &self.values
    }
}

impl From<HashMap<String, Value>> for AgentState {
    fn from(values: HashMap<String, Value>) -> Self {
        Self::from_map(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn set_and_get_round_trip() {
        let mut s = AgentState::new();
        assert!(s.is_empty());

        s.set("count", 3);
        assert_eq!(s.get("count"), Some(&json!(3)));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn set_returns_the_previous_value() {
        let mut s = AgentState::new();
        assert_eq!(s.set("k", 1), None);
        assert_eq!(s.set("k", 2), Some(json!(1)));
        assert_eq!(s.get("k"), Some(&json!(2)));
    }

    #[test]
    fn typed_access_deserializes() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct Profile {
            name: String,
            runs: u32,
        }

        let mut s = AgentState::new();
        s.set_json(
            "profile",
            &Profile {
                name: "a".into(),
                runs: 2,
            },
        )
        .unwrap();

        assert_eq!(
            s.get_as::<Profile>("profile"),
            Some(Profile {
                name: "a".into(),
                runs: 2
            })
        );
    }

    #[test]
    fn typed_access_returns_none_on_a_shape_mismatch() {
        let mut s = AgentState::new();
        s.set("n", "not a number");
        assert_eq!(s.get_as::<u32>("n"), None);
        assert_eq!(s.get_as::<u32>("missing"), None);
    }

    #[test]
    fn remove_and_clear() {
        let mut s = AgentState::new();
        s.set("a", 1);
        s.set("b", 2);

        assert_eq!(s.remove("a"), Some(json!(1)));
        assert!(!s.contains_key("a"));

        s.clear();
        assert!(s.is_empty());
    }

    #[test]
    fn serializes_as_a_plain_object() {
        // Transparent serialization keeps sessions readable and lets state
        // written by another SDK load here unchanged.
        let mut s = AgentState::new();
        s.set("k", "v");

        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, r#"{"k":"v"}"#);

        let back: AgentState = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }
}
