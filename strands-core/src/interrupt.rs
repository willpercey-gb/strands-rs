//! Human-in-the-loop interrupts.
//!
//! A hook can pause the agent to ask a human something — approve this deletion,
//! pick between these options — and the run resumes once an answer is supplied.
//!
//! # Flow
//!
//! 1. A hook calls [`InterruptState::interrupt`] with a name and a reason.
//! 2. It returns `None` (no answer yet), so the hook returns without acting.
//! 3. The loop sees an unanswered interrupt, stops with
//!    [`StopReason::Interrupt`](crate::types::streaming::StopReason::Interrupt),
//!    and returns the pending interrupts on the result.
//! 4. The caller answers via [`Agent::respond`](crate::Agent::respond) and
//!    re-invokes.
//! 5. The same hook runs again; this time `interrupt` returns the answer and
//!    the hook proceeds.
//!
//! # Difference from upstream
//!
//! Upstream raises an exception out of the hook callback and re-enters it on
//! resume. Rust hooks are plain `Fn(&mut HookEvent)` with no unwinding
//! contract, so the request is *recorded* on the event instead and the hook
//! returns normally. The observable handshake is the same; what differs is that
//! a hook must be written to tolerate being called with no answer yet, which
//! the `Option` return makes impossible to overlook.
//!
//! Ported from upstream `interrupt.py` and `types/interrupt.py`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A pause point awaiting a human answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Interrupt {
    /// Stable id, used to match a response back to the request.
    pub id: String,
    /// Caller-defined name. Unique within one invocation.
    pub name: String,
    /// Why the hook paused — shown to whoever answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<Value>,
    /// The human's answer, once supplied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<Value>,
}

impl Interrupt {
    /// Create a new instance.
    pub fn new(name: impl Into<String>, reason: Option<Value>) -> Self {
        let name = name.into();
        Self {
            id: format!("{}:{}", name, uuid::Uuid::new_v4()),
            name,
            reason,
            response: None,
        }
    }

    /// Whether a human has answered this yet.
    pub fn is_answered(&self) -> bool {
        self.response.is_some()
    }
}

/// A response supplied by the caller when resuming.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterruptResponse {
    /// Id of the interrupt being answered.
    pub interrupt_id: String,
    /// The answer.
    pub response: Value,
}

impl InterruptResponse {
    /// Create a new instance.
    pub fn new(interrupt_id: impl Into<String>, response: impl Into<Value>) -> Self {
        Self {
            interrupt_id: interrupt_id.into(),
            response: response.into(),
        }
    }
}

/// Interrupts raised and answered across an invocation.
///
/// Carried on hook events that support pausing, and retained on the agent
/// between invocations so a resumed run can find its answers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InterruptState {
    /// Interrupts by name.
    interrupts: HashMap<String, Interrupt>,
}

impl InterruptState {
    /// Create with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request human input, or collect the answer if one has been supplied.
    ///
    /// Returns `None` the first time, and the answer after the caller resumes.
    /// A hook must handle the `None` case by *not* taking the action it was
    /// asking about — that is the entire safety property, so it is a return
    /// value rather than something a hook can forget to check.
    pub fn interrupt(&mut self, name: &str, reason: Option<Value>) -> Option<Value> {
        if let Some(existing) = self.interrupts.get(name) {
            if let Some(response) = &existing.response {
                return Some(response.clone());
            }
            // Already pending; do not mint a second request for the same name.
            return None;
        }

        self.interrupts
            .insert(name.to_string(), Interrupt::new(name, reason));
        None
    }

    /// Interrupts still waiting for an answer.
    pub fn pending(&self) -> Vec<Interrupt> {
        let mut pending: Vec<Interrupt> = self
            .interrupts
            .values()
            .filter(|i| !i.is_answered())
            .cloned()
            .collect();
        // Stable order so a caller rendering these to a human sees the same
        // sequence on every read.
        pending.sort_by(|a, b| a.name.cmp(&b.name));
        pending
    }

    /// Whether anything is waiting on a human.
    pub fn has_pending(&self) -> bool {
        self.interrupts.values().any(|i| !i.is_answered())
    }

    /// Apply caller-supplied answers.
    ///
    /// Returns the number applied. Responses whose id matches nothing are
    /// ignored: a stale id from an earlier invocation should not fail the
    /// resume, and silently answering the wrong interrupt would be worse.
    pub fn respond(&mut self, responses: &[InterruptResponse]) -> usize {
        let mut applied = 0;
        for response in responses {
            for interrupt in self.interrupts.values_mut() {
                if interrupt.id == response.interrupt_id {
                    interrupt.response = Some(response.response.clone());
                    applied += 1;
                    break;
                }
            }
        }
        applied
    }

    /// Drop unanswered interrupts, keeping answers for the rest of the cycle.
    ///
    /// Called when an invocation completes: an unanswered request from a run
    /// that has since finished would otherwise stall the next one.
    pub fn clear_unanswered(&mut self) {
        self.interrupts.retain(|_, i| i.is_answered());
    }

    /// Drop everything, answers included.
    pub fn clear(&mut self) {
        self.interrupts.clear();
    }

    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.interrupts.is_empty()
    }

    /// Look up an interrupt by name.
    pub fn get(&self, name: &str) -> Option<&Interrupt> {
        self.interrupts.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn first_request_returns_no_answer() {
        let mut state = InterruptState::new();
        assert_eq!(state.interrupt("approve", Some(json!("delete?"))), None);
        assert!(state.has_pending());
        assert_eq!(state.pending().len(), 1);
    }

    #[test]
    fn repeated_requests_do_not_duplicate_the_interrupt() {
        // A hook re-runs on every cycle; each pass must not mint a new request.
        let mut state = InterruptState::new();
        state.interrupt("approve", None);
        state.interrupt("approve", None);
        state.interrupt("approve", None);
        assert_eq!(state.pending().len(), 1);
    }

    #[test]
    fn answering_makes_the_response_available() {
        let mut state = InterruptState::new();
        state.interrupt("approve", Some(json!("delete?")));
        let id = state.pending()[0].id.clone();

        assert_eq!(state.respond(&[InterruptResponse::new(id, "yes")]), 1);
        assert!(!state.has_pending());
        assert_eq!(state.interrupt("approve", None), Some(json!("yes")));
    }

    #[test]
    fn an_unknown_response_id_is_ignored_not_misapplied() {
        // A stale id from an earlier run must neither fail the resume nor
        // answer some unrelated interrupt.
        let mut state = InterruptState::new();
        state.interrupt("approve", None);

        assert_eq!(state.respond(&[InterruptResponse::new("nope", "yes")]), 0);
        assert!(state.has_pending());
        assert_eq!(state.interrupt("approve", None), None);
    }

    #[test]
    fn responses_are_matched_by_id_not_position() {
        let mut state = InterruptState::new();
        state.interrupt("a", None);
        state.interrupt("b", None);

        let b_id = state.get("b").unwrap().id.clone();
        state.respond(&[InterruptResponse::new(b_id, "answer-b")]);

        assert_eq!(state.interrupt("b", None), Some(json!("answer-b")));
        assert_eq!(state.interrupt("a", None), None, "a must remain unanswered");
    }

    #[test]
    fn pending_is_sorted_for_stable_presentation() {
        let mut state = InterruptState::new();
        for name in ["zebra", "alpha", "mike"] {
            state.interrupt(name, None);
        }
        let pending = state.pending();
        let names: Vec<&str> = pending.iter().map(|i| i.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "mike", "zebra"]);
    }

    #[test]
    fn clear_unanswered_keeps_answers() {
        let mut state = InterruptState::new();
        state.interrupt("answered", None);
        state.interrupt("pending", None);

        let id = state.get("answered").unwrap().id.clone();
        state.respond(&[InterruptResponse::new(id, "yes")]);
        state.clear_unanswered();

        assert!(!state.has_pending());
        assert_eq!(state.interrupt("answered", None), Some(json!("yes")));
        // The dropped one starts fresh next time rather than stalling the run.
        assert_eq!(state.interrupt("pending", None), None);
    }

    #[test]
    fn ids_are_unique_across_interrupts() {
        let mut state = InterruptState::new();
        state.interrupt("a", None);
        state.interrupt("b", None);

        let ids: Vec<String> = state.pending().iter().map(|i| i.id.clone()).collect();
        assert_ne!(ids[0], ids[1]);
    }

    #[test]
    fn state_survives_a_serde_round_trip() {
        // Interrupts are session-managed between raise and response, so they
        // must persist across a process restart.
        let mut state = InterruptState::new();
        state.interrupt("approve", Some(json!({"key": "X"})));

        let text = serde_json::to_string(&state).unwrap();
        let mut back: InterruptState = serde_json::from_str(&text).unwrap();

        assert_eq!(back.pending().len(), 1);
        assert_eq!(back.pending()[0].reason, Some(json!({"key": "X"})));

        let id = back.pending()[0].id.clone();
        back.respond(&[InterruptResponse::new(id, "A")]);
        assert_eq!(back.interrupt("approve", None), Some(json!("A")));
    }
}
