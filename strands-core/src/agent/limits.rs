//! Per-invocation budget caps for the agent loop.
//!
//! Each cap bounds a single `prompt()` call; counters do not carry across
//! invocations of the same agent.
//!
//! Caps are checked at the *top* of each loop iteration, never mid-turn. That
//! ordering is deliberate: tools requested by the previous turn always run to
//! completion, so `agent.messages()` is left in a state the caller can re-invoke
//! from rather than with a dangling `ToolUse` no `ToolResult` ever answered.
//!
//! Ported from upstream `types/agent.py`.

use crate::types::streaming::{StopReason, Usage};

/// Budget caps for one invocation. An unset field means no limit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Limits {
    /// Maximum agent loop iterations. One turn is one model call plus any
    /// tool execution that follows it.
    pub turns: Option<usize>,
    /// Maximum cumulative model-generated tokens across every model call.
    ///
    /// A soft cap: it is checked at turn boundaries, so a single oversized
    /// response can overshoot. Distinct from a provider's per-call output cap,
    /// which bounds one response.
    pub output_tokens: Option<u64>,
    /// Maximum cumulative input + output tokens across every model call.
    ///
    /// Also a soft cap, for the same reason.
    pub total_tokens: Option<u64>,
}

impl Limits {
    /// No caps.
    pub fn none() -> Self {
        Self::default()
    }

    /// Cap the number of loop iterations.
    pub fn turns(turns: usize) -> Self {
        Self {
            turns: Some(turns),
            ..Self::default()
        }
    }

    pub fn with_turns(mut self, turns: usize) -> Self {
        self.turns = Some(turns);
        self
    }

    pub fn with_output_tokens(mut self, tokens: u64) -> Self {
        self.output_tokens = Some(tokens);
        self
    }

    pub fn with_total_tokens(mut self, tokens: u64) -> Self {
        self.total_tokens = Some(tokens);
        self
    }

    /// Whether any cap is set.
    pub fn is_unbounded(&self) -> bool {
        self.turns.is_none() && self.output_tokens.is_none() && self.total_tokens.is_none()
    }

    /// The stop reason for the first cap that has been reached, if any.
    ///
    /// Priority on a simultaneous trip is `turns`, then `total_tokens`, then
    /// `output_tokens` — matching upstream, so the same run reports the same
    /// reason across SDKs.
    pub fn exceeded(&self, turns_taken: usize, usage: &Usage) -> Option<StopReason> {
        if self.turns.is_some_and(|cap| turns_taken >= cap) {
            return Some(StopReason::LimitTurns);
        }
        if self
            .total_tokens
            .is_some_and(|cap| usage.total().unwrap_or(0) >= cap)
        {
            return Some(StopReason::LimitTotalTokens);
        }
        if self
            .output_tokens
            .is_some_and(|cap| usage.output_tokens.unwrap_or(0) >= cap)
        {
            return Some(StopReason::LimitOutputTokens);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(input: u64, output: u64) -> Usage {
        Usage {
            input_tokens: Some(input),
            output_tokens: Some(output),
            ..Default::default()
        }
    }

    #[test]
    fn no_caps_never_trip() {
        assert!(Limits::none().is_unbounded());
        assert_eq!(Limits::none().exceeded(1_000, &usage(1e9 as u64, 1e9 as u64)), None);
    }

    #[test]
    fn turn_cap_trips_on_reaching_the_limit() {
        let l = Limits::turns(3);
        assert_eq!(l.exceeded(2, &Usage::default()), None);
        assert_eq!(l.exceeded(3, &Usage::default()), Some(StopReason::LimitTurns));
        assert_eq!(l.exceeded(4, &Usage::default()), Some(StopReason::LimitTurns));
    }

    #[test]
    fn output_token_cap_ignores_input_tokens() {
        let l = Limits::default().with_output_tokens(100);
        assert_eq!(l.exceeded(0, &usage(10_000, 50)), None);
        assert_eq!(
            l.exceeded(0, &usage(0, 100)),
            Some(StopReason::LimitOutputTokens)
        );
    }

    #[test]
    fn total_token_cap_counts_input_and_output() {
        let l = Limits::default().with_total_tokens(100);
        assert_eq!(l.exceeded(0, &usage(60, 30)), None);
        assert_eq!(
            l.exceeded(0, &usage(60, 40)),
            Some(StopReason::LimitTotalTokens)
        );
    }

    #[test]
    fn priority_is_turns_then_total_then_output() {
        // All three trip at once; the reported reason must be deterministic
        // and match upstream so the same run explains itself identically.
        let l = Limits {
            turns: Some(1),
            output_tokens: Some(1),
            total_tokens: Some(1),
        };
        assert_eq!(l.exceeded(5, &usage(50, 50)), Some(StopReason::LimitTurns));

        let l = Limits {
            turns: None,
            output_tokens: Some(1),
            total_tokens: Some(1),
        };
        assert_eq!(
            l.exceeded(5, &usage(50, 50)),
            Some(StopReason::LimitTotalTokens)
        );
    }

    #[test]
    fn unreported_usage_counts_as_zero() {
        // A provider that reports nothing must not be treated as having blown
        // the budget.
        let l = Limits::default().with_total_tokens(10);
        assert_eq!(l.exceeded(0, &Usage::default()), None);
    }
}
