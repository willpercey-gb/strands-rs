//! Goal loop — keeping an agent working until an objective is actually met.
//!
//! Left alone, an agent stops when the model decides it is done, which is not
//! the same as the goal being achieved. This plugin adds a judge: after each
//! turn it asks whether the stated goal is satisfied, and if not, feeds the
//! shortfall back as the next prompt.
//!
//! Ported from upstream `vended_plugins/goal/`.

use async_trait::async_trait;

use crate::error::StrandsError;

/// A judge's verdict on whether a goal has been met.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    /// Whether the goal is met.
    pub achieved: bool,
    /// What is still missing. Fed back to the agent when not achieved.
    pub feedback: String,
}

impl Verdict {
    /// The goal is met.
    pub fn achieved() -> Self {
        Self {
            achieved: true,
            feedback: String::new(),
        }
    }

    /// The goal is not met, with what is missing.
    pub fn not_yet(feedback: impl Into<String>) -> Self {
        Self {
            achieved: false,
            feedback: feedback.into(),
        }
    }
}

/// Decides whether a goal has been met.
#[async_trait]
pub trait GoalJudge: Send + Sync {
    /// Decide whether `output` satisfies `goal`.
    async fn judge(&self, goal: &str, output: &str) -> Result<Verdict, StrandsError>;
}

/// Judges by looking for required substrings in the output.
///
/// Deterministic and cheap. Useful when "done" has a concrete marker; a
/// model-backed judge is the answer when it does not.
pub struct ContainsJudge {
    required: Vec<String>,
}

impl ContainsJudge {
    /// Create a new instance.
    pub fn new<I, S>(required: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            required: required.into_iter().map(Into::into).collect(),
        }
    }
}

#[async_trait]
impl GoalJudge for ContainsJudge {
    async fn judge(&self, _goal: &str, output: &str) -> Result<Verdict, StrandsError> {
        let lower = output.to_lowercase();
        let missing: Vec<&str> = self
            .required
            .iter()
            .filter(|needle| !lower.contains(&needle.to_lowercase()))
            .map(String::as_str)
            .collect();

        if missing.is_empty() {
            Ok(Verdict::achieved())
        } else {
            Ok(Verdict::not_yet(format!(
                "still missing: {}",
                missing.join(", ")
            )))
        }
    }
}

/// Drives an agent toward a goal across multiple turns.
pub struct GoalLoop {
    goal: String,
    judge: Box<dyn GoalJudge>,
    max_attempts: usize,
}

/// How a goal loop ended.
#[derive(Debug, Clone, PartialEq)]
pub struct GoalOutcome {
    /// Whether the goal was met before attempts ran out.
    pub achieved: bool,
    /// How many attempts were made.
    pub attempts: usize,
    /// The judge's last verdict.
    pub last_feedback: String,
}

impl GoalLoop {
    /// Create a new instance.
    pub fn new(goal: impl Into<String>, judge: impl GoalJudge + 'static) -> Self {
        Self {
            goal: goal.into(),
            judge: Box::new(judge),
            max_attempts: 5,
        }
    }

    /// Cap how many times the agent is nudged.
    ///
    /// Bounded because a goal the agent cannot reach would otherwise loop
    /// until the token budget runs out, which is an expensive way to discover
    /// the goal was unreachable.
    /// Set the max attempts.
    pub fn with_max_attempts(mut self, attempts: usize) -> Self {
        self.max_attempts = attempts.max(1);
        self
    }

    /// The objective being pursued.
    pub fn goal(&self) -> &str {
        &self.goal
    }

    /// Run `step` until the judge is satisfied or attempts run out.
    ///
    /// `step` receives the prompt for this attempt and returns the agent's
    /// output.
    pub async fn run<F, Fut>(&self, mut step: F) -> Result<GoalOutcome, StrandsError>
    where
        F: FnMut(String) -> Fut,
        Fut: std::future::Future<Output = Result<String, StrandsError>>,
    {
        let mut prompt = self.goal.clone();
        let mut last_feedback = String::new();

        for attempt in 1..=self.max_attempts {
            let output = step(prompt).await?;
            let verdict = self.judge.judge(&self.goal, &output).await?;
            last_feedback = verdict.feedback.clone();

            if verdict.achieved {
                return Ok(GoalOutcome {
                    achieved: true,
                    attempts: attempt,
                    last_feedback,
                });
            }

            // Restate the goal alongside the shortfall: feedback alone loses
            // the objective once the conversation has been trimmed.
            prompt = format!(
                "The goal is not yet met.\n\nGoal: {}\n\nWhat is missing: {}",
                self.goal, verdict.feedback
            );
        }

        Ok(GoalOutcome {
            achieved: false,
            attempts: self.max_attempts,
            last_feedback,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[tokio::test]
    async fn a_met_goal_finishes_on_the_first_attempt() {
        let loop_ = GoalLoop::new("write a report", ContainsJudge::new(["summary"]));

        let outcome = loop_
            .run(|_| async { Ok("here is the summary".to_string()) })
            .await
            .unwrap();

        assert!(outcome.achieved);
        assert_eq!(outcome.attempts, 1);
    }

    #[tokio::test]
    async fn an_unmet_goal_is_retried_with_feedback() {
        let prompts: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let seen = prompts.clone();
        let attempt = Arc::new(Mutex::new(0));

        let loop_ = GoalLoop::new("write a report", ContainsJudge::new(["summary"]));

        let outcome = loop_
            .run(move |prompt| {
                let seen = seen.clone();
                let attempt = attempt.clone();
                async move {
                    seen.lock().unwrap().push(prompt);
                    let mut n = attempt.lock().unwrap();
                    *n += 1;
                    Ok(if *n < 2 {
                        "nothing useful".to_string()
                    } else {
                        "the summary".to_string()
                    })
                }
            })
            .await
            .unwrap();

        assert!(outcome.achieved);
        assert_eq!(outcome.attempts, 2);

        let prompts = prompts.lock().unwrap();
        assert_eq!(prompts[0], "write a report");
        assert!(
            prompts[1].contains("summary"),
            "the retry must say what was missing: {}",
            prompts[1]
        );
        assert!(
            prompts[1].contains("write a report"),
            "the retry must restate the goal, which trimming may have dropped"
        );
    }

    #[tokio::test]
    async fn an_unreachable_goal_stops_at_the_attempt_cap() {
        // Otherwise this burns the whole token budget discovering the goal was
        // unreachable.
        let calls = Arc::new(Mutex::new(0));
        let counted = calls.clone();

        let loop_ =
            GoalLoop::new("impossible", ContainsJudge::new(["never appears"])).with_max_attempts(3);

        let outcome = loop_
            .run(move |_| {
                let counted = counted.clone();
                async move {
                    *counted.lock().unwrap() += 1;
                    Ok("no".to_string())
                }
            })
            .await
            .unwrap();

        assert!(!outcome.achieved);
        assert_eq!(outcome.attempts, 3);
        assert_eq!(*calls.lock().unwrap(), 3);
        assert!(outcome.last_feedback.contains("never appears"));
    }

    #[tokio::test]
    async fn the_attempt_cap_is_at_least_one() {
        let loop_ = GoalLoop::new("g", ContainsJudge::new(["x"])).with_max_attempts(0);
        let outcome = loop_.run(|_| async { Ok("y".to_string()) }).await.unwrap();
        assert_eq!(outcome.attempts, 1);
    }

    #[tokio::test]
    async fn a_failing_step_surfaces_rather_than_counting_as_unmet() {
        let loop_ = GoalLoop::new("g", ContainsJudge::new(["x"]));
        let result = loop_
            .run(|_| async { Err(StrandsError::Other("boom".into())) })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn contains_judge_is_case_insensitive_and_reports_every_gap() {
        let verdict = ContainsJudge::new(["Alpha", "Beta"])
            .judge("g", "alpha only")
            .await
            .unwrap();

        assert!(!verdict.achieved);
        assert!(verdict.feedback.contains("Beta"));
        assert!(!verdict.feedback.contains("Alpha"));
    }
}
