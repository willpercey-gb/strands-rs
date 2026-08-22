use async_trait::async_trait;

use crate::error::StrandsError;
use crate::types::content::SystemPrompt;
use crate::types::message::Message;

/// Default utilization at which proactive compression fires.
pub const DEFAULT_COMPRESSION_THRESHOLD: f64 = 0.7;

/// When proactive compression should run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProactiveCompression {
    /// Fraction of the context window that triggers compression, in `(0, 1]`.
    pub compression_threshold: f64,
}

impl Default for ProactiveCompression {
    fn default() -> Self {
        Self {
            compression_threshold: DEFAULT_COMPRESSION_THRESHOLD,
        }
    }
}

impl ProactiveCompression {
    /// Build a config, clamping the threshold into `(0, 1]`.
    pub fn new(compression_threshold: f64) -> Self {
        Self {
            compression_threshold: compression_threshold.clamp(f64::EPSILON, 1.0),
        }
    }
}

/// Why [`ConversationManager::reduce_context`] was called, and what the caller
/// knows about the current context.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReduceContext<'a> {
    /// The system prompt in force, which counts toward the context window.
    pub system_prompt: Option<&'a SystemPrompt>,
    /// Fraction of the context window currently in use, when the model could
    /// report a limit.
    ///
    /// `None` means *unknown*, not *low* — a manager must not treat it as
    /// headroom, or it will skip compressing a conversation that is actually
    /// about to overflow.
    pub utilization: Option<f64>,
    /// Set when recovering from a provider context-overflow error.
    ///
    /// Reduction is then mandatory: returning without shrinking the history
    /// means the retried call fails the same way. Managers may take more
    /// aggressive measures here (e.g. truncating large tool results) than
    /// during routine management.
    pub overflow: bool,
}

impl<'a> ReduceContext<'a> {
    /// Routine management, with no overflow and nothing known about usage.
    pub fn routine(system_prompt: Option<&'a SystemPrompt>) -> Self {
        Self {
            system_prompt,
            utilization: None,
            overflow: false,
        }
    }

    /// Reactive recovery from a provider context-overflow error.
    pub fn overflow(system_prompt: Option<&'a SystemPrompt>) -> Self {
        Self {
            system_prompt,
            utilization: None,
            overflow: true,
        }
    }

    pub fn with_utilization(mut self, utilization: Option<f64>) -> Self {
        self.utilization = utilization;
        self
    }

    /// Whether proactive compression should fire under `config`.
    ///
    /// False when utilization is unknown — proactive compression is an
    /// optimisation, and firing it on a guess would discard context for no
    /// measured reason. Genuine overflow is handled by the `overflow` path.
    pub fn should_compress(&self, config: &ProactiveCompression) -> bool {
        self.utilization
            .is_some_and(|u| u >= config.compression_threshold)
    }
}

/// Strategy for managing conversation context window limits.
///
/// Called before each model invocation to ensure the message history fits
/// within the model's context window.
#[async_trait]
pub trait ConversationManager: Send + Sync {
    /// Reduce context if the message list exceeds limits.
    ///
    /// Mutates messages in place. When `ctx.overflow` is set this **must**
    /// shrink the history or the retried model call will fail identically.
    async fn reduce_context(
        &self,
        messages: &mut Vec<Message>,
        ctx: ReduceContext<'_>,
    ) -> Result<(), StrandsError>;

    /// Proactive compression settings, if this manager supports them.
    ///
    /// `None` disables proactive compression; only reactive overflow recovery
    /// runs.
    fn proactive_compression(&self) -> Option<ProactiveCompression> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compression_fires_at_or_above_the_threshold() {
        let cfg = ProactiveCompression::default();
        assert!(ReduceContext::default()
            .with_utilization(Some(0.7))
            .should_compress(&cfg));
        assert!(ReduceContext::default()
            .with_utilization(Some(0.95))
            .should_compress(&cfg));
        assert!(!ReduceContext::default()
            .with_utilization(Some(0.69))
            .should_compress(&cfg));
    }

    #[test]
    fn unknown_utilization_does_not_trigger_compression() {
        // Unknown must not be read as "full" — compressing on a guess throws
        // away context nobody measured a need to lose.
        let cfg = ProactiveCompression::default();
        assert!(!ReduceContext::default().should_compress(&cfg));
    }

    #[test]
    fn threshold_is_clamped_into_range() {
        assert_eq!(ProactiveCompression::new(5.0).compression_threshold, 1.0);
        assert!(ProactiveCompression::new(0.0).compression_threshold > 0.0);
        assert!(ProactiveCompression::new(-1.0).compression_threshold > 0.0);
    }

    #[test]
    fn constructors_set_the_right_mode() {
        assert!(!ReduceContext::routine(None).overflow);
        assert!(ReduceContext::overflow(None).overflow);
    }
}
