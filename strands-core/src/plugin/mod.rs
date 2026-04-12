use crate::hooks::HookRegistry;
use crate::tool::Tool;

/// A plugin bundles hooks and tools into a reusable unit.
///
/// Plugins are registered on the agent builder and automatically
/// contribute their hooks and tools during agent construction.
///
/// # Example
///
/// ```ignore
/// struct LoggingPlugin;
///
/// impl Plugin for LoggingPlugin {
///     fn name(&self) -> &str { "logging" }
///
///     fn register_hooks(&self, registry: &mut HookRegistry) {
///         registry.register(|event: &mut HookEvent| {
///             tracing::info!(?event, "agent event");
///         });
///     }
/// }
///
/// let agent = Agent::builder()
///     .model(model)
///     .plugin(LoggingPlugin)
///     .build()?;
/// ```
pub trait Plugin: Send + Sync {
    /// Unique name for this plugin.
    fn name(&self) -> &str;

    /// Register hooks with the agent's hook registry.
    fn register_hooks(&self, registry: &mut HookRegistry) {
        let _ = registry;
    }

    /// Return tools provided by this plugin.
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        Vec::new()
    }
}
