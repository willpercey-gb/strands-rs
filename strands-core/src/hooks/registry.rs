use super::events::HookEvent;

/// Trait for hook callbacks that react to agent lifecycle events.
///
/// Hooks receive mutable references to events, allowing them to
/// modify writable fields (e.g., cancel tool calls, retry model calls).
pub trait Hook: Send + Sync {
    fn on_event(&self, event: &mut HookEvent);
}

/// Blanket impl so closures can be used as hooks.
impl<F: Fn(&mut HookEvent) + Send + Sync> Hook for F {
    fn on_event(&self, event: &mut HookEvent) {
        self(event);
    }
}

/// Registry of hooks that are dispatched during agent execution.
#[derive(Default)]
pub struct HookRegistry {
    hooks: Vec<Box<dyn Hook>>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, hook: impl Hook + 'static) {
        self.hooks.push(Box::new(hook));
    }

    /// Dispatch an event to all registered hooks.
    /// Hooks receive a mutable reference and can modify writable fields.
    pub fn dispatch(&self, event: &mut HookEvent) {
        for hook in &self.hooks {
            hook.on_event(event);
        }
    }
}

impl std::fmt::Debug for HookRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookRegistry")
            .field("hook_count", &self.hooks.len())
            .finish()
    }
}
