use super::events::HookEvent;

/// Named execution priorities for hooks.
///
/// Lower values run first. Hooks sharing a priority keep registration order,
/// so adding an ordered hook never reshuffles unordered ones relative to each
/// other.
pub mod order {
    /// Runs before everything else — reserved for SDK internals.
    pub const SDK_FIRST: i32 = -100;
    /// Intervention handlers that shape outbound content.
    pub const INTERVENTION_OUTPUT: i32 = -90;
    /// Default priority for user hooks.
    pub const DEFAULT: i32 = 0;
    /// Model routing, which must see a fully prepared request.
    pub const MODEL_ROUTING: i32 = 50;
    /// Intervention handlers that gate inbound content.
    pub const INTERVENTION_INPUT: i32 = 90;
    /// Runs after everything else — reserved for SDK internals.
    pub const SDK_LAST: i32 = 100;
}

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

struct Entry {
    hook: Box<dyn Hook>,
    order: i32,
}

/// Registry of hooks that are dispatched during agent execution.
///
/// Entries are kept in dispatch order at registration time, so dispatch — the
/// hot path — needs only a shared borrow.
#[derive(Default)]
pub struct HookRegistry {
    entries: Vec<Entry>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a hook at the default priority.
    pub fn register(&mut self, hook: impl Hook + 'static) {
        self.register_with_order(hook, order::DEFAULT);
    }

    /// Register a hook at an explicit priority.
    ///
    /// See [`order`] for the named constants.
    pub fn register_with_order(&mut self, hook: impl Hook + 'static, order: i32) {
        self.entries.push(Entry {
            hook: Box::new(hook),
            order,
        });
        // A stable sort keeps same-priority hooks in registration order, so a
        // newly added hook lands last among its equals.
        self.entries.sort_by_key(|e| e.order);
    }

    /// Dispatch an event to all registered hooks.
    ///
    /// Hooks receive a mutable reference and can modify writable fields.
    /// Events that report [`HookEvent::is_teardown`] dispatch in reverse, so
    /// cleanup unwinds in the opposite order to setup.
    pub fn dispatch(&self, event: &mut HookEvent) {
        if event.is_teardown() {
            for entry in self.entries.iter().rev() {
                entry.hook.on_event(event);
            }
        } else {
            for entry in self.entries.iter() {
                entry.hook.on_event(event);
            }
        }
    }

    /// Number of registered hooks.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl std::fmt::Debug for HookRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookRegistry")
            .field("hook_count", &self.entries.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn recorder(log: Arc<Mutex<Vec<&'static str>>>, name: &'static str) -> impl Hook {
        move |_: &mut HookEvent| log.lock().unwrap().push(name)
    }

    #[test]
    fn same_priority_hooks_keep_registration_order() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut reg = HookRegistry::new();
        reg.register(recorder(log.clone(), "first"));
        reg.register(recorder(log.clone(), "second"));
        reg.register(recorder(log.clone(), "third"));

        reg.dispatch(&mut HookEvent::AgentInitialized);
        assert_eq!(*log.lock().unwrap(), vec!["first", "second", "third"]);
    }

    #[test]
    fn lower_order_runs_first_regardless_of_registration() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut reg = HookRegistry::new();
        reg.register(recorder(log.clone(), "default"));
        reg.register_with_order(recorder(log.clone(), "late"), order::SDK_LAST);
        reg.register_with_order(recorder(log.clone(), "early"), order::SDK_FIRST);

        reg.dispatch(&mut HookEvent::AgentInitialized);
        assert_eq!(*log.lock().unwrap(), vec!["early", "default", "late"]);
    }

    #[test]
    fn ordering_is_stable_within_a_priority() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut reg = HookRegistry::new();
        reg.register_with_order(recorder(log.clone(), "a"), 10);
        reg.register_with_order(recorder(log.clone(), "b"), 10);
        reg.register_with_order(recorder(log.clone(), "c"), 5);

        reg.dispatch(&mut HookEvent::AgentInitialized);
        assert_eq!(*log.lock().unwrap(), vec!["c", "a", "b"]);
    }

    #[test]
    fn repeated_dispatch_does_not_reshuffle() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut reg = HookRegistry::new();
        reg.register(recorder(log.clone(), "a"));
        reg.register(recorder(log.clone(), "b"));

        reg.dispatch(&mut HookEvent::AgentInitialized);
        reg.dispatch(&mut HookEvent::AgentInitialized);
        assert_eq!(*log.lock().unwrap(), vec!["a", "b", "a", "b"]);
    }

    #[test]
    fn teardown_events_dispatch_in_reverse() {
        // Cleanup should unwind in the opposite order to setup, so a hook
        // registered later tears down before the one it was layered on.
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut reg = HookRegistry::new();
        reg.register(recorder(log.clone(), "outer"));
        reg.register(recorder(log.clone(), "inner"));

        let mut event = HookEvent::AfterTools(super::super::events::AfterToolsEvent {
            results: Vec::new(),
            end_turn: false,
        });
        reg.dispatch(&mut event);
        assert_eq!(*log.lock().unwrap(), vec!["inner", "outer"]);
    }
}
