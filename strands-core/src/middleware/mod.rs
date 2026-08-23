//! Middleware — interception points around the agent's core operations.
//!
//! A middleware wraps a stage: it sees the context on the way in, decides
//! whether to run the rest of the chain, and sees the result on the way out.
//! That is enough to rate-limit, cache, retry, mock, validate, add telemetry,
//! or swap the model per call, without any of those concerns living in the
//! event loop.
//!
//! # Difference from upstream
//!
//! Upstream splits interception into three phases (Input, Wrap, Output) over
//! async generators, because Python cannot express "call the rest of the chain"
//! ergonomically in one callback. Rust can: [`Middleware::handle`] receives a
//! [`Next`] and awaits it wherever it likes. Input-only, output-only and
//! short-circuiting middleware are all special cases of that, so the three
//! phases collapse into one method with no loss of power.
//!
//! Ported from upstream `_middleware/`.

/// The built-in interception points.
pub mod stages;

use std::sync::Arc;

use futures::future::BoxFuture;

pub use stages::{ExecuteToolContext, InvokeModelContext, ModelCallOutcome};

/// The remainder of a middleware chain, ending in the operation itself.
///
/// Await [`run`](Next::run) to continue. Not awaiting it short-circuits the
/// stage — which is exactly how a cache hit or a rejected call is expressed.
pub struct Next<'a, C, R>
where
    C: Send + 'static,
    R: Send + 'static,
{
    remaining: &'a [Arc<dyn Middleware<C, R>>],
    terminal: &'a (dyn Terminal<C, R> + 'a),
}

impl<'a, C, R> Next<'a, C, R>
where
    C: Send + 'static,
    R: Send + 'static,
{
    /// Run the rest of the chain with `ctx`.
    pub fn run(self, ctx: C) -> BoxFuture<'a, R> {
        match self.remaining.split_first() {
            Some((current, rest)) => {
                let next = Next {
                    remaining: rest,
                    terminal: self.terminal,
                };
                current.handle(ctx, next)
            }
            None => self.terminal.call(ctx),
        }
    }
}

/// The operation a chain wraps — the thing that actually runs when every
/// middleware has called `next`.
///
/// A trait rather than a closure bound so the returned future borrows from
/// `&self`, which keeps its lifetime tied to the chain rather than leaking a
/// free lifetime parameter out to every caller.
pub trait Terminal<C, R>: Send + Sync
where
    C: Send + 'static,
    R: Send + 'static,
{
    /// Run the operation.
    fn call<'a>(&'a self, ctx: C) -> BoxFuture<'a, R>;
}

/// An interception point around a stage.
pub trait Middleware<C, R>: Send + Sync
where
    C: Send + 'static,
    R: Send + 'static,
{
    /// Handle the stage.
    ///
    /// Call `next.run(ctx)` to continue, optionally transforming `ctx` first or
    /// the result afterwards. Returning without calling it short-circuits the
    /// operation.
    fn handle<'a>(&'a self, ctx: C, next: Next<'a, C, R>) -> BoxFuture<'a, R>;
}

/// An ordered chain of middleware for one stage.
pub struct MiddlewareChain<C, R>
where
    C: Send + 'static,
    R: Send + 'static,
{
    middleware: Vec<Arc<dyn Middleware<C, R>>>,
}

impl<C, R> Default for MiddlewareChain<C, R>
where
    C: Send + 'static,
    R: Send + 'static,
{
    fn default() -> Self {
        Self {
            middleware: Vec::new(),
        }
    }
}

impl<C, R> MiddlewareChain<C, R>
where
    C: Send + 'static,
    R: Send + 'static,
{
    /// Create with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a middleware.
    ///
    /// Registration order is execution order on the way in, and the reverse on
    /// the way out — the first registered is outermost, so it sees the original
    /// context and the final result.
    pub fn push(&mut self, middleware: impl Middleware<C, R> + 'static) {
        self.middleware.push(Arc::new(middleware));
    }

    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.middleware.is_empty()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.middleware.len()
    }

    /// Run the chain, ending in `terminal`.
    pub async fn run<'a>(&'a self, ctx: C, terminal: &'a (dyn Terminal<C, R> + 'a)) -> R {
        Next {
            remaining: &self.middleware,
            terminal,
        }
        .run(ctx)
        .await
    }
}

impl<C, R> std::fmt::Debug for MiddlewareChain<C, R>
where
    C: Send + 'static,
    R: Send + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MiddlewareChain")
            .field("len", &self.middleware.len())
            .finish()
    }
}

/// Adapt a closure into a [`Middleware`].
pub struct FnMiddleware<F>(pub F);

impl<C, R, F> Middleware<C, R> for FnMiddleware<F>
where
    F: for<'a> Fn(C, Next<'a, C, R>) -> BoxFuture<'a, R> + Send + Sync,
    C: Send + 'static,
    R: Send + 'static,
{
    fn handle<'a>(&'a self, ctx: C, next: Next<'a, C, R>) -> BoxFuture<'a, R> {
        (self.0)(ctx, next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    type Log = Arc<Mutex<Vec<&'static str>>>;

    /// Records entry and exit, so ordering is observable in both directions.
    struct Recorder {
        name: &'static str,
        log: Log,
    }

    impl Middleware<i32, i32> for Recorder {
        fn handle<'a>(&'a self, ctx: i32, next: Next<'a, i32, i32>) -> BoxFuture<'a, i32> {
            let log = self.log.clone();
            let name = self.name;
            Box::pin(async move {
                log.lock().unwrap().push(name);
                let result = next.run(ctx + 1).await;
                log.lock().unwrap().push(name);
                result
            })
        }
    }

    /// Never calls `next`, so the operation itself never runs.
    struct ShortCircuit(i32);

    impl Middleware<i32, i32> for ShortCircuit {
        fn handle<'a>(&'a self, _ctx: i32, _next: Next<'a, i32, i32>) -> BoxFuture<'a, i32> {
            let value = self.0;
            Box::pin(async move { value })
        }
    }

    /// Transforms the result on the way out.
    struct Doubler;

    impl Middleware<i32, i32> for Doubler {
        fn handle<'a>(&'a self, ctx: i32, next: Next<'a, i32, i32>) -> BoxFuture<'a, i32> {
            Box::pin(async move { next.run(ctx).await * 2 })
        }
    }

    /// The operation at the end of the chain: returns its input unchanged.
    struct Identity;

    impl Terminal<i32, i32> for Identity {
        fn call<'a>(&'a self, ctx: i32) -> BoxFuture<'a, i32> {
            Box::pin(async move { ctx })
        }
    }

    fn terminal() -> Identity {
        Identity
    }

    /// A plain function used as middleware.
    fn times_ten<'a>(ctx: i32, next: Next<'a, i32, i32>) -> BoxFuture<'a, i32> {
        Box::pin(async move { next.run(ctx * 10).await })
    }

    #[tokio::test]
    async fn an_empty_chain_runs_the_operation_directly() {
        let t = terminal();
        let chain: MiddlewareChain<i32, i32> = MiddlewareChain::new();
        assert_eq!(chain.run(5, &t).await, 5);
    }

    #[tokio::test]
    async fn middleware_can_transform_the_context() {
        let t = terminal();
        let mut chain = MiddlewareChain::new();
        chain.push(Recorder {
            name: "a",
            log: Arc::new(Mutex::new(Vec::new())),
        });
        assert_eq!(
            chain.run(5, &t).await,
            6,
            "the +1 should reach the terminal"
        );
    }

    #[tokio::test]
    async fn middleware_can_transform_the_result() {
        let t = terminal();
        let mut chain = MiddlewareChain::new();
        chain.push(Doubler);
        assert_eq!(chain.run(5, &t).await, 10);
    }

    #[tokio::test]
    async fn first_registered_is_outermost() {
        let t = terminal();
        // The way in follows registration order; the way out reverses it. This
        // is what lets the first middleware see the original context and the
        // final result.
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let mut chain = MiddlewareChain::new();
        chain.push(Recorder {
            name: "outer",
            log: log.clone(),
        });
        chain.push(Recorder {
            name: "inner",
            log: log.clone(),
        });
        chain.run(0, &t).await;

        assert_eq!(
            *log.lock().unwrap(),
            vec!["outer", "inner", "inner", "outer"]
        );
    }

    #[tokio::test]
    async fn context_transformations_compose_down_the_chain() {
        let t = terminal();
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let mut chain = MiddlewareChain::new();
        for name in ["a", "b", "c"] {
            chain.push(Recorder {
                name,
                log: log.clone(),
            });
        }
        assert_eq!(chain.run(0, &t).await, 3, "each layer adds one");
    }

    #[tokio::test]
    async fn not_calling_next_short_circuits_the_operation() {
        let t = terminal();
        // This is how a cache hit or a rejected call is expressed: the terminal
        // never runs.
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let mut chain = MiddlewareChain::new();
        chain.push(Recorder {
            name: "outer",
            log: log.clone(),
        });
        chain.push(ShortCircuit(99));
        chain.push(Recorder {
            name: "never",
            log: log.clone(),
        });
        assert_eq!(chain.run(0, &t).await, 99);
        assert_eq!(
            *log.lock().unwrap(),
            vec!["outer", "outer"],
            "middleware after the short circuit must not run"
        );
    }

    #[tokio::test]
    async fn a_plain_function_can_serve_as_middleware() {
        let t = terminal();
        let mut chain = MiddlewareChain::new();
        chain.push(FnMiddleware(
            times_ten as for<'a> fn(i32, Next<'a, i32, i32>) -> BoxFuture<'a, i32>,
        ));
        assert_eq!(chain.run(2, &t).await, 20);
    }
}
