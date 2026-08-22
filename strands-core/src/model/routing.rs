//! Model routing — choosing which model serves a call, and what to do when it
//! fails.
//!
//! Implemented as [`Middleware`] over the model stage rather than as a wrapper
//! `Model`, because routing needs the *request* to decide: message count, tool
//! specs, which cycle this is. A `Model` impl only sees its own arguments and
//! cannot short-circuit or retry the surrounding call.
//!
//! Ported from upstream `models/routing/`.

use std::sync::Arc;

use futures::future::BoxFuture;
use tracing::{debug, warn};

use crate::error::StrandsError;
use crate::middleware::stages::InvokeModelResult;
use crate::middleware::{InvokeModelContext, Middleware, Next};

use super::Model;

/// Decides which model should serve a given call.
pub trait RoutingStrategy: Send + Sync {
    /// Pick a model for this request, or `None` to keep the current one.
    fn select(&self, ctx: &InvokeModelContext) -> Option<Arc<dyn Model>>;

    fn name(&self) -> &str {
        "routing"
    }
}

/// Routes every call to one model.
pub struct StaticStrategy {
    model: Arc<dyn Model>,
}

impl StaticStrategy {
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self { model }
    }
}

impl RoutingStrategy for StaticStrategy {
    fn select(&self, _ctx: &InvokeModelContext) -> Option<Arc<dyn Model>> {
        Some(self.model.clone())
    }

    fn name(&self) -> &str {
        "static"
    }
}

/// Routes on a caller-supplied predicate over the request.
pub struct PredicateStrategy<F> {
    predicate: F,
    model: Arc<dyn Model>,
}

impl<F> PredicateStrategy<F>
where
    F: Fn(&InvokeModelContext) -> bool + Send + Sync,
{
    pub fn new(model: Arc<dyn Model>, predicate: F) -> Self {
        Self { predicate, model }
    }
}

impl<F> RoutingStrategy for PredicateStrategy<F>
where
    F: Fn(&InvokeModelContext) -> bool + Send + Sync,
{
    fn select(&self, ctx: &InvokeModelContext) -> Option<Arc<dyn Model>> {
        (self.predicate)(ctx).then(|| self.model.clone())
    }

    fn name(&self) -> &str {
        "predicate"
    }
}

/// Middleware that applies a [`RoutingStrategy`] to each call.
pub struct ModelRouter {
    strategy: Box<dyn RoutingStrategy>,
}

impl ModelRouter {
    pub fn new(strategy: impl RoutingStrategy + 'static) -> Self {
        Self {
            strategy: Box::new(strategy),
        }
    }

    /// Always route to `model`.
    pub fn to(model: Arc<dyn Model>) -> Self {
        Self::new(StaticStrategy::new(model))
    }
}

impl Middleware<InvokeModelContext, InvokeModelResult> for ModelRouter {
    fn handle<'a>(
        &'a self,
        mut ctx: InvokeModelContext,
        next: Next<'a, InvokeModelContext, InvokeModelResult>,
    ) -> BoxFuture<'a, InvokeModelResult> {
        if let Some(model) = self.strategy.select(&ctx) {
            debug!(
                strategy = self.strategy.name(),
                model_id = model.model_id().unwrap_or("<unknown>"),
                cycle = ctx.cycle,
                "Routing model call"
            );
            ctx.model = model;
        }
        Box::pin(async move { next.run(ctx).await })
    }
}

/// Middleware that retries a failed call against fallback models in order.
///
/// Only failures worth retrying elsewhere are followed up: a quota or auth
/// error against one provider says nothing about the next, but a cancellation
/// is the caller's decision and must not be worked around.
pub struct FallbackStrategy {
    fallbacks: Vec<Arc<dyn Model>>,
}

impl FallbackStrategy {
    pub fn new(fallbacks: Vec<Arc<dyn Model>>) -> Self {
        Self { fallbacks }
    }

    /// Whether a failure should be retried against the next model.
    ///
    /// Cancellation is deliberate and must never be retried. Everything else —
    /// including quota and auth failures, which are permanent for *that*
    /// provider but say nothing about another — is worth trying elsewhere.
    fn should_fall_back(error: &StrandsError) -> bool {
        !matches!(error, StrandsError::Cancelled)
    }
}

impl Middleware<InvokeModelContext, InvokeModelResult> for FallbackStrategy {
    fn handle<'a>(
        &'a self,
        ctx: InvokeModelContext,
        next: Next<'a, InvokeModelContext, InvokeModelResult>,
    ) -> BoxFuture<'a, InvokeModelResult> {
        Box::pin(async move {
            // The chain can only be run once, so the primary attempt goes
            // through `next` and the fallbacks call their models directly.
            let messages = ctx.messages.clone();
            let system_prompt = ctx.system_prompt.clone();
            let tool_specs = ctx.tool_specs.clone();

            let primary = next.run(ctx).await;
            let Err(error) = primary else {
                return primary;
            };

            if !Self::should_fall_back(&error) {
                return Err(error);
            }

            warn!(error = %error, "Primary model failed; trying fallbacks");
            let mut last_error = error;

            for model in &self.fallbacks {
                debug!(
                    model_id = model.model_id().unwrap_or("<unknown>"),
                    "Attempting fallback model"
                );
                match crate::model::routing::direct_call(
                    model.as_ref(),
                    &messages,
                    system_prompt.as_ref(),
                    &tool_specs,
                )
                .await
                {
                    Ok(outcome) => return Ok(outcome),
                    Err(e) => {
                        warn!(error = %e, "Fallback model failed");
                        if !Self::should_fall_back(&e) {
                            return Err(e);
                        }
                        last_error = e;
                    }
                }
            }

            Err(last_error)
        })
    }
}

/// Invoke a model and collect its stream into an outcome.
///
/// Used by the fallback path, which cannot re-enter the middleware chain.
pub(crate) async fn direct_call(
    model: &dyn Model,
    messages: &[crate::types::message::Message],
    system_prompt: Option<&crate::types::content::SystemPrompt>,
    tool_specs: &[crate::types::tools::ToolSpec],
) -> InvokeModelResult {
    use crate::types::streaming::{StopReason, StreamEvent};
    use futures::StreamExt;

    let mut stream = model.stream(messages, system_prompt, tool_specs).await?;
    let mut content = Vec::new();
    let mut text = String::new();
    let mut stop_reason = StopReason::EndTurn;
    let mut usage = crate::types::streaming::Usage::default();
    let mut metrics = crate::types::streaming::Metrics::default();

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::ContentBlockDelta {
                delta: crate::types::streaming::DeltaContent::TextDelta(t),
                ..
            } => text.push_str(&t),
            StreamEvent::MessageStop { stop_reason: sr } => stop_reason = sr,
            StreamEvent::Metadata {
                usage: u,
                metrics: m,
            } => {
                usage = u;
                metrics = m;
            }
            _ => {}
        }
    }

    if !text.is_empty() {
        content.push(crate::types::content::ContentBlock::Text { text });
    }

    Ok(crate::middleware::ModelCallOutcome {
        content,
        stop_reason,
        usage,
        metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::MiddlewareChain;
    use crate::middleware::Terminal;
    use crate::model::ModelStream;
    use crate::types::content::{ContentBlock, SystemPrompt};
    use crate::types::message::Message;
    use crate::types::streaming::{ContentBlockType, DeltaContent, StopReason, StreamEvent};
    use crate::types::tools::ToolSpec;
    use async_trait::async_trait;
    use futures::stream;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct NamedModel {
        id: &'static str,
        calls: Arc<AtomicUsize>,
        fail_with: Option<fn() -> StrandsError>,
    }

    impl NamedModel {
        fn ok(id: &'static str) -> (Arc<dyn Model>, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Arc::new(NamedModel {
                    id,
                    calls: calls.clone(),
                    fail_with: None,
                }),
                calls,
            )
        }

        fn failing(id: &'static str, e: fn() -> StrandsError) -> (Arc<dyn Model>, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Arc::new(NamedModel {
                    id,
                    calls: calls.clone(),
                    fail_with: Some(e),
                }),
                calls,
            )
        }
    }

    #[async_trait]
    impl Model for NamedModel {
        async fn stream(
            &self,
            _messages: &[Message],
            _system_prompt: Option<&SystemPrompt>,
            _tool_specs: &[ToolSpec],
        ) -> Result<ModelStream, StrandsError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if let Some(make) = self.fail_with {
                return Err(make());
            }
            let events = vec![
                Ok(StreamEvent::ContentBlockStart {
                    index: 0,
                    content_type: ContentBlockType::Text,
                }),
                Ok(StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: DeltaContent::TextDelta(self.id.to_string()),
                }),
                Ok(StreamEvent::ContentBlockStop { index: 0 }),
                Ok(StreamEvent::MessageStop {
                    stop_reason: StopReason::EndTurn,
                }),
            ];
            Ok(Box::pin(stream::iter(events)))
        }

        fn model_id(&self) -> Option<&str> {
            Some(self.id)
        }
    }

    /// Terminal that calls whatever model the context ended up with.
    struct CallCtxModel;

    impl Terminal<InvokeModelContext, InvokeModelResult> for CallCtxModel {
        fn call<'a>(&'a self, ctx: InvokeModelContext) -> BoxFuture<'a, InvokeModelResult> {
            Box::pin(async move {
                direct_call(
                    ctx.model.as_ref(),
                    &ctx.messages,
                    ctx.system_prompt.as_ref(),
                    &ctx.tool_specs,
                )
                .await
            })
        }
    }

    fn context(model: Arc<dyn Model>) -> InvokeModelContext {
        InvokeModelContext {
            messages: vec![Message::user("hi")],
            system_prompt: None,
            tool_specs: Vec::new(),
            model,
            cycle: 0,
        }
    }

    fn text_of(outcome: &crate::middleware::ModelCallOutcome) -> String {
        outcome
            .content
            .iter()
            .filter_map(ContentBlock::as_text)
            .collect()
    }

    #[tokio::test]
    async fn router_replaces_the_model() {
        let (primary, primary_calls) = NamedModel::ok("primary");
        let (routed, routed_calls) = NamedModel::ok("routed");

        let terminal = CallCtxModel;
        let mut chain = MiddlewareChain::new();
        chain.push(ModelRouter::to(routed));

        let result = chain.run(context(primary), &terminal).await.unwrap();

        assert_eq!(text_of(&result), "routed");
        assert_eq!(routed_calls.load(Ordering::SeqCst), 1);
        assert_eq!(primary_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_strategy_returning_none_keeps_the_current_model() {
        let (primary, primary_calls) = NamedModel::ok("primary");
        let (other, other_calls) = NamedModel::ok("other");

        let terminal = CallCtxModel;
        let mut chain = MiddlewareChain::new();
        chain.push(ModelRouter::new(PredicateStrategy::new(other, |_| false)));

        let result = chain.run(context(primary), &terminal).await.unwrap();

        assert_eq!(text_of(&result), "primary");
        assert_eq!(primary_calls.load(Ordering::SeqCst), 1);
        assert_eq!(other_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn predicate_routes_on_the_request() {
        let (primary, _) = NamedModel::ok("primary");
        let (big, big_calls) = NamedModel::ok("big");

        let terminal = CallCtxModel;
        let mut chain = MiddlewareChain::new();
        chain.push(ModelRouter::new(PredicateStrategy::new(big, |ctx| {
            !ctx.messages.is_empty()
        })));

        let result = chain.run(context(primary), &terminal).await.unwrap();
        assert_eq!(text_of(&result), "big");
        assert_eq!(big_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn fallback_takes_over_when_the_primary_fails() {
        let (primary, primary_calls) =
            NamedModel::failing("primary", || StrandsError::Model("boom".into()));
        let (backup, backup_calls) = NamedModel::ok("backup");

        let terminal = CallCtxModel;
        let mut chain = MiddlewareChain::new();
        chain.push(FallbackStrategy::new(vec![backup]));

        let result = chain.run(context(primary), &terminal).await.unwrap();

        assert_eq!(text_of(&result), "backup");
        assert_eq!(primary_calls.load(Ordering::SeqCst), 1);
        assert_eq!(backup_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn fallback_tries_each_model_in_order() {
        let (primary, _) = NamedModel::failing("primary", || StrandsError::Model("a".into()));
        let (first, first_calls) =
            NamedModel::failing("first", || StrandsError::Model("b".into()));
        let (second, second_calls) = NamedModel::ok("second");

        let terminal = CallCtxModel;
        let mut chain = MiddlewareChain::new();
        chain.push(FallbackStrategy::new(vec![first, second]));

        let result = chain.run(context(primary), &terminal).await.unwrap();

        assert_eq!(text_of(&result), "second");
        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn quota_failures_do_fall_back() {
        // A quota error is permanent for that provider but says nothing about
        // another, so it is exactly the case fallback exists for.
        let (primary, _) =
            NamedModel::failing("primary", || StrandsError::Quota("exhausted".into()));
        let (backup, backup_calls) = NamedModel::ok("backup");

        let terminal = CallCtxModel;
        let mut chain = MiddlewareChain::new();
        chain.push(FallbackStrategy::new(vec![backup]));

        let result = chain.run(context(primary), &terminal).await.unwrap();
        assert_eq!(text_of(&result), "backup");
        assert_eq!(backup_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cancellation_is_never_worked_around() {
        // Cancellation is the caller's decision; falling back would override it.
        let (primary, _) = NamedModel::failing("primary", || StrandsError::Cancelled);
        let (backup, backup_calls) = NamedModel::ok("backup");

        let terminal = CallCtxModel;
        let mut chain = MiddlewareChain::new();
        chain.push(FallbackStrategy::new(vec![backup]));

        let result = chain.run(context(primary), &terminal).await;

        assert!(matches!(result, Err(StrandsError::Cancelled)));
        assert_eq!(backup_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn the_last_error_surfaces_when_everything_fails() {
        let (primary, _) = NamedModel::failing("primary", || StrandsError::Model("first".into()));
        let (backup, _) = NamedModel::failing("backup", || StrandsError::Model("last".into()));

        let terminal = CallCtxModel;
        let mut chain = MiddlewareChain::new();
        chain.push(FallbackStrategy::new(vec![backup]));

        let err = chain.run(context(primary), &terminal).await.unwrap_err();
        assert!(err.to_string().contains("last"), "got {err}");
    }

    #[tokio::test]
    async fn a_successful_primary_never_touches_the_fallbacks() {
        let (primary, _) = NamedModel::ok("primary");
        let (backup, backup_calls) = NamedModel::ok("backup");

        let terminal = CallCtxModel;
        let mut chain = MiddlewareChain::new();
        chain.push(FallbackStrategy::new(vec![backup]));

        let result = chain.run(context(primary), &terminal).await.unwrap();
        assert_eq!(text_of(&result), "primary");
        assert_eq!(backup_calls.load(Ordering::SeqCst), 0);
    }
}
