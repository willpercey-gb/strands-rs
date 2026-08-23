# Middleware

Middleware wraps the agent's model calls. It sees the request on the way in,
decides whether to run the rest of the chain, and sees the result on the way
out — enough to cache, rate-limit, retry, mock, validate, add telemetry, or swap
the model per call, without any of that living in the event loop.

## The shape

```rust,ignore
pub trait Middleware<C, R>: Send + Sync {
    fn handle<'a>(&'a self, ctx: C, next: Next<'a, C, R>) -> BoxFuture<'a, R>;
}
```

Upstream's Python SDK splits this into three phases (Input, Wrap, Output),
because Python cannot express "call the rest of the chain" ergonomically in a
single callback. Rust can: `handle` receives a `Next` and awaits it wherever it
likes. Input-only, output-only and short-circuiting middleware are all special
cases of the one method.

## Registering

```rust,ignore
let agent = Agent::builder()
    .model(model)
    .model_middleware(RateLimiter::new(10))
    .model_middleware(ResponseCache::new(store))
    .build()?;
```

Registration order is outermost-first. The first registered sees the original
request and the final result; the last sits closest to the model.

## Transforming the request

```rust,ignore
impl Middleware<InvokeModelContext, InvokeModelResult> for InjectPreamble {
    fn handle<'a>(
        &'a self,
        mut ctx: InvokeModelContext,
        next: Next<'a, InvokeModelContext, InvokeModelResult>,
    ) -> BoxFuture<'a, InvokeModelResult> {
        ctx.system_prompt = Some(SystemPrompt::from("Answer tersely."));
        Box::pin(async move { next.run(ctx).await })
    }
}
```

## Short-circuiting

Not calling `next` skips the model entirely. This is how a cache hit is
expressed:

```rust,ignore
fn handle<'a>(
    &'a self,
    ctx: InvokeModelContext,
    next: Next<'a, InvokeModelContext, InvokeModelResult>,
) -> BoxFuture<'a, InvokeModelResult> {
    if let Some(hit) = self.cache.get(&ctx.messages) {
        return Box::pin(async move { Ok(hit) });   // the model is never called
    }
    Box::pin(async move { next.run(ctx).await })
}
```

## Swapping the model

`ctx.model` is an `Arc<dyn Model>` and may be replaced per call. This is the
mechanism [model routing](14-model-routing.md) is built on.

```rust,ignore
ctx.model = self.cheap_model.clone();
Box::pin(async move { next.run(ctx).await })
```

## What the context carries

| Field | Notes |
|-------|-------|
| `messages` | Owned copy — rewrite freely without touching agent history |
| `system_prompt` | `Option<SystemPrompt>`; may be text or blocks with cache points |
| `tool_specs` | Owned copy of the advertised tools |
| `model` | Shared; replace to route this call elsewhere |
| `cycle` | Which iteration of the agent loop this is |
