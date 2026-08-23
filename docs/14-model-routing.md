# Model Routing & Fallback

Routing picks which model serves a call. Fallback decides what happens when it
fails. Both are [middleware](13-middleware.md) rather than wrapper `Model`
implementations, because the decision depends on the *request* — message count,
tool specs, which cycle this is — and a `Model` impl only sees its own arguments.

## Routing to a fixed model

```rust,ignore
use strands_core::model::ModelRouter;

let agent = Agent::builder()
    .model(default_model)
    .model_middleware(ModelRouter::to(preferred.clone()))
    .build()?;
```

## Routing on the request

```rust,ignore
use strands_core::model::{ModelRouter, PredicateStrategy};

// Long conversations go to the large-context model.
let router = ModelRouter::new(PredicateStrategy::new(
    big_context_model.clone(),
    |ctx| ctx.messages.len() > 50,
));
```

A strategy returning `None` leaves the current model in place, so several
routers can be layered without each having to know about the others.

## Custom strategies

```rust,ignore
impl RoutingStrategy for TimeOfDay {
    fn select(&self, ctx: &InvokeModelContext) -> Option<Arc<dyn Model>> {
        (ctx.cycle == 0).then(|| self.fast.clone())
    }
}
```

## Fallback

```rust,ignore
use strands_core::model::FallbackStrategy;

let agent = Agent::builder()
    .model(primary)
    .model_middleware(FallbackStrategy::new(vec![secondary, tertiary]))
    .build()?;
```

Fallbacks are tried in order. When everything fails, the **last** error
surfaces — the most recent attempt is usually the more informative one.

### What does and does not fall back

| Failure | Falls back? | Why |
|---------|-------------|-----|
| `Model` (transient) | Yes | Another provider may well succeed |
| `Quota` / auth | **Yes** | Permanent for *that* provider, but says nothing about another — this is precisely what fallback is for |
| `ContextWindowOverflow` | Yes | A model with a larger window may accept it |
| `Cancelled` | **No** | Cancellation is the caller's decision; falling back would override it |
