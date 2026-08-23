# Interventions

An intervention is policy over what the agent is allowed to do. A hook can
already cancel a tool call; an intervention is the structured version — allow,
deny with a reason, or escalate to a human.

Keeping deny and escalate distinct matters: a denial must be final and explain
itself to the model, whereas an escalation must pause rather than guess on the
human's behalf. A boolean loses that.

## Built-in policies

```rust,ignore
use strands_core::interventions::{AllowList, DenyList, InterventionRegistry};

let mut registry = InterventionRegistry::new();
registry.register(DenyList::new(["delete_everything"]));
registry.register(AllowList::new(["search", "read_file", "summarize"]));
```

`AllowList` fails closed by construction — a tool added later is denied until
someone permits it explicitly, which is what makes it safe as the outermost
policy.

## Evaluation order

Handlers run in registration order and the **first non-allow decision wins**.
A later handler cannot un-deny what an earlier one denied.

## Human-in-the-loop

```rust,ignore
use strands_core::interventions::{HumanInTheLoop, RiskLevel, ToolNameClassifier};

registry.register(HumanInTheLoop::for_tools(["deploy", "delete_records"]));
```

Escalation rides the [interrupt](15-interrupts.md) mechanism: the call is
withheld, a human is asked, and the same call is re-proposed once answered.

For an open-ended tool set — anything arriving from MCP — treat unrecognised
tools as suspicious rather than safe:

```rust,ignore
let classifier = ToolNameClassifier::new(Vec::<String>::new())
    .with_default_risk(RiskLevel::High);
registry.register(HumanInTheLoop::new(classifier));
```

## Applying them

```rust,ignore
use strands_core::interventions::InterventionExecutor;
use strands_core::tool::SequentialToolExecutor;

let agent = Agent::builder()
    .model(model)
    .tool_executor(InterventionExecutor::new(SequentialToolExecutor, registry))
    .build()?;
```

This wraps the tool executor rather than hooking, because hooks are synchronous
and a policy that queries a service or a policy engine cannot run there.

## Custom policies

```rust,ignore
#[async_trait]
impl InterventionHandler for BudgetGuard {
    fn name(&self) -> &str { "budget" }

    async fn evaluate(&self, ctx: &InterventionContext) -> InterventionAction {
        if self.spend_today().await > self.cap {
            InterventionAction::deny("daily budget exhausted")
        } else {
            InterventionAction::Allow
        }
    }
}
```
