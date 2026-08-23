# Interrupts

An interrupt pauses the agent to ask a human something — approve this deletion,
choose between these options — and resumes once answered.

## The cycle

1. A hook calls `event.interrupt(name, reason)`.
2. It returns `None` — no answer yet — so the hook declines to act.
3. The loop stops with `StopReason::Interrupt`, returning the pending interrupts.
4. The caller answers via `agent.respond(...)` and re-invokes.
5. The same hook runs again; this time `interrupt` returns the answer.

## Asking

```rust,ignore
agent_builder.hook(|event: &mut HookEvent| {
    if let HookEvent::BeforeToolCall(e) = event {
        if e.tool_name != "delete_records" {
            return;
        }
        match e.interrupt("approve_delete", Some(json!("Delete these records?"))) {
            None => e.cancel = true,               // no answer yet — do not act
            Some(answer) if answer != "approve" => e.cancel = true,
            Some(_) => {}                          // approved; let it run
        }
    }
});
```

The `None` arm is the whole safety property, which is why `interrupt` returns an
`Option` rather than setting a flag a hook could forget to check.

## Answering

```rust,ignore
let result = agent.prompt("delete the old records").await?;

if result.is_interrupted() {
    for interrupt in &result.interrupts {
        println!("{}: {:?}", interrupt.name, interrupt.reason);
    }

    let id = result.interrupts[0].id.clone();
    agent.respond(&[InterruptResponse::new(id, "approve")]);

    let result = agent.prompt("continue").await?;   // resumes
}
```

## Guarantees

- **The loop stops at a turn boundary**, after tool results are already in the
  history — so a resumed run continues from a conversation the provider accepts,
  not one with a `ToolUse` nothing answered.
- **Responses match by id.** An unknown id is ignored rather than applied to
  some other pending interrupt.
- **A completed run clears unanswered requests**, so a stale one cannot stall the
  next invocation. Answers are retained for the rest of the cycle.
- **State is serializable**, so interrupts survive a process restart between
  being raised and being answered.

## Difference from upstream

Upstream's Python SDK raises an exception out of the hook and re-enters it on
resume. Rust hooks are plain `Fn(&mut HookEvent)` with no unwinding contract, so
the request is *recorded* on the event and the hook returns normally. The
observable handshake is identical.
