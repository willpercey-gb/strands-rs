# Upstream Sync Marker

strands-rs is a Rust port of the [AWS Strands Agents SDK](https://github.com/strands-agents/sdk-python).
This file records exactly which upstream state the port is aligned to, so the next
sync is a diff rather than an archaeology dig.

## Current alignment

| Field | Value |
|-------|-------|
| Upstream repo | `https://github.com/strands-agents/sdk-python` |
| Aligned to tag | `python/v1.53.0` |
| Tag commit | `bc37995231143870c649a5c43fbdf798f5517786` |
| Tag date | 2026-08-20 |
| Sync status | 59/80 tracked items landed — see [`docs/12-upstream-sync-ledger.md`](docs/12-upstream-sync-ledger.md) for what remains and why |

### Previous alignment

| Field | Value |
|-------|-------|
| Aligned to tag | `python/v1.37.0` |
| Tag commit | `50439e01514c9a8bf59ca041a2699367a0263a17` |
| Tag date | 2026-04-22 |

## Upstream layout note

As of mid-2026 upstream restructured into a **monorepo**. The Python SDK moved:

```
src/strands/            →  strands-py/src/strands/
```

Sibling packages (`strands-ts/`, `strands-mcp/`, `site/`) and the vacuumed-in
`evals`/`harness-sdk` histories now share the same commit log. Filter to the
Python SDK or the commit counts are wildly inflated — the v1.37→v1.53 range is
1830 commits total but only **233** touch `strands-py/src/strands`.

Release tags are `python/vX.Y.Z` post-restructure; bare `vX.Y.Z` tags exist for
releases up to v1.41.0.

## How to run the next sync

```sh
# 1. Refresh upstream
cd ../strands && git fetch origin && git merge --ff-only origin/main

# 2. What changed in the Python SDK since we last aligned?
git log --oneline python/v1.53.0..HEAD -- strands-py/src/strands

# 3. Features only, grouped by release
git log --format='%s' python/v1.53.0..HEAD -- strands-py/src/strands | grep -E '^feat'
```

### Method

Use **releases for the ledger, HEAD for the code.**

- Releases give *coverage* (nothing silently missed), *provenance* (each line
  points at a commit you can read for rationale), and *dependency order* for
  free — upstream had to land middleware before the stages that plug into it,
  storage before memory, interrupts before checkpointing.
- HEAD gives the *code*. Implement each item once in its final form. Do not
  replay upstream's own churn: the middleware system alone was introduced in
  v1.44 and reworked in v1.50, v1.51 and v1.52, and replaying that means
  writing it four times.

strands-rs is an idiomatic reimagining, not a transliteration — hooks are a
`&mut HookEvent` enum where Python uses typed event classes, and
`ConversationManager` / `SessionManager` have different contracts. A
line-by-line diff replay does not apply cleanly and should not be attempted.
