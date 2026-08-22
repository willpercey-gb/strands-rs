# Upstream Sync Ledger — python/v1.37.0 → python/v1.53.0

Triaged inventory of everything that changed in the Python SDK between the state
strands-rs was ported from and the current upstream release. See
[`UPSTREAM.md`](../UPSTREAM.md) for the sync marker and method.

**Range:** 16 releases, Apr–Aug 2026. 233 commits touching `strands-py/src/strands`
(93 `feat`, 110 `fix`, 12 `refactor`, 4 breaking).

**Scope decision:** port everything portable — existing-surface parity, core
architecture, and the new subsystems. Excluded only where the feature is
inherently tied to a provider or runtime strands-rs does not target.

## Legend

| Mark | Meaning |
|------|---------|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Landed |
| `N/A` | Not portable — reason given |

---

## Phase 0 — Type foundations

These reshape types that everything else depends on. Nothing in later phases
lands cleanly until these do.

| | Item | Upstream | Notes for Rust |
|---|------|----------|----------------|
| `[x]` | Durable message tracking ids | v1.47 `feat: add durable identifiers to messages` | `Message` gains `tracking_id: Option<String>` (UUID v4, assigned on push, stripped before model calls, survives session round-trip). Touches `types/message.rs`, session serde, every adapter's request builder. |
| `[x]` | Message metadata envelope | v1.47 | `MessageMetadata { usage, metrics, custom }` on `Message`, `NotRequired`-equivalent. Stripped before model calls, persisted in sessions. |
| `[~]` | Cache points + TTL | v1.38 `feat(cache): add TTL support to CachePoint`, v1.41, v1.53 | New `ContentBlock::CachePoint { cache_type, ttl }`. Also `SystemContentBlock` — system prompt becomes `String \| Vec<SystemContentBlock>` so cache points can sit in it. This is the `split_system_prompt` change; it alters the `Model::stream` signature. **Types + threading landed** (`SystemPrompt`, `SystemContentBlock`, `CachePoint`, `Model::stream` now takes `Option<&SystemPrompt>`); still outstanding: mapping a cache point onto each provider's own `cache_control` field. |
| `[x]` | Audio content blocks | v1.53 `feat(python): add audio content blocks` | `ContentBlock::Audio`. |
| `[x]` | Document + video content blocks | pre-existing upstream, never ported | `ContentBlock::Document`, `ContentBlock::Video`. strands-rs only has Text/ToolUse/ToolResult/Image. |
| `[x]` | Reasoning content blocks | pre-existing + v1.51 `fix(streaming): consume reasoning signature per content block` | `ContentBlock::Reasoning { text, signature, redacted }`. Needed by Claude/Gemini adapters, which currently drop it. |
| `[x]` | Citations content block | pre-existing, never ported | `ContentBlock::Citations`. |
| `[x]` | Guard content block | pre-existing, never ported | `ContentBlock::GuardContent`. Low priority — Bedrock-facing. |
| `[x]` | Tool annotations in `ToolSpec` | v1.53 `feat(mcp): surface tool annotations in ToolSpec` | `ToolSpec.annotations` (readOnlyHint, destructiveHint, idempotentHint, openWorldHint). Feeds `strands-claude-mcp`. |
| `[x]` | `Usage` / `Metrics` split + cache token counters | v1.42 gemini cache tokens, v1.53 `py-anthropic` cache_read/write | Rust `Usage` is missing `cache_read_input_tokens`, `cache_write_input_tokens`. `total_duration_ns` should move to a separate `Metrics` type to match. |

**Phase 0 status:** landed. `Model::stream` now takes `Option<&SystemPrompt>`;
adapters without cache-point support call `SystemPrompt::as_text()`. Messages
carry durable `tracking_id` + `MessageMetadata`, both stripped by
`Message::for_model()` before every model call. `Usage` gained cache-token
counters and split from the new `Metrics`; the event loop accumulates both.

## Phase 1 — Parity on existing surface

Items that touch subsystems strands-rs already has.

### Token accounting

| | Item | Upstream | Notes |
|---|------|----------|-------|
| `[ ]` | `Model::count_tokens` | v1.38 (×3), v1.39, v1.40 | Land the **final** shape only: trait method with a default estimator, provider override where native counting exists, `use_native_token_count` flag defaulting to **false** (v1.40 `fix: set use_native_token_count default to false`), and error caching for providers that reject it. Do not replay the four-step evolution. |
| `[ ]` | Pre-call input token estimation | v1.38 `feat: estimate input tokens before model calls` | |
| `[ ]` | Context window limit table | v1.39 `feat: add context window limit lookup table`, v1.43, v1.51 | Static map of model id → context window. v1.51 adds Claude 5 and GPT-5.6 families. |
| `[ ]` | `context_window_limit` on model configs | v1.37-era `feat: add context_window_limit to model configs` | |
| `[ ]` | `Model::estimate_utilization` | v1.51 `feat(model): add estimateUtilization method` | |
| `[ ]` | Count JSON blocks when counting tokens | v1.40 `fix(core): include json blocks in counting tokens` | |

### Conversation management

| | Item | Upstream | Notes |
|---|------|----------|-------|
| `[ ]` | Proactive context compression | v1.40 `feat: add proactive context compression to conversation managers` | New `conversation/compression/` module. Compresses *before* overflow rather than reacting to it. |
| `[ ]` | Message pinning | v1.43 `feat(context): add message pinning to conversation managers` | Pinned messages survive reduction. `conversation/compression/pin_message.rs`. |
| `[x]` | `window_size = 0` handling | v1.44 `fix(conversation-manager): handle window_size=0 and reject negative values` | **Live bug in strands-rs**: `SlidingWindowConversationManager` with `window_size: 0` drains the entire history. |
| `[x]` | Fallback trim point for tool-heavy conversations | v1.37-era `fix: add fallback trim point ... in SlidingWindowConversationManager` | Current Rust `drain(..n)` can split a `ToolUse` from its `ToolResult`, producing an invalid history. |
| `[ ]` | `context_manager="auto"` facade | v1.43 `feat(context): add context_manager="auto" facade on Agent` | |
| `[ ]` | Agentic context management | v1.44 `feat: port agentic context management to python` | `_context_manager/modes/agentic/`. Model-driven context curation. |

### Hooks

| | Item | Upstream | Notes |
|---|------|----------|-------|
| `[ ]` | Batch `BeforeTools` / `AfterTools` events | v1.51 `feat(python): add BeforeToolsEvent and AfterToolsEvent batch hooks` | Fire once around the whole tool batch, not per tool. New `HookEvent` variants. |
| `[ ]` | Hook ordering | v1.43 `feat(strands-py): add optional hook order` | `HookRegistry` currently dispatches in registration order with no way to influence it. |
| `[ ]` | After-tool-call duration | v1.53 `feat: add after tool call duration` | `AfterToolCallEvent.duration`. |
| `[x]` | Count usage from hook-retried model calls | v1.45 `fix(python): count usage from hook-retried model calls` | **Already correct in strands-rs** — `event_loop.rs:197` accumulates `cycle_usage` before the retry `continue`. Verified, no change needed. |

### Tools

| | Item | Upstream | Notes |
|---|------|----------|-------|
| `[ ]` | Agent-as-tool delegation | v1.53 `feat(py): add agent-as-tool delegation` | `agent/_agent_delegation.py`. Richer than the current `AgentTool` wrapper — supports handing the sub-agent the live conversation rather than a fresh prompt. |
| `[ ]` | Structured output | pre-existing upstream, never ported | `tools/structured_output/`. Schema-constrained responses via a synthetic tool. Sizeable. |
| `[ ]` | Pluggable tool executors | pre-existing upstream, never ported | strands-rs has a `concurrent_tools: bool`; upstream has a `ToolExecutor` trait with concurrent/sequential impls. Needed before middleware's `ExecuteToolStage`. |
| `[ ]` | Bound tool schema normalization recursion | v1.49 `fix(core): bound tool schema normalization recursion depth` | |
| `[x]` | Concurrent tool results in request order | v1.44 `fix(core): keep concurrent tool results in request order` | Rust `join_all` already preserves order — verify and note. |
| `[ ]` | Do not synthesize exception for cancelled tools | v1.50 `fix(core): do not synthesize exception for cancelled tools` | |
| `[x]` | Retry re-invokes with original input | — | **Live bug in strands-rs**: `execute_tools_concurrent` retries with `Value::Null` instead of the original input (`event_loop.rs:962`). Sequential path is correct. |

### Multi-agent

| | Item | Upstream | Notes |
|---|------|----------|-------|
| `[ ]` | `MultiAgentPlugin` for Swarm/Graph | v1.41 `feat(plugins): add MultiAgentPlugin` | Plus `plugins/multiagent_registry.py`. |
| `[ ]` | `invocation_state` into edge conditions | v1.44 (re-landed after v1.42 revert) | Note the revert — take the v1.44 form. |
| `[ ]` | `Display` for `MultiAgentResult` / `NodeResult` | v1.51 `feat(multiagent): add __str__ support` | |
| `[ ]` | Accumulate cache token counters in Graph/Swarm | v1.51 `fix(multiagent)` | |
| `[ ]` | Preserve failed status from graph nodes | v1.46 `fix(multiagent)` | |
| `[ ]` | Preserve shared context across serialize/deserialize | v1.52 `fix(multiagent)` | |
| `[ ]` | Graph resume: AND-join edge/fan-in fixes | v1.48, v1.50 `fix(graph)` ×2 | Only relevant once interrupts/checkpointing land. |
| `[ ]` | `reset_executor_state` state corruption | v1.45 `fix(graph)` | |
| `[ ]` | Swarm crash-restart resume | v1.49 `fix(swarm)` | |

### Sessions

| | Item | Upstream | Notes |
|---|------|----------|-------|
| `[ ]` | Snapshot session manager | v1.51 `feat: add snapshot session manager to python` | Plus `types/_snapshot.py` and v1.43 `feat: add model_state as a snapshot field`. |
| `[ ]` | S3 session manager | pre-existing, never ported | Plus v1.42 `feat: add endpoint_url parameter to S3SessionManager`. |
| `[x]` | Symlink attack prevention | v1.47 `fix(session): prevent symlink attacks in FileSessionManager` | **Applies directly** — Rust `FileSessionManager` joins an unsanitised `session_id` into a path. |
| `[ ]` | Repair mid-iteration skip of orphaned toolUse | v1.50 `fix(session)` | |

### Agent

| | Item | Upstream | Notes |
|---|------|----------|-------|
| `[ ]` | `Limits` on invoke/stream | v1.42 `feat: add Limits and support it during invoke/stream` | Supersedes the bare `max_cycles: usize`. |
| `[ ]` | Per-invocation idempotency token | v1.45 `feat: added per-invocation idempotency support` | Plus v1.49 `fix(agent): stop idempotency waiters from blocking thread-pool workers`. |
| `[ ]` | Configurable retry exceptions | v1.50 `feat(py): configurable retry exceptions` | Generalises the existing `classify_cli_failure` / `StrandsError::Quota` short-circuit. |
| `[ ]` | Agent state as a typed store | pre-existing `agent/state.py`, never ported | strands-rs uses a bare `HashMap<String, Value>`. |

## Phase 2 — Core architecture

Load-bearing structure in Python that strands-rs has no equivalent for. These
reshape the event loop, so they land before Phase 3 depends on them.

| | Item | Upstream | Notes |
|---|------|----------|-------|
| `[ ]` | **Middleware / stage system** | v1.44 → v1.52 | Land the **v1.52 final shape only**: `_middleware/{registry,stages,types}` with `InvokeModelStage`, `ExecuteToolStage`, `AgentStreamStage`. Introduced v1.44, reworked v1.46 (result handling, model-state isolation, system-prompt fidelity), v1.50, v1.51, v1.52. Replaying costs 4× and converges on the same place. |
| `[ ]` | **Interrupts** | pre-existing + v1.50, v1.51 | `interrupt.py`, `types/interrupt.py`. Pause/resume mid-tool with a caller round-trip. Prerequisite for checkpointing and for the graph/swarm resume fixes. v1.50 adds middleware-initiated interrupts; v1.51 adds per-call MCP tool cancellation and A2A round-trip. |
| `[ ]` | **Checkpointing** | v1.43 `feat(checkpoint): wire checkpointing into agent event loop` | `experimental/checkpoint/`. Depends on interrupts + snapshot sessions. |
| `[ ]` | **Model routing + fallback** | v1.51, v1.52 | `models/routing/{router,strategy,fallback_strategy}`. `ModelRouter` accepted via `Agent(model=)`; per-call model threaded through `InvokeModelStage`. Depends on middleware. |
| `[ ]` | Structured output context | pre-existing | `tools/structured_output/_structured_output_context.py`. Pairs with the Phase 1 structured output item. |

## Phase 3 — New subsystems

Fresh ports. Each is self-contained; order within the phase is by dependency.

| | Item | Upstream | Notes |
|---|------|----------|-------|
| `[ ]` | **Unified storage** | v1.48 `feat(py): add unified storage interface`, v1.52 `feat(storage-py): add top level storage` | `storage/{storage,in_memory_storage,local_file_storage,s3_storage}`. Land first — memory and the context offloader both build on it. Plus v1.50 `fix(core): reject keys for s3 storage if not configured`. |
| `[ ]` | **Memory** | v1.44 (×4), v1.45, v1.46 (×3), v1.53 | `memory/` — manager, extraction (coordinator, model extractor, triggers, config resolution), types. Plus `vended_memory_stores/` (Bedrock KB, test store). Agent gains a `memory_manager` param with configurable sync auto-flush. Note v1.45 renamed `LocalMemoryStore` → `TestMemoryStore` (breaking) — take the new name. |
| `[ ]` | **Sandbox** | v1.43 (×2), v1.44 | `sandbox/{base,docker,ssh,posix_shell,stream_process,not_a_sandbox_local_environment,constants,errors,types}`. Core abstraction + Docker/SSH impls + Agent integration. |
| `[ ]` | **Interventions** | v1.43, v1.44 (×2), v1.51 | `interventions/{actions,handler,registry}` + `vended_interventions/cedar/` (Cedar authorization, schema generator, file loaders) and `vended_interventions/hitl/` (human-in-the-loop, + v1.51 LLM risk classifier). Cedar is a policy-engine dependency — flag if no suitable Rust crate exists. |
| `[ ]` | **Telemetry / OTel** | pre-existing + v1.47, v1.48, v1.51 | `telemetry/{config,metrics,metrics_constants,tracer}`. Plus span redaction (v1.47), `gen_ai_span_attributes_only` env var (v1.48), bidi telemetry (v1.51), and the fixes: thread-safe `MetricsClient` singleton, `gen_ai.tool.call.arguments/result` on execute_tool spans, `tool_trace` on interrupted calls, dropped `gen_ai.agent.name` from multiagent spans. |
| `[ ]` | **Vended tools** | v1.50 (×3), v1.53 | `vended_tools/` — `shell` (renamed from `bash` in v1.50, breaking; keep deprecated aliases per v1.51 fix), `http_request`, `sleep`, `stop`, `file_editor`. |
| `[ ]` | **Context offloader plugin** | v1.38, v1.44, v1.45, v1.51 + fixes | `vended_plugins/context_offloader/{plugin,storage,search}`. Large tool result offload (v1.38), turn-based eviction (v1.44), search/grep retrieval (v1.45), `should_offload` callback (v1.51). Depends on unified storage. |
| `[ ]` | **Skills plugin** | pre-existing + v1.37-era fix | `vended_plugins/skills/`. Plus `fix(skills): preserve cache points in system prompt during skills injection` — depends on Phase 0 cache points. |
| `[ ]` | **Goal loop plugin** | v1.44 `feat(strands-py): add GoalLoop vended plugin` | `vended_plugins/goal/{plugin,judge}`. |
| `[ ]` | **Context injector plugin** | pre-existing + v1.53 `feat: add injected content behind cache points` | `vended_plugins/context_injector/` and `injection/{_message_injection,_xml,types}`. |
| `[ ]` | **Steering plugin** | pre-existing, never ported | `vended_plugins/steering/` — context providers, actions, LLM handler + mappers. Note upstream has it in both `experimental/` and `vended_plugins/`; take `vended_plugins/`. |
| `[ ]` | Plugin discovery | v1.44-era `plugins/_discovery.py` | |
| `[ ]` | MCP client parity | v1.38, v1.42, v1.45, v1.46, v1.47, v1.48, v1.51, v1.53 | Progress notifications, JSON server config loading, `continue_on_error`, `client_name`, per-call cancellation, OAuth for streamable HTTP, `isError` preservation, content-to-tool-result public API. Feeds `strands-claude-mcp`. |

## Phase 4 — Portable bug fixes

Fixes against subsystems strands-rs has, not already covered above.

| | Item | Upstream |
|---|------|----------|
| `[ ]` | Ollama: generate unique `toolUseId` instead of reusing tool name | v1.38 `fix(ollama)` |
| `[ ]` | Ollama: avoid crash on empty model stream | v1.46 `fix(ollama)` |
| `[ ]` | Ollama/llama/mistral/writer: raise context-window-overflow error | v1.42 `fix(models)` |
| `[x]` | Streaming: warn when tool input JSON is malformed | v1.49 `fix(streaming)` — strands-rs currently swallows this with `unwrap_or(default)` |
| `[ ]` | Streaming: handle tool use metadata in `contentBlockDelta` for non-standard models | v1.44 `fix(streaming)` |
| `[ ]` | Preserve non-ASCII text in tool-result / tool-call serialization | v1.47, v1.48 `fix(core)` ×2 |
| `[ ]` | Handle `None`/empty text in message content sanitization | v1.45, v1.51 `fix(core)` ×2 |
| `[ ]` | Include reasoning block with empty text and non-empty signature | v1.51 `fix(core)` |
| `[ ]` | Event loop: log exception type, not full traceback, on cycle failure | v1.46 `fix(event_loop)` |
| `[ ]` | Clarify max-tokens-reached error message | v1.44 `fix(core)` |
| `[ ]` | Recover message on max tokens reached | pre-existing `event_loop/_recover_message_on_max_tokens_reached.py`, never ported |
| `[ ]` | Tools: load directory tools under a namespaced module key | v1.50 `fix(tools)` — only if a tool loader is ported |

---

## Not portable

| Area | Items | Reason |
|------|-------|--------|
| Bedrock | 12 fixes + `strict_tools`, cache-point placement, default model bump, on-demand throughput, guardContent, 3gp normalization, region verification | No Bedrock adapter in strands-rs and none planned. Revisit if one is added. |
| Anthropic SDK provider | `model_dump` bypass, webp mapping, unsupported format rejection, `cache_config`/`cache_tools` | strands-rs reaches Anthropic via `strands-claude-cli`, not the Python SDK path. The **cache_config concept** is captured in Phase 0 instead. |
| OpenAI / Responses | 8 fixes — vllm reasoning deltas, mantle base-path routing, document `file_data`, non-streaming completions, tool list mutation, assistant text replay, cached token reporting, AWS profile support | strands-rs has no OpenAI SDK adapter; `strands-openrouter` and `strands-codex-cli` cover that ground differently. |
| Gemini SDK provider | safety-blocked metadata, tool-use-after-reasoning, empty tools in Vertex mode, throttling detection, thought signature deltas, tool choice | strands-rs uses `strands-gemini-cli`. **Thought-signature-as-reasoning-delta** is worth taking — folded into Phase 0 reasoning blocks. |
| Other providers | LiteLLM, SageMaker, Mistral 2.x, llama.cpp, LlamaAPI, Writer | No adapters. |
| A2A | 5 items — task lifecycle, agent factory, `agent_card_url`, interrupt round-trip, conversation isolation | Agent-to-Agent protocol server; out of scope for the Rust port unless explicitly wanted. |
| Bidi | 11 items — Nova Sonic, Gemini Live, OpenAI Realtime, echo suppression, 8kHz audio, telemetry | Bidirectional realtime audio. Very large, provider-bound, and depends on `experimental/bidi/_async` task machinery. Out of scope. |
| Python-specific | `FieldInfo` unwrap, pydantic warnings, `MCPClient.__exit__` annotations, interpreter-finalization cleanup, thread-pool blocking, doc URL fixes | No Rust analogue. |

---

## Live bugs found in strands-rs during triage

Not upstream ports — defects in the existing Rust code surfaced by reading the
corresponding upstream fixes. Worth landing early and independently.

1. ~~**`event_loop.rs`** — concurrent tool retry re-invokes with `Value::Null`
   instead of the original input.~~ **Fixed.** Regression test
   `test_concurrent_tool_retry_reuses_original_input` verified failing against
   the unfixed code (`left: {"name":"World"}, right: Null`).
2. ~~**`event_loop.rs`** — a hook that always sets `retry: true` on
   `AfterModelCall` loops forever.~~ **Fixed** via `RetryConfig.max_hook_retries`
   (default 3). Verified: without the guard the suite hangs outright.
3. ~~**`sliding_window.rs`** — `window_size: 0` drains the whole history.~~
   **Fixed** — now explicit "clear all" semantics matching upstream v1.44.
4. ~~**`sliding_window.rs`** — `drain(..n)` can separate a `ToolUse` from its
   `ToolResult`.~~ **Fixed** — new `conversation/trim.rs` ports upstream's
   `find_valid_trim_point` / tool-pair fallback; the window may now retain
   slightly more than `window_size` rather than emit an invalid history.
5. ~~**`session/file.rs`** — `session_id` joined into a path unsanitised.~~
   **Fixed** — id validation rejects separators/traversal, writes are atomic
   (temp + rename), symlinks are refused on read and write, and
   `with_default_dir()` uses a 0700 `~/.strands/sessions` instead of shared temp.
6. ~~**`event_loop.rs`** — malformed tool-input JSON silently swallowed.~~
   **Fixed** — now warns with tool name, id and the truncated raw fragment,
   matching upstream v1.52.
