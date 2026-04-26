# ADR-0013: Code Mode supersedes DSML XML for tool calls

- **Status**: proposed → accepted-v0.3 (this ADR)
- **Date**: 2026-04-25
- **Deciders**: Dmytro
- **Related**: AUGMENTATIONS rows AGT-01 (Code Mode, accepted-v0.3),
  ETC-02 (revised — DSML XML now rejected), MEM-04 (in-context
  injection patterns survive),
  `docs/specs/v0.2-onward.md` v0.3 section
- **Sources**: [Cloudflare Code Mode + MCP](https://blog.cloudflare.com/code-mode-mcp/)
  (Feb 2026), [Anthropic Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
  (independent, Feb 2026)
- **Supersedes:** ETC-02 in `docs/AUGMENTATIONS.md`

## Context

The original AUGMENTATIONS.md row ETC-02 listed:
> "MCP integration alongside DSML — accepted-v0.1 (schema only).
> DSML kept as fallback."

DSML (proprietary XML tool-call schema) was reserved as a fallback
in case MCP adoption stalled. Q1 2026 evidence (research review
2026-04-25):

- **MCP has won at scale.** 97M monthly SDK downloads, ~10K
  active public servers, ~2K in registry as of March 2026.
  OpenAI / Google / AWS / Cloudflare / Microsoft all adopted.
  ([New Stack](https://thenewstack.io/model-context-protocol-roadmap-2026/))
- **MCP 2026 roadmap** confirms governance maturation, OAuth 2.1
  + PKCE for Streamable HTTP, MCP Server Cards. ([MCP roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/))
- **Code Mode** (Cloudflare + Anthropic, independent, Feb 2026):
  the biggest tool-use shift of the year. Convert MCP tools into
  a TypeScript API; the LLM writes code that calls them in a
  sandbox. Measured wins:
  - Cloudflare: 99.9% input-token reduction on Cloudflare API
    exposure
  - Anthropic: 98.7% reduction on a Drive→Salesforce scenario
  - Both: 30-40% latency drop
  - Single tool call → single composition step in the model's
    output rather than dozens of round trips for multi-step tasks.
- **DSML XML schema:** not a recognized industry term in Q1 2026
  search. No frontier project uses XML tool-call schemas as a
  primary protocol; Anthropic's own internal practice has shifted
  to Code Mode + MCP.

So:
1. MCP is no longer at risk of stalling — it's the default.
2. The fallback we reserved (DSML XML) is itself obsolete; the
   actual successor pattern is Code Mode, not XML.

## Decision

Two-part decision:

**(a) Reject DSML XML.** Remove DSML XML from the planned protocol
stack. The tokenizer slots reserved for XML tool-call markers are
freed for other uses (or stay reserved unused per TOK-04 lazy
allocation). No code is written against DSML XML.

**(b) Accept Code Mode (AGT-01) for v0.3.** Tool calls in saint-llm
are emitted as **executable code in a sandbox**, not as
schema-tagged structured output. Workflow:

1. Tools registered via MCP servers (the protocol stays MCP).
2. At inference time, a **Code Mode runtime** generates a
   TypeScript-or-Python API surface from the available MCP
   tools. The model receives this API as a code-callable
   namespace (e.g. `tools.search.web(...)`, `tools.fs.read(...)`).
3. The model writes a code snippet that calls those APIs. The
   runtime executes the snippet in a sandbox (function_call /
   container / microvm tier per ROADMAP v0.3 D0.3.4). The result
   comes back as structured output for the next model turn.
4. Tool errors flow through the same code-execution channel —
   exceptions become the runtime's response, just as in normal
   programming.

## Consequences

**Intended:**
- 80%+ input-token cost reduction at production: tools no longer
  need full schema in the prompt; the API surface is referenced,
  not enumerated.
- 30-40% latency drop measured at frontier — fewer round trips
  for multi-step tasks.
- Composes with Anthropic Skills (AGT-02): a Skill folder can
  contain helper code that the runtime exposes as part of the
  tool API.
- Native fit with sandbox tier (AGT-06 / AGT-07 / saint-llm
  sandbox package): code execution is what the sandbox is for.
- Aligns saint-llm with the frontier convention rather than
  inventing an idiosyncratic schema (DSML XML).

**Unintended / accepted:**
- Existing `Tool` / `ToolRegistry` / `ToolCall` abstractions in
  `packages/agents/` remain valid — they describe the *logical*
  tool surface — but the **runtime path** for invoking them at
  v0.3 changes from "MCP/JSON tool-call message" to "Code Mode
  TS-API + sandbox execution". The agent-runtime code in
  `packages/agents/` doesn't know or care which path is used;
  it produces and consumes `Message` + `ToolCall` regardless.
- The `agents` package's current direct-call `ToolRegistry.execute`
  is the v0.0 ergonomic shortcut for tests and trivial
  single-process tools. It coexists with Code Mode (used in v0.3)
  via a runtime adapter — Code Mode is a `Policy` implementation
  detail, not a replacement for the abstractions.
- The model needs to be trained to write tool-call code, not
  emit a structured tool-call envelope. v0.2 post-training data
  must include code-calling traces. This is a v0.2 deliverable.
- Sandbox is now load-bearing: without a working sandbox at v0.3,
  Code Mode can't ship. Sandbox tiers (function_call / container /
  microvm / fullvm) per `packages/sandbox/` are blocking.

**Explicit non-effects:**
- Does NOT change `Message` / `ToolCall` dataclass shape.
- Does NOT change MCP adoption — MCP remains the *protocol*; Code
  Mode is how the LLM *consumes* MCP-described tools.
- Does NOT change A2A protocol plans (AGT-03) — A2A is for
  cross-vendor agent comms; Code Mode is for tool execution.
  Orthogonal.

## Alternatives considered

- **Keep DSML XML as fallback "in case MCP fails."** Rejected:
  MCP didn't fail. Reserving slots for an obsolete fallback is
  pure debt with no upside.
- **Use OpenAI/Anthropic structured tool-call envelopes** (JSON
  schema in/out) as the primary path. Considered. Works at small
  scale. At ≥20-tool surfaces, breaks down — input tokens
  explode, multi-step composition requires N round trips. Code
  Mode is the empirically-measured fix.
- **Hybrid: Code Mode for ≥N tools, structured JSON for ≤N.**
  Considered. Adds runtime branching for marginal benefit. v0.3
  ships Code Mode-only; if specific deployments need
  JSON-envelope serving, they can switch via `Policy` choice
  without affecting the agent abstraction.
- **Defer to v0.4.** Considered, rejected. Code Mode is the v0.3
  agent-runtime story. Without it, v0.3's "agent runtime
  integrated end-to-end" exit criteria can't be met (input-token
  cost reduction is one of the criteria).

## Implementation notes

### Files affected (v0.3 work, named here for traceability)

- `packages/agents/src/saint_llm_agents/code_mode/` — new
  subpackage:
  - `tool_to_api.py` — convert `ToolSpec` registry into a
    TypeScript or Python API stub.
  - `sandbox_executor.py` — execute LLM-generated code in
    sandbox.
  - `parse_response.py` — extract structured result from
    sandbox output.
  - `code_mode_policy.py` — `Policy` implementation that wraps
    a `SaintLLMPolicy` (or external-LLM policy) and routes
    tool-call generation through the Code Mode path.
- `packages/agents/src/saint_llm_agents/runtime.py` — no changes;
  Code Mode is a Policy detail, not a Runtime detail.
- `packages/sandbox/` — populate with the four substrate
  implementations (function_call / container / microvm / fullvm)
  per existing pyproject description.

### Tests

- Unit tests for the TS-API generator from a `ToolRegistry`.
- Unit tests for sandbox-executor with a mock sandbox.
- Integration test: end-to-end Code Mode roundtrip with a
  real Python sandbox (function_call tier) and a fake LLM
  policy that emits a known code snippet.

### v0.2 post-training data implication

v0.2 SFT + RL data must include Code-Mode-style tool-calling
traces. Otherwise the v0.3 model won't know how to write
tool-calling code. Capture:
- Synthetic agentic traces with code-call patterns.
- Existing open SFT datasets with code-calling subsets (Tülu 3
  Tools, MUA-RL traces, DeepSWE traces).

### Out of scope for this ADR

- A2A protocol details (AGT-03) — separate ADR at v0.3 work.
- Sandbox tier implementation choices (build vs E2B/Daytona) —
  separate ADR.
- Memory tool integration with Code Mode — separate ADR.

### Tokenizer slot disposition

DSML XML reserved slots (estimated ~50-100 control tokens for
`<tool>`, `</tool>`, `<arg>` etc. in TOK-04 lazy allocation)
are released for other use. Lazy allocation means no actual
embedding rows are wasted; the slot IDs simply remain unused.

### AUGMENTATIONS revisions

- **AGT-01 Code Mode**: status `proposed` → `accepted-v0.3`
- **ETC-02 MCP integration**: revised — DSML XML rejected,
  MCP-only. Tokenizer-slot allocation deferred per TOK-04.
- **TOK-04 reserved control slots**: no change needed; DSML XML
  slots stay reserved (lazy) but unused.
