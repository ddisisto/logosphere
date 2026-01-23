# Substrate: Master Plan

## Vision

**Substrate** — the medium patterns propagate through.

Not the container (sphere), not the shape (field), not the motion (drift). The medium itself.

| Domain | Substrate as... |
|--------|-----------------|
| Biology | Growth medium, culture substrate, what organisms colonize |
| Chemistry | What enzymes act upon, transformed by interaction |
| Computing | Substrate independence — the pattern matters, not what it runs on |
| Geology | The layer beneath, what everything else rests on |

It doesn't center the AI or the human. Neither is the point. The substrate is.

**Research question:** What emerges on the substrate?

- Minds are processes that run on it
- Users are observers who occasionally perturb it
- Clusters are persistent structures
- Thoughts are transient excitations

Entry point: `subs`

---

## Architecture Overview

See `docs/architecture-greenfield.md` for full details. Summary:

```
substrate/
├── protocol/              # Mind I/O format (definition + implementation)
│   ├── system_prompt.py   # Compose system prompt from sections
│   ├── user_prompt.py     # Compose input YAML from sections
│   ├── parser.py          # Parse output using section knowledge
│   ├── types.py           # Shared types (MindOutput, Age, etc.)
│   └── sections/          # Per-section: definition + format + examples + parse
│
├── storage/               # Pure data persistence
│   ├── thoughts.py
│   ├── messages.py
│   ├── drafts.py
│   ├── users.py
│   └── io.py
│
├── session/               # Workspace management
│   ├── session.py
│   ├── config.py
│   └── lock.py
│
├── sampling/              # Selection logic
│   ├── recent.py
│   ├── random.py
│   ├── recall.py
│   └── limits.py
│
├── clustering/            # Semantic organization
│   ├── embeddings.py
│   ├── algorithm.py
│   └── manager.py
│
├── inference/             # LLM invocation
│   └── client.py
│
├── orchestration/         # Runtime coordination
│   ├── runner.py
│   ├── lifecycle.py
│   ├── metrics.py
│   └── conditions.py
│
├── interfaces/            # User-facing
│   ├── ops.py
│   ├── cli/
│   └── tui/
│
└── scripts/
    └── subs.py            # Single entry point
```

**Dependency flow:** interfaces → orchestration → sampling/protocol/clustering → session → storage + inference

---

## Principles

1. **Single responsibility** — each module does one thing well
2. **Dependency direction** — lower layers don't know about higher layers
3. **Domain boundaries** — clear separation, narrow interfaces
4. **Composition** — build from small pieces
5. **Explicit dependencies** — inject what you need
6. **Testable isolation** — each piece works alone
7. **Storage is pure** — no business logic in data layer
8. **Interfaces are leaves** — depend on everything, nothing depends on them
9. **Session config is truth** — no runtime overrides, no env except secrets + path
10. **Protocol is coherent** — definition, input, output always in sync

---

## Phased Implementation

### Phase 0: Repository Setup

**Goal:** Empty structure, tooling, CLAUDE.md foundation

**Tasks:**
- [ ] Create `substrate/` directory tree (empty `__init__.py` files)
- [ ] Set up `pyproject.toml` (rename from logosphere, new entry point)
- [ ] Create initial `CLAUDE.md` with structure overview
- [ ] Tag current logosphere as `pre-substrate`

**Outputs:**
- Working `uv sync`
- `subs --help` prints placeholder
- CLAUDE.md describes project structure and current phase

---

### Phase 1: Protocol

**Goal:** Define the mind I/O contract, generate system prompts and parse outputs

**Why first:** Protocol is the specification. Everything else implements it. Getting this right means the contract is locked before building around it.

**Tasks:**
- [ ] `protocol/types.py` — Age, MindOutput, Thought, Draft, Message types
- [ ] `protocol/sections/meta.py` — definition, format, examples
- [ ] `protocol/sections/thoughts.py` — including recent/sampled split
- [ ] `protocol/sections/dialogue.py` — history + awaiting
- [ ] `protocol/sections/drafts.py` — phase-aware (processing vs drafting)
- [ ] `protocol/sections/users.py` — presence, state history
- [ ] `protocol/sections/orientation.py` — re-orientation footer
- [ ] `protocol/system_prompt.py` — compose from sections
- [ ] `protocol/user_prompt.py` — compose input YAML
- [ ] `protocol/parser.py` — parse mind output YAML
- [ ] Tests: round-trip (generate input → parse example output)

**Outputs:**
- Versioned system prompt generated from code
- Example user prompts generated for various states
- Parser validated against example outputs
- CLAUDE.md updated with protocol section

**Key decisions embedded:**
- Recent/sampled thought split (from narrowing-draft RFC)
- Phase indicator (processing/drafting)
- No explicit signalling mechanics

---

### Phase 2: Storage

**Goal:** Pure data persistence layer

**Tasks:**
- [ ] `storage/io.py` — YAML/NPY utilities
- [ ] `storage/thoughts.py` — ThoughtPool (embeddings + metadata)
- [ ] `storage/messages.py` — MessageStore (append-only)
- [ ] `storage/drafts.py` — DraftStore (append-only)
- [ ] `storage/users.py` — UserRegistry (presence state)
- [ ] Tests: CRUD operations, persistence round-trips

**Outputs:**
- All storage classes working independently
- Test fixtures for each store type
- CLAUDE.md updated with storage section

**Notes:**
- No sampling logic here — just retrieval by ID, iteration, age
- No display limiting — that's sampling's job
- Generator patterns

---

### Phase 3: Session

**Goal:** Workspace facade, config, locking

**Tasks:**
- [ ] `session/config.py` — SessionConfig dataclass
- [ ] `session/lock.py` — file-based locking
- [ ] `session/session.py` — Session class (owns storage instances)
- [ ] Tests: session creation, config persistence, lock behavior

**Outputs:**
- Session can be created, opened, saved
- Config read/write works
- Locking prevents concurrent access
- CLAUDE.md updated with session section

---

### Phase 4: Sampling

**Goal:** Selection logic separate from storage

**Tasks:**
- [ ] `sampling/limits.py` — DisplayLimiter (chars/count)
- [ ] `sampling/recent.py` — deterministic recent selection
- [ ] `sampling/random.py` — random pool sampling
- [ ] `sampling/recall.py` — cluster-aware recall (configurable)
- [ ] Tests: limit behavior, selection properties

**Outputs:**
- Sampling functions work on storage objects
- Recall can be enabled/disabled via config
- CLAUDE.md updated with sampling section

---

### Phase 5: Clustering

**Goal:** Semantic organization

**Tasks:**
- [ ] `clustering/embeddings.py` — embedding client
- [ ] `clustering/algorithm.py` — HDBSCAN, centroid matching
- [ ] `clustering/manager.py` — ClusterManager (state, assignments)
- [ ] Tests: embedding, clustering, centroid updates

**Outputs:**
- Clustering works on thought pools
- Assignments persist correctly
- CLAUDE.md updated with clustering section

---

### Phase 6: Inference

**Goal:** LLM invocation (thin wrapper)

**Tasks:**
- [ ] `inference/client.py` — OpenRouter API wrapper
- [ ] Tests: mock API responses

**Outputs:**
- `invoke_mind()` accepts formatted prompt, returns raw response
- CLAUDE.md updated with inference section

---

### Phase 7: Orchestration

**Goal:** Iteration loop, phases, metrics, conditions

**Tasks:**
- [ ] `orchestration/metrics.py` — noise ratio, stability, production tracking
- [ ] `orchestration/conditions.py` — start/stop evaluation
- [ ] `orchestration/lifecycle.py` — phase transitions
- [ ] `orchestration/runner.py` — iteration loop (step, run)
- [ ] Tests: lifecycle transitions, stop conditions

**Outputs:**
- Runner can step through iterations
- Phase transitions work (processing → drafting)
- Metrics tracked correctly
- CLAUDE.md updated with orchestration section

---

### Phase 8: Interfaces

**Goal:** CLI and TUI

**Tasks:**
- [ ] `interfaces/ops.py` — shared operations
- [ ] `interfaces/cli/` — command handlers
- [ ] `interfaces/tui/` — Textual app (port existing)
- [ ] `scripts/subs.py` — unified entry point
- [ ] Tests: CLI commands, TUI smoke test

**Outputs:**
- `subs` launches TUI
- `subs <cmd>` runs commands
- Existing TUI functionality preserved
- CLAUDE.md finalized

---

## Session Coordination

Each session should:

1. **Start with context:**
   - Read current CLAUDE.md
   - Read this master plan
   - Read relevant RFC if applicable

2. **Clarify scope:**
   - Which phase/tasks are in scope
   - Any uncertainties to resolve first

3. **Execute with sub-agents:**
   - Use Explore agent for codebase understanding
   - Use focused agents for specific file implementations
   - Keep main thread for coordination and decisions

4. **End with updates:**
   - Update CLAUDE.md with completed work
   - Note any deferred items or blockers
   - Commit with clear message

---

## Clean Break from Logosphere

This is a **new project**, not a migration.

**Logosphere:** archived
- Tag as `logosphere-final`
- Sessions remain as-is, human-readable (YAML + embeddings)
- Tooling preserved in git history
- Reference material, not runtime dependency

**Substrate:** new project
- Optimal format from day one
- No legacy handling, no format detection, no deprecation
- Clean boundary: this is where the experiment begins fresh

**What carries over (as reference):**
- Core algorithms (clustering, HDBSCAN logic) — reimplemented cleanly
- TUI patterns (views, panels, modals) — adapted to new structure
- Design learnings — embedded in architecture decisions

**What does NOT carry over:**
- Session data compatibility — old sessions don't load
- Storage format constraints — free to redesign
- Field name legacy — no aliases needed
- Migration code — none exists

---

## Reference Documents

- `docs/architecture-greenfield.md` — full architecture rationale
- `docs/rfc-narrowing-draft-function.md` — draft/signal redesign
- `docs/rfc-recall-sampling.md` — recall sampling mechanism
- `docs/system_prompt_v1.5.md` — current protocol (for reference)

---

## Success Criteria

Phase complete when:
1. All tasks checked off
2. Tests pass
3. CLAUDE.md updated and accurate
4. Can demonstrate the phase's outputs

Project complete when:
1. `subs` launches and runs iterations
2. New session can be created, iterated, persisted
3. All RFCs implemented or explicitly deferred
4. CLAUDE.md is the complete, accurate source of truth

---

*This plan coordinates work across multiple sessions. Each session picks up where the last left off, guided by CLAUDE.md state.*
