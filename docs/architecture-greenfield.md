# Architecture: Greenfield Design

## Context

The current codebase evolved organically. This document captures what a clean redesign might look like, knowing the current scope and complexity. Not a migration plan - a north star for evaluating future changes.

## Core Domains

| Domain | Responsibility |
|--------|----------------|
| **Protocol** | I/O format - system_prompt & user_prompt construction, output parsing |
| **Session** | Workspace management - config, state, owns storage pools |
| **Inference** | LLM invocation - API calls, thin wrapper |
| **Storage** | Pure data persistence - thoughts, messages, drafts, users |
| **Sampling** | Selection logic - what the mind sees (recent, random, recall, limits) |
| **Clustering** | Semantic organization - embeddings, centroids, assignments |
| **Orchestration** | Runtime coordination - iteration loop, phases, metrics, conditions |
| **Interface** | User-facing - CLI, TUI, shared operations |

## Principles

| Principle | Application |
|-----------|-------------|
| **Single responsibility** | Each module does one thing well |
| **Dependency direction** | Lower layers don't know about higher layers |
| **Domain boundaries** | Clear separation, narrow interfaces between domains |
| **Composition** | Build from small pieces, avoid deep inheritance |
| **Explicit dependencies** | Inject what you need, no hidden globals |
| **Testable isolation** | Each piece can be tested alone |
| **Storage is pure** | Data layer has no business logic |
| **Interfaces are leaves** | Depend on everything, nothing depends on them |

## Proposed Structure

```
logos/
├── session/               # Workspace management
│   ├── session.py         # Session class - config, state, pool facade
│   ├── config.py          # SessionConfig dataclass
│   └── lock.py            # Session locking for CLI/TUI coordination
│
├── storage/               # Pure data pools (owned by Session)
│   ├── thoughts.py        # ThoughtPool - embeddings + metadata
│   ├── messages.py        # MessageStore - conversation history
│   ├── drafts.py          # DraftStore - draft responses
│   ├── users.py           # UserRegistry - users + presence state
│   └── io.py              # YAML/NPY persistence utilities
│
├── sampling/              # Selection logic (stateless, operates on pools)
│   ├── recent.py          # Deterministic recent thought selection
│   ├── random.py          # Random pool sampling
│   ├── recall.py          # Cluster-aware recall sampling
│   └── limits.py          # DisplayLimiter (chars/count)
│
├── clustering/            # Semantic organization
│   ├── embeddings.py      # Embedding client (OpenRouter)
│   ├── algorithm.py       # HDBSCAN, centroid matching
│   └── manager.py         # ClusterManager - state, assignments
│
├── protocol/              # Mind I/O format (definition + implementation)
│   ├── system_prompt.py   # Compose system prompt from sections
│   ├── user_prompt.py     # Compose input YAML from sections
│   ├── parser.py          # Parse output using section knowledge
│   ├── types.py           # Shared types (MindOutput, Age, etc.)
│   │
│   └── sections/          # Per-section: definition + format + examples + parse
│       ├── meta.py
│       ├── thoughts.py
│       ├── dialogue.py
│       ├── drafts.py
│       ├── users.py
│       └── orientation.py
│
├── inference/             # LLM invocation (thin)
│   └── client.py          # API wrapper, invoke_mind()
│
├── orchestration/         # Runtime coordination
│   ├── runner.py          # Iteration loop (step, run)
│   ├── lifecycle.py       # Phase transitions (processing/drafting)
│   ├── metrics.py         # Noise ratio, stability, production rate
│   └── conditions.py      # Start/stop condition evaluation
│
├── interfaces/            # User-facing (leaves of dependency tree)
│   ├── ops.py             # Shared operations (accept, send, etc.)
│   ├── cli/
│   │   ├── commands.py    # CLI command handlers
│   │   └── main.py        # CLI entry logic
│   └── tui/
│       ├── app.py         # Textual app
│       ├── views/         # Main view components
│       ├── panels/        # Sidebar panels
│       └── modals/        # Dialog modals
│
└── scripts/
    └── logos.py           # Single entry point (CLI + TUI)
```

## Dependency Flow

```
                    interfaces (cli, tui)
                           ↓
                    orchestration
                    (runner, lifecycle, metrics)
                           ↓
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
    sampling           protocol          clustering
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
                       session
                           ↓
                       storage
                           ↓
                      inference
```

**Key relationships:**
- Orchestration operates *on* a Session
- Session owns storage pool instances
- Sampling/Protocol/Clustering read from storage via Session
- Inference is independent (just API calls)
- Interfaces are leaves (depend on everything, nothing depends on them)

## Session as First-Class Concept

Session is more than storage. It's the **unit of work**:

| Aspect | Current Location | Greenfield Location |
|--------|------------------|---------------------|
| Config (model, limits) | SessionV2 | session/config.py |
| State (iteration, is_drafting) | SessionV2 + derived | session/session.py |
| Pool facade | SessionV2 | session/session.py |
| Locking | core/lock.py | session/lock.py |
| Directory structure | SessionV2 | session/session.py |

Session sits between orchestration and storage:
- Orchestration operates on a Session
- Session provides access to storage pools
- Storage pools are pure data, no business logic

## Protocol: Definition + Implementation as One

The protocol is a coherent unit with three synchronized aspects:

| Aspect | File | Role |
|--------|------|------|
| **Definition** | `system_prompt.py` | Tells the mind what format to expect and produce |
| **Input** | `user_prompt.py` | Implements the format the definition describes |
| **Output** | `parser.py` | Reads responses according to the definition |

Changes must be coordinated: add a section to input → update system prompt explanation → update parser if output affected.

Each section module (`sections/*.py`) is self-contained:

```python
# sections/thoughts.py

DEFINITION = """
# THINKING POOL
Transmitted thoughts persist for a time and compete for attention...
"""

EXAMPLES = """
thinking_pool:
  recent:
    - |  # age: 1
      latest thought
  sampled:
    - |  # age: 42, cluster: {id: 3, size: 8}
      older thought surfaced by sampling
"""

def format_input(recent: list[Thought], sampled: list[Thought], current_iter: int) -> str:
    """Build thinking_pool section of input YAML."""
    ...

def parse_output(yaml_block: dict) -> list[str]:
    """Extract thoughts from mind output."""
    ...
```

Top-level files are thin coordinators that import from sections and assemble complete artifacts.

## Orchestration: Coordination Only

Runner doesn't own data or format - it coordinates:

```python
# orchestration/runner.py
class Runner:
    def step(self, session: Session) -> IterationResult:
        # 1. Check phase, evaluate conditions
        phase = self.lifecycle.current_phase(session, self.metrics)

        # 2. Sample thoughts (via sampling module)
        recent = select_recent(session.thoughts, ...)
        sampled = sample_random(session.thoughts, ...)

        # 3. Build input (via protocol module)
        input_yaml = compose_input(...)

        # 4. Invoke LLM (via inference module)
        output = invoke_mind(input_yaml, ...)

        # 5. Process output, update session
        ...

        # 6. Update metrics, check stop conditions
        ...
```

## Open Questions

**Entry point unification:**
- `logos` (no args) → interactive mode (TUI)
- `logos <subcommand>` → one-shot command, reuse output rendering where possible
- Single entry point, consistent experience
**Optional features:**: Session config → protocols
**Session config**
- Session config is **absolute truth**
- No runtime overrides (unless first written to session file)
- No environment variables except:
  - API keys (secrets)
  - Session path selection (which workspace to load)

---

*This document captures architectural direction. Implementation decisions should reference these principles but adapt to practical constraints.*
