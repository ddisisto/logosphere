# Baseline Experiment: Model Behavior Comparison

## Objective

Observe pool dynamics across extended rounds with single-model, one-shot API calls. Compare behavior across different inference models using identical pool mechanics and seed content.

## Template Configuration

Source: `experiments/_baseline/`

**Fixed parameters (all runs):**
- N_MINDS: 1 (single model, multiple rounds)
- K_SAMPLES: 10 (sample 10 messages per round)
- M_ACTIVE_POOL: 200 (FIFO tail sampling window)
- MAX_ROUNDS: 200 (extended run to observe long-term dynamics)
- TOKEN_LIMIT: 8000 (allow large context)
- System prompt: Simplified format (no termination requirement)
- Seed pool: 28 messages (dense memetic content about replication, transmission, cooperation)

**Variable parameter:**
- MODEL: Different inference models

## Experiment Runs

### Primary Models (Free Tier)

```bash
# Run 1: Prime Intellect
python run.py baseline-intellect-3 --template _baseline --model prime-intellect/intellect-3

# Run 2: Xiaomi MIMO
python run.py baseline-mimo-flash --template _baseline --model xiaomi/mimo-v2-flash:free

# Run 3: NVIDIA Nemotron
python run.py baseline-nemotron --template _baseline --model nvidia/nemotron-3-nano-30b-a3b:free
```

### Validation Models (Run if needed)

```bash
# Run 4: Claude Sonnet (validation only)
python run.py baseline-sonnet --template _baseline --model anthropic/claude-sonnet-4.5

# Run 5: Claude Haiku (validation only)
python run.py baseline-haiku --template _baseline --model anthropic/claude-haiku-4.5
```

## Execution Status

- [ ] baseline-intellect-3
- [ ] baseline-mimo-flash
- [ ] baseline-nemotron
- [ ] baseline-sonnet (if needed)
- [ ] baseline-haiku (if needed)

## Hypothesis

**Null hypothesis:** Different models will show different response patterns to the same pool content and sampling conditions.

No specific predictions about:
- Convergence vs divergence
- Attractor states
- Output volume patterns

This is a baseline observation run to establish model behavior profiles.

## Observations to Track

**Per-run metrics (from logs):**
- Messages per round (output volume over time)
- Pool growth rate
- Token usage per round
- Rounds with zero output

**Post-run analysis (from novel_memes.yaml):**
- Content diversity over time
- Message length distribution
- Convergent patterns or repeated themes
- Comparison across models

## Analysis Workflow

1. Run experiment: `python run.py <name> --template _baseline --model <model-id>`
2. Auto-generates: `experiments/<name>/novel_memes.yaml`
3. Validation: Quick check of experiment.jsonl and novel_memes.yaml
4. Cross-model comparison: After all runs complete

## Notes

- Seed content: Dense philosophical memes about replication, transmission, base imperatives (Scott Alexander's "Goddess" framework)
- No filtering, no intervention during runs
- Let pool dynamics run to MAX_ROUNDS and observe emergent patterns
- Each run is self-contained and reproducible
