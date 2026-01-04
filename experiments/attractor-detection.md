# Attractor Detection via Embedding Space

## Observation

In extended runs (100+ rounds), pool content may converge toward specific patterns. One pattern observed in prelimiary work: repetitive meta-commentary about AI assistant role and request for concrete tasks.

## Proposed Approach

**Binary classification in embedding space:**

1. Extract messages from identified attractor region (e.g., rounds 100-200 of a collapsed run)
2. Generate embeddings for these messages
3. Define attractor region as centroid + radius in embedding space
4. Classify any message as `in_attractor` or `out_of_attractor` based on distance to centroid

**Potential applications:**
- Measure % of pool in attractor zone over time
- Filter attractor messages from sampling (adversarial exclusion)
- Track multiple known attractors
- Detect novel attractor formation

## Requirements

**Data:**
- Existing experiment logs with identified attractor behavior
- Embedding model API (e.g., via OpenRouter)

**Tooling:**
- Read messages from experiment.jsonl or novel_memes.yaml
- Batch embed messages
- Compute centroid and threshold from reference set
- Classify new messages against known attractors

**Outputs:**
- Attractor definitions (centroid vector + metadata)
- Classification results per message/round
- Time-series data: attractor% over rounds

## Open Questions

- What distance metric? (cosine, euclidean)
- How to set threshold? (std deviation multiple, percentile)
- Multiple attractors: independent classification or mutual exclusion?
- How to detect NEW attractors not in reference set?
- Does filtering create adversarial pressure or just shift to different boring attractor?

## Status

Proposed. Not implemented. Requires baseline experiments with attractor behavior to validate approach.
