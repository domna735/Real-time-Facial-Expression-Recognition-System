# Part 4 — Metrics + Acceptance Criteria

We keep the gates minimal and measurable.

## Primary offline metrics
- Macro-F1 (higher is better)
- Minority-F1 (track the lowest-3 classes by support; higher is better)
- NLL (lower is better)
- ECE (lower is better)

## Deployment-facing signals
- Flip-rate (label changes per minute) from demo CSV logs
- Confidence stability (variance of max probability)
- “Confident wrong” rate (high-confidence errors)

## Acceptance gates (initial)
- NegL pass:
  - ECE improves by ≥20% OR NLL improves by ≥10% vs baseline under same eval protocol.
- NL pass:
  - No minority-F1 degradation after epoch ~10 in long-run training.
- Combined pass:
  - Does not regress Macro-F1 by >1pp while improving ECE/NLL, OR improves Macro-F1 while keeping ECE stable.

## Non-goals (for this phase)
- No feature-level distillation or attention alignment yet.
- No new backbones yet.
- No changing teacher set while evaluating NL/NegL.
