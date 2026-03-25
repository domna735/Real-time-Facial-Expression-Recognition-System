# Domain Shift Self-Learning + NegL — Index

This folder is the working notebook for improving real-time FER under webcam domain shift using a **safe adaptation loop**.

Primary source (full plan):

- [plan of self-learning + negative learning for domain shift.md](plan%20of%20self-learning%20%2B%20negative%20learning%20for%20domain%20shift.md)

## Documents (do these in order)

1. [01_problem_map.md](01_problem_map.md) — what “domain shift” means here + failure modes
2. [02_experiment_framework.md](02_experiment_framework.md) — hypotheses + ablations + evidence checklist
3. [03_implementation_steps.md](03_implementation_steps.md) — concrete code wiring tasks for Self-Learning + NegL
4. [04_metrics_acceptance.md](04_metrics_acceptance.md) — pass/fail gates (webcam-mini + offline eval-only + stability)
5. [05_commands_checklist.md](05_commands_checklist.md) — one-copy command recipes (record → score → buffer → tune → eval)
6. [06_assumption_check_and_next_steps_2026-01-28.md](06_assumption_check_and_next_steps_2026-01-28.md) — current status + immediate next actions

## Report

- [domain shift improvement via Self-Learning + Negative Learning report/domain shift improvement via Self-Learning + Negative Learning report.md](domain%20shift%20improvement%20via%20Self-Learning%20%2B%20Negative%20Learning%20report/domain%20shift%20improvement%20via%20Self-Learning%20%2B%20Negative%20Learning%20report.md) — rolling write-up for supervisor/report.

## Feb 2026 update

- Added evidence-backed KD baseline vs KD+LP gate results (eval-only + ExpW) to the rolling report and evaluation plan.

## Config

- [neglconfig.json](neglconfig.json) — thresholds/weights for planned NegL wiring (domain-shift version)
- [neglrules.md](neglrules.md) — “rules of engagement” so NegL stays safe
- [domain shift improvement via Self-Learning + Negative Learning study.md](domain%20shift%20improvement%20via%20Self-Learning%20%2B%20Negative%20Learning%20study.md) — study notes / references (domain-shift context)
