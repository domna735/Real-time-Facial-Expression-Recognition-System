# Results Table Templates — Domain Shift / Field Transfer

Last updated: 2026-02-05

Use these templates to keep comparisons consistent across runs.

---

## Template A — Offline (eval-only gate)

| Method | Base ckpt | Adaptation scope | Target buffer | Steps | LR | Gate macro-F1 | Gate minority-F1 | TS ECE | TS NLL | Gate pass? | Artifact path |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Baseline | <path> | none | n/a | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | baseline | outputs/evals/students/<run>/reliabilitymetrics.json |

Example baselines (2026-02-05; macro-F1/TS metrics copied from artifacts):

| Method | Base ckpt | Adaptation scope | Target buffer | Steps | LR | Gate macro-F1 | Gate minority-F1 | TS ECE | TS NLL | Gate pass? | Artifact path |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| KD baseline | outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308/best.pt | none | n/a | 0 | 0 | 0.4385411 | (derive) | 0.0217606 | 1.2961859 | baseline | outputs/evals/students/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308__classification_manifest_eval_only__test__20260205_163424/reliabilitymetrics.json |
| KD + LP-loss | outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653/best.pt | none | n/a | 0 | 0 | 0.4411229 | (derive) | 0.0374865 | 1.2773255 | baseline | outputs/evals/students/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653__classification_manifest_eval_only__test__20260205_171945/reliabilitymetrics.json |
| TENT | <path> | norm affine | webcam buffer | 200 | 1e-4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | TODO | outputs/evals/students/<run>/reliabilitymetrics.json |
| SAR-lite | <path> | norm affine | webcam buffer | 200 | 1e-4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | TODO | outputs/evals/students/<run>/reliabilitymetrics.json |

---

## Template B — Offline (ExpW target benchmark)

| Method | Base ckpt | Adaptation scope | Steps | LR | ExpW acc | ExpW macro-F1 | ExpW minority-F1 | Raw ECE | Raw NLL | TS ECE | TS NLL | Artifact path |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | <path> | none | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | outputs/evals/students/<run>/reliabilitymetrics.json |

Example baselines (2026-02-05; copied from artifacts):

| Method | Base ckpt | Adaptation scope | Steps | LR | ExpW acc | ExpW macro-F1 | ExpW minority-F1 | Raw ECE | Raw NLL | TS ECE | TS NLL | Artifact path |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| KD baseline | outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308/best.pt | none | 0 | 0 | 0.6311145 | 0.4595847 | (derive) | 0.2324980 | 1.7437216 | 0.0276567 | 1.0635237 | outputs/evals/students/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308__expw_full_manifest__test__20260205_163538/reliabilitymetrics.json |
| KD + LP-loss | outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653/best.pt | none | 0 | 0 | 0.6356902 | 0.4583109 | (derive) | 0.2286580 | 1.5911980 | 0.0197645 | 1.0421315 | outputs/evals/students/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653__expw_full_manifest__test__20260205_172039/reliabilitymetrics.json |
| TENT | <path> | norm affine | 200 | 1e-4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | outputs/evals/students/<run>/reliabilitymetrics.json |
| SAR-lite | <path> | norm affine | 200 | 1e-4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | outputs/evals/students/<run>/reliabilitymetrics.json |

---

## Template C — Live webcam scoring (same labeled session re-scored)

| Method | Base ckpt | Stabilizer policy | Raw acc | Raw macro-F1 (present) | Smoothed acc | Smoothed macro-F1 (present) | Smoothed minority-F1 (present) | Flips/min | Notes | Artifact path |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Baseline | <path> | <ema/vote/hyst config> | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 | same labels | demo/outputs/<run_stamp>/score_results.json |
| Adapted | <path> | <same config> | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 | same labels | demo/outputs/<run_stamp>/score_results.json |

---

## Template D — Streaming stability / safety events

| Method | Collapse detected? | Recovery/reset events | Rollback events | Gate pass rate | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| TENT | TODO | 0 | 0 | 0.00 | |
| SAR-lite | TODO | 0 | 0 | 0.00 | |

---

## Template E — One-line summary for the final report

| Best method | Why it wins | Where it wins | What it costs | What it risks |
| --- | --- | --- | --- | --- |
| <method> | <minority-F1↑ + stable flips/min> | ExpW + webcam | <extra compute + complexity> | <drift/collapse mitigated by gate> |
