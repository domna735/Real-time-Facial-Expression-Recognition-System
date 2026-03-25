# Process Log - Week 1 of January 2026
This document captures the daily activities, decisions, and reflections during the first week of January 2026, focusing on reconstructing the facial expression recognition system as per the established plan.

## 2026-01-01 — NL(proto) “make it bite” + NegL gate stress test (KD 5ep)

Goal:
- Follow the “one-by-one” plan: (1) make NL actually apply across epochs, (2) make NegL apply meaningfully, then (3) only test synergy.

One-click command used:
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_3run_nl_negl_oneclick.ps1 -UseClahe -UseAmp`

Baseline reference (KD 5 epochs, NegL/NL off):
- `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119/`

Runs executed (all KD 5 epochs):

1) NL(proto, penultimate embed) with fixed threshold:
- Run dir: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20260101_084847/`
- Compare: `outputs/students/_compare_20260101_084847_kd5_nlproto_penultimate_fixed_thr0p05_vs_kd5.md`
- Key metrics (from compare): acc 0.721527, macro-F1 0.718989, TS ECE 0.030271, TS NLL 0.807121, minority-F1 0.686280
- NL applied_frac by epoch: [0.084308, 0.000258, 0.000110, 0.000033, 0.000014]

2) NL(proto, penultimate embed) with top-k gating (target frac=0.1):
- Run dir: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20260101_091806/`
- Compare: `outputs/students/_compare_20260101_084847_kd5_nlproto_penultimate_topk0p1_vs_kd5.md`
- Key metrics (from compare): acc 0.723015, macro-F1 0.718769, TS ECE 0.040034, TS NLL 0.809448, minority-F1 0.686940
- NL applied_frac by epoch: [0.109375, 0.109375, 0.109375, 0.109375, 0.109375]

3) NegL-only (entropy gate thr=0.4):
- Run dir: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20260101_094542/`
- Compare: `outputs/students/_compare_20260101_084847_kd5_negl_entropy_ent0p4_vs_kd5.md`
- Key metrics (from compare): acc 0.723899, macro-F1 0.720618, TS ECE 0.039708, TS NLL 0.829301, minority-F1 0.690973
- NegL applied_frac by epoch: [0.163261, 0.073691, 0.061450, 0.048292, 0.040523]

Interpretation (so far):
- Fixed-threshold NL(proto) still “fires early then dies”.
- Top-k gating successfully keeps NL active every epoch (by construction), but metrics did not improve in this short run.
- Lowering NegL entropy threshold to 0.4 makes NegL apply meaningfully (few % → ~4–16%), but improvements are not yet clear.

Planned next experiments:
- NL-only: try top-k with smaller target fraction (e.g., 0.05) and/or a small NL weight sweep.
- NegL-only: entropy threshold sweep around 0.4 (e.g., 0.3 and 0.5) and record applied_frac curves.
- Only then: run NL(top-k) + NegL(entropy gate) together (KD 5ep) for a clean synergy check.

## 2026-01-01 — Next-planned KD sweep + DKD resume/tooling fixes

Goal:
- Execute the “planned next experiments” as a consistent KD screening batch.

One-click command used:
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_nextplanned_nl_negl_oneclick.ps1 -UseClahe -UseAmp`

Baseline reference (KD 5 epochs, NegL/NL off):
- `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119/`

Runs executed (all KD 5 epochs):

1) NL-only: NL(proto, penultimate embed) with top-k gating (target frac=0.05, w=0.1)
- Run dir: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20260101_153900/`
- Compare: `outputs/students/_compare_20260101_153859_kd5_nlproto_penultimate_topk0p05_w0p1_vs_kd.md`
- Key metrics (from compare): acc 0.727759, macro-F1 0.725666, TS ECE 0.037482, TS NLL 0.797487, minority-F1 0.693276
- NL applied_frac by epoch (from `history.json` → `nl.applied_frac`): [0.0625, 0.0625, 0.0625, 0.0625, 0.0625]

2) NegL-only: entropy gate thr=0.3
- Run dir: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20260101_165108/`
- Compare: `outputs/students/_compare_20260101_153859_kd5_negl_entropy_ent0p3_vs_kd.md`
- Key metrics (from compare): acc 0.728177, macro-F1 0.726967, TS ECE 0.046010, TS NLL 0.827339, minority-F1 0.698288
- NegL applied_frac by epoch (from `history.json` → `negl.applied_frac`): [0.227703, 0.127009, 0.109995, 0.086042, 0.066261]

3) NegL-only: entropy gate thr=0.5
- Run dir: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20260101_171607/`
- Compare: `outputs/students/_compare_20260101_153859_kd5_negl_entropy_ent0p5_vs_kd.md`
- Key metrics (from compare): acc 0.726782, macro-F1 0.725032, TS ECE 0.044099, TS NLL 0.824008, minority-F1 0.690081
- NegL applied_frac by epoch (from `history.json` → `negl.applied_frac`): [0.113780, 0.046293, 0.038647, 0.029508, 0.021247]

4) Synergy: NL(top-k=0.05, w=0.1) + NegL(entropy gate thr=0.4)
- Run dir: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20260101_174040/`
- Compare: `outputs/students/_compare_20260101_153859_kd5_nlproto_topk0p05_plus_negl_entropy_ent0p4_vs_kd.md`
- Key metrics (from compare): acc 0.725155, macro-F1 0.722802, TS ECE 0.042345, TS NLL 0.795800, minority-F1 0.686232
- NL applied_frac by epoch: [0.054688, 0.054688, 0.054688, 0.054688, 0.054688]
- NegL applied_frac by epoch: [0.162841, 0.073639, 0.060271, 0.048092, 0.040303]

Interpretation (short-budget screening only):
- NL(top-k=0.05) stayed active across epochs as intended.
- NegL threshold sweep behaved as expected (lower thr -> higher applied_frac).
- No clear metric gain appeared in these KD-5ep settings; synergy run was worse than baseline on acc/macro-F1 and minority-F1.

DKD tooling fixes (to enable fair DKD screening):
- DKD resume from KD checkpoint hit an optimizer state mismatch; fixed by skipping optimizer/scaler restore when resuming across modes (KD -> DKD).
- DKD one-click no longer parses `Run stamp:` (host-only output); it locates the newest DKD output folder to run comparisons.

## 2026-01-01 — DKD next-planned one-click sweep (CLAHE+AMP)

Goal:
- Run the DKD version of the “next-planned” screening set (resume-from-KD baseline checkpoint) and produce compare markdowns.

One-click command used:
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_nextplanned_nl_negl_oneclick_dkd.ps1 -UseClahe -UseAmp -NegLEntropyThreshesCsv "0.3,0.5"`

Baseline reference (DKD):
- `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722/`

Compare markdowns produced (all vs baseline DKD):
- `outputs/students/_compare_20260101_204953_dkd5_nlproto_penultimate_topk0p05_w0p1_vs_dkd.md`
- `outputs/students/_compare_20260101_204953_dkd5_negl_entropy_ent0p3_vs_dkd.md`
- `outputs/students/_compare_20260101_204953_dkd5_negl_entropy_ent0p5_vs_dkd.md`
- `outputs/students/_compare_20260101_204953_dkd5_nlproto_topk0p05_plus_negl_entropy_ent0p4_vs_dkd.md`

Runs executed:

1) NL-only (proto, penultimate, top-k=0.05, w=0.1)
- Run dir: `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_204953/`
- Key metrics: acc 0.719807, macro-F1 0.717861, TS ECE 0.045183, TS NLL 0.844715, minority-F1 0.688264
- NL applied_frac by epoch: [0.0625, 0.0625, 0.0625, 0.0625, 0.0625]

2) NegL-only (entropy ent=0.3, w=0.05, ratio=0.5)
- Run dir: `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_212203/`
- Key metrics: acc 0.731479, macro-F1 0.730934, TS ECE 0.041676, TS NLL 0.812235, minority-F1 0.705310
- NegL applied_frac by epoch: [0.088500, 0.059899, 0.041239, 0.031594, 0.028544]

3) NegL-only (entropy ent=0.5, w=0.05, ratio=0.5)
- Run dir: `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_214949/`
- Key metrics: acc 0.730410, macro-F1 0.729865, TS ECE 0.035637, TS NLL 0.805373, minority-F1 0.703345
- NegL applied_frac by epoch: [0.035827, 0.022669, 0.013310, 0.008920, 0.007631]

4) Synergy (NL top-k=0.05, w=0.1 + NegL entropy ent=0.4)
- Run dir: `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_221602/`
- Key metrics: acc 0.733712, macro-F1 0.733798, Raw ECE 0.202779, Raw NLL 1.412536, TS ECE 0.037443, TS NLL 0.786831, minority-F1 0.701544
- NL applied_frac by epoch: [0.054688, 0.054688, 0.054688, 0.054688, 0.054688]
- NegL applied_frac by epoch: [0.056472, 0.037731, 0.025051, 0.018169, 0.014909]

Interpretation (DKD short sweep only):
- NL-only (top-k) is materially worse than the DKD baseline in this setting.
- NegL-only ent=0.3/0.5 did not improve the main metrics vs baseline.
- Synergy improves raw ECE/NLL but does not improve acc/macro-F1 or minority-F1 vs baseline.