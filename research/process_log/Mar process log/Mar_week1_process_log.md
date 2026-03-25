# Process Log - Week 1 of March 2026
This document captures the daily activities, decisions, and reflections during the first week of March 2026, focusing on reconstructing the facial expression recognition system as per the established plan.

Follow the template below to document your activities, decisions, and reflections for each day of the week.

## YYYY-MM-DD | Short Title
Intent:
Action:
Result:
Decision / Interpretation:
Next:

## 2026-03-07 | Final report academic polish — two-round structural + quality pass
Intent:
- Raise the final report (`research/final report/final report version 2.md`) from B+/A− to solid A− quality by addressing structural, tonal, and cross-reference issues identified through a full-document assessment.
- Remove all internal-documentation artifacts (meta blocks, planning language, informal labels) so the report reads as a submission-ready FYP document.

Action:
- Performed a full read-through assessment of the ~1,580-line final report and identified 14 improvement areas ranked by impact; selected the top 8 for a first pass.

- **Round 1 — 8 targeted improvements (all applied and verified):**
	1. **Moved misplaced sections:** Relocated Sections 9.3.2–9.3.4 (offline safety gate, deployment-facing stability, real-time smoothing analysis) into Section 4 as 4.11–4.13 with all cross-references updated — these were results content stranded in the conclusion.
	2. **Removed Section 4.9 redundancy:** Deleted the duplicate "what we tried / what it means" sub-block (old 4.9.1) and streamlined 4.9.2, stripping planning-style blocks.
	3. **Cleaned internal-documentation tone:** Removed 12 instances of meta labels ("Interpretation note:", "Reproducibility rule:", "Working hypothesis:", etc.) and rewrote each as standard academic prose.
	4. **Fixed top-level section numbering:** Renumbered Section 3 subsections from 3.0–3.7 → 3.1–3.8 and Section 6 from 6.0–6.2 → 6.1–6.3.
	5. **Formal institution name:** Replaced all "HKpolyU" occurrences with "The Hong Kong Polytechnic University (PolyU)".
	6. **Added ethics paragraph:** Inserted Section 6.4 (Ethical considerations) covering consent, bias, and data governance.
	7. **Expanded Table of Contents:** Added subsection-level entries for all major sections.
	8. **Flagged FPS benchmark absence:** Added a formal limitation statement in Section 5.2 noting that FPS was not benchmarked under controlled conditions.

- Re-read the full report post–Round 1 and identified 8 remaining issues (cascaded numbering, dead cross-refs, meta content, informal titles).

- **Round 2 — 8 remaining fixes (all applied and verified):**
	1. **Cascaded subsection numbering:** Updated child headings that were not renumbered in Round 1 (3.2.1→3.3.1, 3.6.1→3.7.1, 3.6.2→3.7.2, 3.7.1→3.8.1, 6.0.x→6.1.x, 6.1.x→6.2.x — 10 headings total).
	2. **Dead cross-references:** Replaced 2 occurrences of "Section 4.9.2" (removed subsection) with "Section 4.9".
	3. **Removed Appendix A.6:** Deleted the "MathJax Compatibility Checklist" appendix (internal meta content, not suitable for submission).
	4. **Figure numbering:** Updated "Figure 3.0" → "Figure 3.1" to match the renumbered section.
	5. **Removed internal label:** Rewrote the "Supervisor clarification:" paragraph as standard academic prose explaining the system's deployment-first objective.
	6. **Removed notation rule block:** Deleted the "Notation rule used in this final report (for HTML/MathJax and docx conversion)" meta block between Sections 2.9 and 3.
	7. **Formal section titles:** Renamed "Discussion refinement (what results mean, and what they do not)" → "Discussion of key findings" and "Analytical comparison vs papers (trade-off analysis, not 'winning')" → "Analytical comparison with published results". Updated ToC to match.
	8. **Tightened Section 1.2:** Condensed the 8-line "Relationship to Interim Report" section into a single concise paragraph.

- All edits performed via PowerShell `[System.IO.File]::ReadAllText/WriteAllText` due to Unicode characters (smart quotes U+201C/U+201D, em-dashes, en-dashes) preventing standard editor replace operations.

Result:
- Final report is now structurally clean:
	- All section/subsection numbers cascade correctly (3.x.y, 6.x.y).
	- No dead cross-references remain.
	- No internal-documentation artifacts (meta blocks, planning labels, informal titles) survive.
	- ToC is complete and consistent with section headings.
	- Ethics section present (Section 6.4).
- Report reduced from ~1,640 lines to ~1,560 lines through redundancy removal and tightening.
- No numeric results, artifact paths, or evidence claims were altered — all changes were structural/tonal.

Decision / Interpretation:
- The report is now at submission-ready quality for a FYP final report. Remaining optional work would be cosmetic (citation formatting, minor phrasing) rather than structural.
- The two-round approach (broad pass → targeted residual pass) was effective for catching cascaded issues that only become visible after the first round of fixes.

Next:
- Optional: a final cosmetic pass (tense consistency, citation style unification) if time permits before submission.
- Prepare submission artifacts (PDF export, any required cover sheets).

## 2026-03-08 | Final report quality pass — 12-point improvement sweep + cut analysis

Intent:
- Systematically implement 12 quality improvements identified from a high-standards critique of the report (`research/final report/final report cut pass version.md`, starting at ~1,211 lines).
- After all improvements are applied, perform a full re-read to identify parts that can be cut (redundant, low-value, or non-submission-ready content) and areas that still need further improvement.

Action:

**12-point improvement sweep (all applied and verified):**

1. **Differentiated Abstract vs Executive Summary.** The Executive Summary previously duplicated the Abstract's bullet-point findings. Replaced with a structured Deliverables table (6 rows: data pipeline → dual-gate protocol) and a Headline Results table (6 metrics with section source pointers), plus a single "Key insight" sentence. The Abstract and Executive Summary now serve distinct purposes.

2. **Added 6 recent FER papers to the literature review.** Sections 2.1, 2.3, and 2.8 now reference SCN [23], RAN [24], POSTER V2 [25], DAN [26], EAC [27], and MA-Net [28]. Section 2.1 discusses noise-handling approaches (SCN, EAC); Section 2.3 covers FER-specific architectures (RAN, DAN, MA-Net, POSTER V2) with SOTA numbers; Section 2.8 (Research Gap) now explicitly names these methods when arguing that no existing work provides dual-evaluation. All 6 references added to Section 10.

3. **Expanded paper comparison table.** Added Table 4.13-D: RAF-DB accuracy landscape with 8 methods (SCN 87.03%, RAN 86.90%, LP-loss 84.13%, MA-Net 88.42%, EAC 89.99%, DAN 89.70%, POSTER V2 92.21%, Ours 86.28%). Shows MobileNetV3 (5.4M params) is competitive with ResNet-18-class methods (11M) and contextualises the ~6pp gap to POSTER V2 (100M params).

4. **Cleaned methodology section.** Condensed Section 3.3.1 dataset provenance (removed verbose per-file bullet lists) and Section 3.4 teacher training (removed inline source filter counts like `{ferplus: 138,526, ...}`). Body now states summary facts; details remain in artifacts.

5. **Toned down dual-gate contribution language.** Replaced all instances of "dual-gate evaluation framework" → "dual-gate evaluation protocol" (6 occurrences), "central methodological contribution" → "key engineering contribution" (2 occurrences in Abstract and Section 9.1), and "protocol-aware comparison framework" → "protocol-aware comparison methodology". Verified zero remaining "framework" usages (except one legitimate reference to `field_transfer_framework.md` filename, which is correct).

6. **Reframed discussion to lead with positives.** Added a "What this project delivers" paragraph at the start of Section 6.1 highlighting: competitive RAF-DB accuracy, good calibration, full reproducibility, and working real-time demo. Added "Single-seed experiments" limitation to Section 6.1.5 acknowledging the need for repeated runs.

7. **Fixed academic register / lab-note passages.** 8+ informal passages cleaned:
   - "Working hypotheses for the hard-gate gap (to be tested, not assumed)" → "Hypothesised causes of the hard-gate gap"
   - "This submission-cut version reports..." → "This section reports..."
   - "Submission-cut interpretation" → "Interpretation"
   - "Repro note" → "Note"
   - "Submission-cut pointer list (minimal)" → "Condensed pointer list"
   - "Submission-cut evidence index" → "Evidence index"
   - "A first conservative Self-Learning + manifest-driven NegL A/B attempt was executed on 2026-02-21" → cleaner phrasing
   - "This subsection records the first short-budget screening results" → "This subsection reports the initial short-budget screening results"
   - "rather than claims" → "rather than claiming"

8. **Rounded decimal precision in all tables.** All tables with 5–7 decimal place values rounded to 3 d.p. Affected tables: 4.2-1 (teacher metrics), 4.2-3 (hard gates), 4.3-1 (ensemble), 4.4 (student CE/KD/DKD), 4.6 (webcam metrics), 4.7-1 (safety gate), 4.8-1 (ExpW), 4.11-A/B1/B2 (LP-loss), 4.13-B (FER2013 official). Verified no 5+ digit decimals remain.

9. **Expanded ethical considerations.** Added two new paragraphs to Section 6.4:
   - "Cultural validity of emotion categories" — notes Ekman's framework is contested; references Barrett et al. (2019) on non-universal emotion-face mappings; frames system outputs as learned statistical associations.
   - "Institutional ethics" — states project followed university undergraduate project guidelines; no new human-subjects data collected; webcam testing was author-only.

10. **Condensed timeline section.** Tightened Section 7 table cells: removed verbose descriptions (e.g., "ArcFace-style margins; ensemble selection and softlabel export" → "ensemble selection; softlabel export"); standardised date formatting ("Aug–Oct 2025" vs "Aug - Oct 2025"); "(planned)" removed from Apr 2026 row.

11. **Strengthened FPS limitation.** Rewrote Section 5.2 limitation paragraph to lead with the significance ("This gap is significant for a project titled 'Real-time' FER"), added a concrete 3-step benchmark procedure (replay 60s clip → compute median/p95 latency → report sustained FPS), and added GFLOPs context (5.4M parameters, ~0.22 GFLOPs).

12. **Final verification.** Confirmed: section numbering 1–11 cascades correctly; all [23]–[28] references cited in body and listed in Section 10; no stale "submission-cut" / "framework" / "central methodological" language remains; no 5+ digit decimal values survive.

Result:
- Report now at 1,257 lines (up from ~1,211 due to added substantive content: comparison table, lit review paragraphs, ethical considerations, benchmarking protocol).
- All 12 improvements verified with targeted grep searches.
- Report quality improved from B+/A− to solid A− / borderline A.

**Full re-read analysis — areas for further improvement and potential cuts:**

After the 12-point sweep, a full re-read identified the following remaining issues, organised as (a) content that could be cut to tighten the report, and (b) areas that would further strengthen it if addressed.

### Potential cuts (redundant or low-value content)

| # | Section | Lines (approx.) | Issue | Recommendation |
|---|---------|-----------------|-------|----------------|
| C1 | 4.6.1 (Qualitative checkpoint preference) | ~5 | Informal observation already discussed more formally in Section 5.3. Redundant. | **Cut.** Remove Section 4.6.1 entirely; Section 5.3 already covers the same observation with more analysis. |
| C2 | 4.8 (ExpW cross-dataset gate) | ~25 | Shows only 2 DKD checkpoints; thin standalone evidence. The ExpW gate result is already captured in the consolidated cross-gate comparison (4.4.1) and the LP-loss gate tables (4.11-B2). | **Merge.** Move the key number (DKD best macro-F1 0.460 on ExpW) into Section 4.4.1 as a single sentence and cut the standalone subsection. |
| C3 | 4.13-B "Additional evidence" paragraph | ~5 | Describes a Kaggle FER2013 folder dataset (msambare) evaluation that is explicitly "not a strict match" and adds confusion vs the official-split table directly above it. | **Cut.** Remove the "Additional evidence" paragraph; the official-split table is the anchor. |
| C4 | 4.13-C (AffectNet comparison) | ~8 | Comparison is explicitly called "not appropriate" due to balanced-subset mismatch. Low value if the comparison cannot be made. | **Cut or reduce to a single sentence** noting the evaluation exists but is not comparable. |
| C5 | 4.9 Mermaid diagram | ~15 | A second flowchart for the domain-shift loop. The pipeline overview (Figure 3.1) already shows the adaptation→gate→promote flow. | **Cut.** The diagram repeats Figure 3.1's adaptation tail. The text description is sufficient. |
| C6 | A.0 (Interim report figure mapping) | ~3 | Tells the reader which interim figures map to which final-report sections. Not needed in a standalone submission. | **Cut.** Only relevant to readers cross-referencing the interim report. |
| C7 | 6.3 (FYP requirements checklist) | ~20 | Useful for supervisor sign-off but reads as internal documentation in a polished academic report. | **Move to appendix** (e.g., A.6) rather than sitting in the Discussion. |

Estimated savings: ~80–100 lines if all cuts applied.

### Areas needing further improvement

| # | Area | Issue | Recommendation |
|---|------|-------|----------------|
| I1 | **FPS benchmark (Section 5.2)** | Still listed as "Not yet measured." This is the #1 gap for a project titled "Real-time FER." | **Priority 1.** Run the timed demo and fill in the measured FPS value. The pipeline already logs timestamps. |
| I2 | **Barrett et al. (2019) reference** | Cited in Section 6.4 (ethical considerations) but not listed in the reference section. | Add as [29] in Section 10. |
| I3 | **ToC vs heading mismatch** | ToC line says "2.8 Synthesis" but the actual heading is "2.8 Synthesis and Research Gap." | Update the ToC entry to match. |
| I4 | **Key phrase repetition** | "Offline non-regression is necessary but insufficient for deployment improvement" appears 4 times (Abstract, Section 4.10.2, Section 6.1.3, Section 9.1). Deliberate emphasis is acceptable, but 4× may read as copy-paste. | Vary the phrasing in at least 2 of the 4 instances. |
| I5 | **Figure file existence** | All `![...](figures/fig*.png)` references should be checked against actual files in `research/final report/figures/`. | Verify all 10 figure paths resolve. |
| I6 | **Per-class F1 table (4.2-2, 4.4)** | Still at 4 d.p. while the main metrics tables were rounded to 3 d.p. Inconsistent precision. | Round to 3 d.p. for consistency, or add a note explaining why per-class F1 uses 4 d.p. |

Decision / Interpretation:
- The 12-point quality sweep addresses all the high-impact structural and tonal issues. The report is now at submission-ready quality.
- The remaining cuts (C1–C7) would tighten the report by ~80–100 lines without losing any core argument. These are recommended but not urgent.
- The remaining improvements (I1–I6) are a mix of quick fixes (I2, I3, I6: <5 min each) and one significant task (I1: running the FPS benchmark). I1 is the single highest-impact remaining improvement because the project title includes "Real-time."
- The Barrett reference (I2) and ToC mismatch (I3) should be fixed before any submission.

Next:
- Decide whether to apply the cuts (C1–C7) — these are optional but would give a cleaner, tighter report.
- Fix the quick issues: add Barrett [29] reference, fix ToC 2.8 entry, optionally round per-class F1 tables.
- **Priority task:** run the FPS/latency benchmark using the existing demo pipeline and fill in the "Not yet measured" cell in Section 5.2.

## 2026-03-09 | Final report supervisor-led revision pass — claim calibration, live-run evidence, and final check
Intent:
- Update the dissertation-style final report (`research/final report/final report cut pass version.md`) using a supervisor-style critique rather than a pure proofreading pass.
- Tighten claim scope, remove any remaining overstatement, and align all "real-time" wording with the actual evidence available: successful live webcam operation plus saved runtime logs, not a formally benchmarked FPS target.
- Perform a final consistency check so the report is submission-ready in content, with only non-blocking markdown-to-DOCX issues left.

Action:
- Performed a full academic-supervisor review of the report and converted the feedback into a concrete section-by-section revision checklist before editing.

- **Pass 1 — claim hierarchy and academic framing (applied and verified):**
	1. **Softened novelty / research-gap wording.** Replaced absolute statements with bounded phrasing such as "to the best of our knowledge" and reframed the contribution as a deployment-oriented evaluation protocol rather than an over-claimed novel framework.
	2. **Added claim hierarchy in the discussion.** Reorganised the Discussion so the report clearly distinguishes between methodological contribution, engineering delivery, and empirical findings.
	3. **Added single-seed caution.** Inserted an explicit limitation in the student-model comparison section noting that CE/KD/DKD differences are directional, not statistically definitive from single-seed runs.
	4. **Improved paper-comparison framing.** Rewrote the comparative-results section so it reads as bounded contextualisation against protocol-mismatched literature rather than leaderboard-style claiming.
	5. **Strengthened conclusion and future-work tone.** Reworded the closing sections so they emphasise evaluation discipline, deployment realism, and honest limitations rather than benchmark-style triumphalism.

- **Pass 2 — real-time wording aligned to actual evidence (applied and verified):**
	1. **Reframed "real-time" throughout the report.** Updated the Abstract, Executive Summary, Discussion, KPI section, Conclusion, and Timeline so "real-time" consistently means successful live webcam operation in an end-to-end pipeline, not a controlled throughput benchmark.
	2. **Clarified KPI limitation language.** Replaced wording that could imply missing functionality with wording that accurately says the system ran live, but formal FPS benchmarking was not the focus of the submitted evidence.
	3. **Preserved the project title's real-time framing honestly.** Kept the real-time language because the project does deliver a runnable live FER system, while explicitly separating this from stronger performance claims.

- **Pass 3 — final micro-edits and evidence strengthening (applied and verified):**
	1. **Added run-log evidence sentence to Section 5.2.** Stated explicitly that saved per-frame runtime logs and scoring artifacts show continuous full-session live webcam operation, not just offline replay.
	2. **Clarified the FER2013 result as a real weakness.** Added a sentence in the analytical comparison section stating that FER2013 should be interpreted as a genuine weakness of the present system rather than a near-SOTA result.
	3. **Incorporated cautious live timing evidence.** Preserved the approximate live-run timing estimate derived from an existing saved session (`demo/outputs/20260227_130315/per_frame.csv`: 8,148 frames over 732.66 s, approx. 11.1 FPS) while keeping the text clear that this is not a fully controlled benchmark.

- **Final verification:**
	1. Re-read the revised report sections around the comparison, KPI, discussion, and conclusion areas.
	2. Verified the presence of key inserted phrases such as "successful live webcam operation," "single-seed runs," "genuine weakness of the present system," and the live-session evidence wording.
	3. Checked diagnostics; only markdown-style issues remained (heading spacing, emphasis-as-heading, unlabeled fenced code block, bare URL), which are non-blocking because the report will be exported to DOCX.

Result:
- The report now presents a much stronger academic argument without overclaiming:
	- the main contribution is framed as a deployment-oriented / dual-gate evaluation protocol;
	- the system is described honestly as a working live webcam FER pipeline;
	- real-time language is retained but scoped to runnable live operation rather than hard FPS claims;
	- FER2013 is explicitly acknowledged as a weakness;
	- single-seed comparisons are properly caveated.
- Section 5.2 now contains stronger evidence wording linking the deployment claim to saved live-run artifacts rather than generic demo statements alone.
- Final status: content-ready for submission, with no major academic inconsistency remaining.

Decision / Interpretation:
- The main risk in the report was no longer missing content, but mismatch between claim strength and evidence strength. This pass fixed that root problem.
- The report is now materially stronger because it sounds like a disciplined FYP dissertation rather than a lab notebook or an over-optimistic benchmark paper.
- Further content edits would likely produce churn rather than meaningful improvement unless a supervisor requests specific changes.

Next:
- Convert to DOCX/PDF and perform an output-format proofread (figure numbering, table layout, appendix formatting, path readability).
- If needed, prepare viva-style notes for likely questions on the real-time claim, FER2013 weakness, single-seed limitation, and the dual-gate protocol contribution.
