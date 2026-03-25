import re

file_path = r"c:\Real-time-Facial-Expression-Recognition-System_v2_restart\research\final report\final report version 3.md"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Replacements for figures
replacements = {
    r"!\[System Architecture Pipeline\]\(\.\./figures/fig0_pipeline_architecture\.png\)": r"![System Architecture Pipeline](../figures/fig0_pipeline_architecture.png)\n*Figure 3.1: High-level pipeline architecture mapping multi-source raw data to a real-time deployment loop.*",
    r"!\[Dataset Imbalance Distribution\]\(\.\./figures/fig0_data_imbalance\.png\)": r"![Dataset Imbalance Distribution](../figures/fig0_data_imbalance.png)\n*Figure 4.1: Extreme class imbalance present in the HQ training manifest.*",
    r"!\[Training Curves - Cross Entropy vs Distillation\]\(\.\./figures/fig5_training_curves_ce\.png\)": r"![Training Curves - Cross Entropy vs Distillation](../figures/fig5_training_curves_ce.png)\n*Figure 4.2: Student training progression comparing standard Cross-Entropy against Distillation targets.*",
    r"!\[Real-Time Hysteresis Jitter Plot\]\(\.\./figures/fig11_hysteresis_jitter\.png\)": r"![Real-Time Hysteresis Jitter Plot](../figures/fig11_hysteresis_jitter.png)\n*Figure 4.3: Temporal stabilisation effect suppressing jitter during continuous inference.*",
    r"!\[Teacher Per-class F1 Scores\]\(\.\./figures/fig2_teacher_perclass_f1\.png\)": r"![Teacher Per-class F1 Scores](../figures/fig2_teacher_perclass_f1.png)\n*Figure 5.1: Teacher ensemble performance across specific emotion classes (Stage-A validation).*",
    r"!\[Student Per-class F1 Scores\]\(\.\./figures/fig1_student_perclass_f1\.png\)": r"![Student Per-class F1 Scores](../figures/fig1_student_perclass_f1.png)\n*Figure 5.2: Student bottleneck capacity shown via per-class F1 drops.*",
    r"!\[Calibration Comparison - Reliability Diagrams\]\(\.\./figures/fig3_calibration_comparison\.png\)": r"![Calibration Comparison - Reliability Diagrams](../figures/fig3_calibration_comparison.png)\n*Figure 5.3: Reliability diagrams demonstrating improved probabilistic calibration via KD/DKD.*",
    r"!\[Confusion Matrix - Webcam Domain\]\(\.\./figures/fig4_confusion_matrix_webcam\.png\)": r"![Confusion Matrix - Webcam Domain](../figures/fig4_confusion_matrix_webcam.png)\n*Figure 5.4: Live webcam domain confusion matrix illustrating severe degradation in minority classes.*",
    r"!\[Webcam Raw vs Smoothed Output\]\(\.\./figures/fig8_webcam_raw_vs_smoothed\.png\)": r"![Webcam Raw vs Smoothed Output](../figures/fig8_webcam_raw_vs_smoothed.png)\n*Figure 5.5: Live telemetry comparison between raw argmax and smoothed hysteresis predictions.*",
    r"!\[Adaptation A/B Comparison Failure\]\(\.\./figures/fig10_adaptation_ab\.png\)": r"![Adaptation A/B Comparison Failure](../figures/fig10_adaptation_ab.png)\n*Figure 5.6: A/B replay metrics visualization demonstrating the regression of the NR-1 adaptation candidate.*",
    r"!\[Cross-Dataset Macro-F1 Degradation\]\(\.\./figures/fig6_crossdataset_macro_f1\.png\)": r"![Cross-Dataset Macro-F1 Degradation](../figures/fig6_crossdataset_macro_f1.png)\n*Figure 5.7: Degradation of generalisation metrics when evaluating models on out-of-domain sources.*",
    r"!\[Teacher Domain Shift Validation\]\(\.\./figures/fig7_teacher_domain_shift\.png\)": r"![Teacher Domain Shift Validation](../figures/fig7_teacher_domain_shift.png)\n*Figure 5.8: Distributional shift mapped via Teacher confidence outputs across varying datasets.*",
    r"!\[FER2013 Benchmark Comparison\]\(\.\./figures/fig9_fer2013_official\.png\)": r"![FER2013 Benchmark Comparison](../figures/fig9_fer2013_official.png)\n*Figure 5.9: Comparative benchmarks against standard FER2013 leaderboards.*",
}

for old, new in replacements.items():
    text = re.sub(old, new, text)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Captions added successfully.")
