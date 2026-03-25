"""
Generate figures for the Final Report from stored JSON artifacts.

Outputs all figures to: research/final report/figures/
Each figure is saved as PNG (300 dpi) for report embedding.

Usage:
    python scripts/generate_report_figures.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "research" / "final report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
CLASS_SHORT = ["Ang", "Dis", "Fea", "Hap", "Sad", "Sur", "Neu"]

# color palette
COLORS = {
    "CE": "#2196F3",
    "KD": "#FF9800",
    "DKD": "#4CAF50",
    "RN18": "#E91E63",
    "B3": "#9C27B0",
    "CNXT": "#00BCD4",
}


def load_json(path):
    with open(ROOT / path, "r") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Per-class F1 comparison — Student CE vs KD vs DKD (HQ-train val)
# ═══════════════════════════════════════════════════════════════════════
def fig1_student_perclass_f1():
    ce = load_json("outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/reliabilitymetrics.json")
    # KD and DKD from archive
    kd_path = "outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119/reliabilitymetrics.json"
    dkd_path = "outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722/reliabilitymetrics.json"

    # Try archive paths first, then direct
    try:
        kd = load_json(kd_path)
    except FileNotFoundError:
        kd_path_alt = "outputs/students/_archive/2025-12-23/KD/mobilenetv3_large_100_img224_seed1337_KD_20251223_225031/mobilenetv3_large_100_img224_seed1337_KD_20251223_225031/reliabilitymetrics.json"
        kd = load_json(kd_path_alt)

    try:
        dkd = load_json(dkd_path)
    except FileNotFoundError:
        dkd_path_alt = "outputs/students/_archive/2025-12-23/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251223_225031/mobilenetv3_large_100_img224_seed1337_DKD_20251223_225031/reliabilitymetrics.json"
        dkd = load_json(dkd_path_alt)

    ce_f1 = [ce["raw"]["per_class_f1"][c] for c in CLASSES]
    kd_f1 = [kd["raw"]["per_class_f1"][c] for c in CLASSES]
    dkd_f1 = [dkd["raw"]["per_class_f1"][c] for c in CLASSES]

    x = np.arange(len(CLASSES))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_ce = ax.bar(x - w, ce_f1, w, label="CE", color=COLORS["CE"], edgecolor="white")
    bars_kd = ax.bar(x, kd_f1, w, label="KD", color=COLORS["KD"], edgecolor="white")
    bars_dkd = ax.bar(x + w, dkd_f1, w, label="DKD", color=COLORS["DKD"], edgecolor="white")

    ax.set_ylabel("Per-class F1", fontsize=12)
    ax.set_title("Student Per-class F1: CE vs KD vs DKD (HQ-train Validation)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=11)
    ax.set_ylim(0.55, 0.85)
    ax.legend(fontsize=11, loc="lower right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)

    # Value labels on top of bars
    for bars in [bars_ce, bars_kd, bars_dkd]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_student_perclass_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig1_student_perclass_f1.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Teacher per-class F1 comparison (Stage-A validation)
# ═══════════════════════════════════════════════════════════════════════
def fig2_teacher_perclass_f1():
    rn18 = load_json("outputs/teachers/RN18_resnet18_seed1337_stageA_img224/reliabilitymetrics.json")
    b3 = load_json("outputs/teachers/B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224/reliabilitymetrics.json")
    cnxt = load_json("outputs/teachers/CNXT_convnext_tiny_seed1337_stageA_img224/reliabilitymetrics.json")

    rn18_f1 = [rn18["raw"]["per_class_f1"][c] for c in CLASSES]
    b3_f1 = [b3["raw"]["per_class_f1"][c] for c in CLASSES]
    cnxt_f1 = [cnxt["raw"]["per_class_f1"][c] for c in CLASSES]

    x = np.arange(len(CLASSES))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w, rn18_f1, w, label="ResNet-18", color=COLORS["RN18"], edgecolor="white")
    bars2 = ax.bar(x, b3_f1, w, label="EfficientNet-B3", color=COLORS["B3"], edgecolor="white")
    bars3 = ax.bar(x + w, cnxt_f1, w, label="ConvNeXt-Tiny", color=COLORS["CNXT"], edgecolor="white")

    ax.set_ylabel("Per-class F1", fontsize=12)
    ax.set_title("Teacher Per-class F1: RN18 vs B3 vs CNXT (Stage-A Validation)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=11)
    ax.set_ylim(0.65, 0.95)
    ax.legend(fontsize=11, loc="lower right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_teacher_perclass_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig2_teacher_perclass_f1.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Calibration comparison — Raw ECE vs TS ECE (students)
# ═══════════════════════════════════════════════════════════════════════
def fig3_calibration_comparison():
    models = ["CE", "KD", "DKD"]
    raw_ece = [0.131019, 0.215289, 0.209450]
    ts_ece = [0.049897, 0.027764, 0.026605]
    raw_nll = [1.315335, 2.093148, 1.511788]
    ts_nll = [0.777757, 0.768196, 0.765203]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ECE panel
    x = np.arange(len(models))
    w = 0.35
    bars1 = ax1.bar(x - w / 2, raw_ece, w, label="Raw ECE", color="#ef5350", edgecolor="white")
    bars2 = ax1.bar(x + w / 2, ts_ece, w, label="TS ECE", color="#66BB6A", edgecolor="white")
    ax1.set_ylabel("ECE (lower is better)", fontsize=11)
    ax1.set_title("Expected Calibration Error", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # NLL panel
    bars3 = ax2.bar(x - w / 2, raw_nll, w, label="Raw NLL", color="#ef5350", edgecolor="white")
    bars4 = ax2.bar(x + w / 2, ts_nll, w, label="TS NLL", color="#66BB6A", edgecolor="white")
    ax2.set_ylabel("NLL (lower is better)", fontsize=11)
    ax2.set_title("Negative Log-Likelihood", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    for bars in [bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Student Calibration: CE vs KD vs DKD (HQ-train Validation)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_calibration_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig3_calibration_comparison.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Confusion matrix — webcam demo (smoothed predictions)
# ═══════════════════════════════════════════════════════════════════════
def fig4_confusion_matrix_webcam():
    score = load_json("demo/outputs/20260126_205446/score_results.json")
    cm = np.array(score["metrics"]["smoothed"]["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASS_SHORT, fontsize=11)
    ax.set_yticklabels(CLASSES, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Webcam Demo (Smoothed)\nSession 20260126_205446 (n=4,154)", fontsize=13, fontweight="bold")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_confusion_matrix_webcam.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig4_confusion_matrix_webcam.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 5: Training loss curve — Student CE (10 epochs)
# ═══════════════════════════════════════════════════════════════════════
def fig5_training_curves_ce():
    history = load_json("outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/history.json")

    epochs = [h["epoch"] + 1 for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_acc = [h["val"]["raw"]["accuracy"] for h in history]
    val_macro_f1 = [h["val"]["raw"]["macro_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, train_loss, "o-", color="#E91E63", linewidth=2, markersize=6)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Training Loss", fontsize=12)
    ax1.set_title("Training Loss", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.3)
    ax1.set_xticks(epochs)

    # Val accuracy + macro-F1
    ax2.plot(epochs, val_acc, "o-", color=COLORS["CE"], linewidth=2, markersize=6, label="Accuracy")
    ax2.plot(epochs, val_macro_f1, "s--", color=COLORS["DKD"], linewidth=2, markersize=6, label="Macro-F1")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_title("Validation Accuracy & Macro-F1", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(epochs)

    fig.suptitle("Student CE Training Curves (MobileNetV3-Large, 10 Epochs)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_training_curves_ce.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig5_training_curves_ce.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 6: Cross-dataset macro-F1 comparison (CE vs KD vs DKD on hard gates)
# ═══════════════════════════════════════════════════════════════════════
def fig6_crossdataset_macro_f1():
    data = load_json("outputs/benchmarks/overall_summary__20260208/overall_summary.json")

    datasets_order = ["eval_only", "expw_full", "test_all_sources", "fer2013_folder"]
    dataset_labels = ["Eval-only\n(mixed)", "ExpW\n(in-the-wild)", "Test All\nSources", "FER2013\n(folder)"]
    model_order = ["CE_20251223_225031", "KD_20251229_182119", "DKD_20251229_223722"]
    model_labels = ["CE", "KD", "DKD"]

    # Build lookup
    lookup = {}
    for row in data:
        lookup[(row["model"], row["dataset"])] = row

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(datasets_order))
    w = 0.25

    for i, (model_id, label) in enumerate(zip(model_order, model_labels)):
        vals = [lookup.get((model_id, ds), {}).get("raw_macro_f1", 0) for ds in datasets_order]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=COLORS[label], edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Raw Macro-F1", fontsize=12)
    ax.set_title("Student Macro-F1 Across Hard Gates (Domain Shift Stress Tests)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=10)
    ax.set_ylim(0.35, 0.60)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_crossdataset_macro_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig6_crossdataset_macro_f1.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 7: Teacher Stage-A val vs Hard gates (performance drop)
# ═══════════════════════════════════════════════════════════════════════
def fig7_teacher_domain_shift():
    # Stage-A val macro F1
    rn18_val = 0.780828
    b3_val = 0.790988
    cnxt_val = 0.788959

    # Hard gate macro F1 (from report Section 4.2.2)
    rn18_eo = 0.372670
    b3_eo = 0.392831
    cnxt_eo = 0.388980

    rn18_expw = 0.374009
    b3_expw = 0.406649
    cnxt_expw = 0.382112

    rn18_tas = 0.617067
    b3_tas = 0.645421
    cnxt_tas = 0.638065

    teachers = ["ResNet-18", "EfficientNet-B3", "ConvNeXt-Tiny"]
    stage_a = [rn18_val, b3_val, cnxt_val]
    eval_only = [rn18_eo, b3_eo, cnxt_eo]
    expw = [rn18_expw, b3_expw, cnxt_expw]
    test_all = [rn18_tas, b3_tas, cnxt_tas]

    x = np.arange(len(teachers))
    w = 0.2

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - 1.5 * w, stage_a, w, label="Stage-A Val", color="#4CAF50", edgecolor="white")
    ax.bar(x - 0.5 * w, eval_only, w, label="Eval-only", color="#F44336", edgecolor="white")
    ax.bar(x + 0.5 * w, expw, w, label="ExpW", color="#FF9800", edgecolor="white")
    ax.bar(x + 1.5 * w, test_all, w, label="Test All Sources", color="#2196F3", edgecolor="white")

    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_title("Teacher Performance: Stage-A Validation vs Hard Gates (Domain Shift)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(teachers, fontsize=11)
    ax.set_ylim(0.0, 0.90)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Add drop annotation
    for i in range(3):
        drop = stage_a[i] - eval_only[i]
        ax.annotate(f"−{drop:.2f}", xy=(x[i] - 0.5 * w, eval_only[i]),
                    xytext=(0, -15), textcoords="offset points",
                    ha="center", fontsize=9, color="#D32F2F", fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_teacher_domain_shift.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig7_teacher_domain_shift.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 8: Webcam live scoring — raw vs smoothed per-class F1
# ═══════════════════════════════════════════════════════════════════════
def fig8_webcam_raw_vs_smoothed():
    score = load_json("demo/outputs/20260126_205446/score_results.json")
    raw_f1 = score["metrics"]["raw"]
    smooth_f1 = score["metrics"]["smoothed"]

    raw_vals = [raw_f1["per_class_f1"][c] for c in CLASSES]
    smooth_vals = [smooth_f1["per_class_f1"][c] for c in CLASSES]
    supports = [smooth_f1["per_class_support"][c] for c in CLASSES]

    x = np.arange(len(CLASSES))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars1 = ax.bar(x - w / 2, raw_vals, w, label="Raw", color="#ef5350", edgecolor="white")
    bars2 = ax.bar(x + w / 2, smooth_vals, w, label="Smoothed (EMA + Hysteresis)", color="#42A5F5", edgecolor="white")

    ax.set_ylabel("Per-class F1", fontsize=12)
    ax.set_title("Webcam Demo: Raw vs Smoothed Per-class F1\nSession 20260126_205446", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={s})" for c, s in zip(CLASS_SHORT, supports)], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_webcam_raw_vs_smoothed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig8_webcam_raw_vs_smoothed.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 9: FER2013 Official Split — Accuracy comparison (CE vs KD vs DKD)
# ═══════════════════════════════════════════════════════════════════════
def fig9_fer2013_official():
    data = load_json("outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.json")

    # Extract data by model + split + protocol
    results = {}
    for row in data["rows"]:
        key = (row["model"].split("_")[0], row["dataset"].replace("fer2013_", ""), row["protocol"])
        results[key] = {"acc": row["raw_acc"], "macro_f1": row["raw_macro_f1"]}

    models = ["CE", "KD", "DKD"]
    splits = ["publictest", "privatetest"]
    protocols = ["singlecrop", "tencrop"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for idx, split in enumerate(splits):
        ax = axes[idx]
        x = np.arange(len(models))
        w = 0.3

        sc_acc = [results.get((m, split, "singlecrop"), {}).get("acc", 0) for m in models]
        tc_acc = [results.get((m, split, "tencrop"), {}).get("acc", 0) for m in models]

        bars1 = ax.bar(x - w / 2, sc_acc, w, label="Single-crop", color="#2196F3", edgecolor="white")
        bars2 = ax.bar(x + w / 2, tc_acc, w, label="Ten-crop", color="#FF9800", edgecolor="white")

        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"FER2013 Official {split.replace('test', ' Test').title()}\n(n=3,589)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=12)
        ax.set_ylim(0.55, 0.65)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    fig.suptitle("FER2013 Official Split Evaluation (Student MobileNetV3-Large)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_fer2013_official.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig9_fer2013_official.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 10: Domain shift adaptation A/B — baseline vs adapted (webcam)
# ═══════════════════════════════════════════════════════════════════════
def fig10_adaptation_ab():
    metrics = ["Accuracy", "Macro-F1", "Minority-F1\n(lowest 3)", "Jitter\n(flips/min)"]
    baseline = [0.5879, 0.5248, 0.1609, 14.86]
    adapted = [0.5269, 0.4667, 0.1384, 14.16]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: classification metrics
    x = np.arange(3)
    w = 0.3
    bars1 = axes[0].bar(x - w / 2, baseline[:3], w, label="Baseline (CE)", color="#2196F3", edgecolor="white")
    bars2 = axes[0].bar(x + w / 2, adapted[:3], w, label="Adapted (SL+NegL)", color="#F44336", edgecolor="white")

    axes[0].set_ylabel("Score", fontsize=11)
    axes[0].set_title("Classification Metrics (Smoothed)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics[:3], fontsize=10)
    axes[0].set_ylim(0, 0.7)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            axes[0].annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # Right panel: jitter
    x2 = np.arange(1)
    bars3 = axes[1].bar(x2 - w / 2, [baseline[3]], w, label="Baseline", color="#2196F3", edgecolor="white")
    bars4 = axes[1].bar(x2 + w / 2, [adapted[3]], w, label="Adapted", color="#F44336", edgecolor="white")
    axes[1].set_ylabel("Flips/min", fontsize=11)
    axes[1].set_title("Temporal Stability", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(["Jitter"], fontsize=11)
    axes[1].set_ylim(0, 20)
    axes[1].legend(fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    for bars in [bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            axes[1].annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Domain Shift Adaptation A/B — Same Webcam Session\n(Smoothed; NR-1 Negative Result)", fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_adaptation_ab.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig10_adaptation_ab.png")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print(f"Output directory: {FIG_DIR}")
    print("Generating figures...")

    fig1_student_perclass_f1()
    fig2_teacher_perclass_f1()
    fig3_calibration_comparison()
    fig4_confusion_matrix_webcam()
    fig5_training_curves_ce()
    fig6_crossdataset_macro_f1()
    fig7_teacher_domain_shift()
    fig8_webcam_raw_vs_smoothed()
    fig9_fer2013_official()
    fig10_adaptation_ab()

    print(f"\nAll figures saved to: {FIG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
