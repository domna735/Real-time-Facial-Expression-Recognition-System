from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "research" / "presentation" / "figures"


def _arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.5,
            color="#2c3e50",
        )
    )


def _box(ax, x: float, y: float, w: float, h: float, text: str, fc: str) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="#2c3e50", linewidth=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color="#111111")


def fig_data_provenance_flow() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.03, 0.58, 0.18, 0.25, "Multi-source\nraw datasets", "#d6eaf8")
    _box(ax, 0.27, 0.58, 0.2, 0.25, "Manifest-first\ningestion", "#d5f5e3")
    _box(ax, 0.53, 0.58, 0.2, 0.25, "Validation gates\n(path+label)", "#fcf3cf")
    _box(ax, 0.79, 0.58, 0.18, 0.25, "Provenance\nSHA256", "#f9ebea")

    _arrow(ax, 0.21, 0.705, 0.27, 0.705)
    _arrow(ax, 0.47, 0.705, 0.53, 0.705)
    _arrow(ax, 0.73, 0.705, 0.79, 0.705)

    ax.text(
        0.5,
        0.34,
        "Evidence: 466,284 rows validated | 0 missing paths | 0 invalid labels",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.18,
        "Outcome: reproducible, audit-ready comparisons across teacher, student, and replay gates",
        ha="center",
        va="center",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figP1_data_provenance_flow.png", dpi=220)
    plt.close(fig)


def fig_teacher_ensemble_arcface() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.04, 0.62, 0.18, 0.22, "ResNet-18\n(w=0.4)", "#d6eaf8")
    _box(ax, 0.04, 0.36, 0.18, 0.22, "EffNet-B3\n(w=0.4)", "#d5f5e3")
    _box(ax, 0.04, 0.10, 0.18, 0.22, "ConvNeXt-T\n(w=0.2)", "#fcf3cf")
    _box(ax, 0.35, 0.35, 0.22, 0.3, "Weighted\nlogit fusion", "#e8daef")
    _box(ax, 0.69, 0.35, 0.26, 0.3, "Soft targets\nfor KD/DKD student", "#f9ebea")

    _arrow(ax, 0.22, 0.73, 0.35, 0.53)
    _arrow(ax, 0.22, 0.47, 0.35, 0.5)
    _arrow(ax, 0.22, 0.21, 0.35, 0.47)
    _arrow(ax, 0.57, 0.5, 0.69, 0.5)

    ax.text(0.5, 0.92, "ArcFace-style margin in teacher training: cos(theta + m)", ha="center", fontsize=11)
    ax.text(0.5, 0.04, "Mixed-domain evidence: test_all_sources acc=0.687, macro-F1=0.660", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figP2_teacher_ensemble_arcface.png", dpi=220)
    plt.close(fig)


def fig_kd_dkd_decomposition() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.05, 0.55, 0.22, 0.3, "KD\nSingle transfer term", "#d6eaf8")
    _box(ax, 0.39, 0.62, 0.22, 0.23, "DKD\nTarget-class term\n(alpha*TCKD)", "#d5f5e3")
    _box(ax, 0.39, 0.27, 0.22, 0.23, "DKD\nNon-target term\n(beta*NCKD)", "#fcf3cf")
    _box(ax, 0.73, 0.42, 0.22, 0.28, "Better early\ngradient control\n+ calibration", "#f9ebea")

    _arrow(ax, 0.27, 0.7, 0.39, 0.73)
    _arrow(ax, 0.27, 0.7, 0.39, 0.38)
    _arrow(ax, 0.61, 0.73, 0.73, 0.58)
    _arrow(ax, 0.61, 0.38, 0.73, 0.53)

    ax.text(0.16, 0.16, "p_T = softmax(z/T)", ha="center", fontsize=11)
    ax.text(0.5, 0.1, "L_DKD = alpha*L_TCKD + beta*L_NCKD", ha="center", fontsize=11)
    ax.text(0.83, 0.16, "ECE: CE 0.050 -> KD 0.028 -> DKD 0.027", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figP3_kd_dkd_decomposition.png", dpi=220)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_data_provenance_flow()
    fig_teacher_ensemble_arcface()
    fig_kd_dkd_decomposition()
    print(f"Saved presentation figures to: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
