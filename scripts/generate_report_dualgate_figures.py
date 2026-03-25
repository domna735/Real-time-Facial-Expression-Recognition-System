from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "research" / "final report" / "figures"


def _add_box(ax, x: float, y: float, w: float, h: float, text: str, fc: str) -> None:
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor="#333333",
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)


def _arrow(ax, x0: float, y0: float, x1: float, y1: float, color: str, label: str | None = None) -> None:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=1.8, color=color),
    )
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.02, label, ha="center", va="bottom", fontsize=9, color=color)


def make_fig12() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Main vertical flow nodes
    _add_box(ax, 0.37, 0.89, 0.26, 0.07, "Candidate checkpoint", "#e8f1ff")
    _add_box(ax, 0.33, 0.75, 0.34, 0.09, "Offline gate\n(eval-only + ExpW)", "#fff4d6")
    _add_box(ax, 0.30, 0.53, 0.40, 0.10, "Same-session replay A/B\n(fixed labels, fixed stabilizer)", "#ffe9d6")
    _add_box(ax, 0.37, 0.35, 0.26, 0.07, "Promotion decision", "#e8f1ff")

    # Threshold detail callouts
    _add_box(
        ax,
        0.05,
        0.71,
        0.24,
        0.15,
        "eval-only FAIL if\nmacro-F1 drop > 0.01\nOR minority-F1 drop > 0.02",
        "#fff9f2",
    )
    _add_box(
        ax,
        0.71,
        0.71,
        0.24,
        0.15,
        "ExpW WIN requires\nminority-F1 gain >= 0.01\nwith macro-F1 drop <= 0.01",
        "#fff9f2",
    )

    _add_box(
        ax,
        0.05,
        0.47,
        0.24,
        0.14,
        "Replay PASS requires\nnon-regression or improvement\nin smoothed macro-F1\nand minority-F1",
        "#fff9f2",
    )

    # End states
    _add_box(ax, 0.06, 0.19, 0.20, 0.09, "Reject", "#ffe3e3")
    _add_box(ax, 0.74, 0.19, 0.20, 0.09, "Promotable", "#e3f9e5")

    # Arrows main path
    _arrow(ax, 0.50, 0.89, 0.50, 0.84, "#2f2f2f")
    _arrow(ax, 0.50, 0.75, 0.50, 0.63, "#2f2f2f")
    _arrow(ax, 0.50, 0.53, 0.50, 0.42, "#2f2f2f")

    # Pass/fail branches
    _arrow(ax, 0.33, 0.79, 0.26, 0.24, "#c0392b", "FAIL")
    _arrow(ax, 0.70, 0.58, 0.74, 0.24, "#c0392b", "FAIL")
    _arrow(ax, 0.63, 0.39, 0.74, 0.24, "#1f8f3a", "PASS")

    # Callout connectors
    _arrow(ax, 0.29, 0.79, 0.33, 0.79, "#555555")
    _arrow(ax, 0.71, 0.79, 0.67, 0.79, "#555555")
    _arrow(ax, 0.29, 0.54, 0.30, 0.58, "#555555")

    note = (
        "Note: The same run ID can have different values under\n"
        "live-scoring and replay-scoring protocols."
    )
    ax.text(0.50, 0.06, note, ha="center", va="center", fontsize=9, color="#444444")

    fig.suptitle("Dual-Gate Promotion Decision Flow", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUT_DIR / "fig12_dual_gate_decision_flow.png", dpi=300)
    plt.close(fig)


def make_fig13() -> None:
    metrics = ["Accuracy", "Macro-F1", "Minority-F1", "Jitter"]
    baseline = np.array([0.588, 0.525, 0.161, 14.86], dtype=float)
    adapted = np.array([0.527, 0.467, 0.138, 14.16], dtype=float)
    delta = adapted - baseline

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.25, 1.0]})

    # Panel A: grouped bars
    ax = axes[0]
    x = np.arange(len(metrics))
    w = 0.36
    b1 = ax.bar(x - w / 2, baseline, w, label="Baseline", color="#4e79a7")
    b2 = ax.bar(x + w / 2, adapted, w, label="Adapted", color="#f28e2b")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Metric Value")
    ax.set_title("Panel A: Baseline vs Adapted")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    for bars, fmt_small, fmt_jitter in [(b1, "{:.3f}", "{:.2f}"), (b2, "{:.3f}", "{:.2f}")]:
        for i, bar in enumerate(bars):
            h = bar.get_height()
            label = fmt_jitter.format(h) if i == 3 else fmt_small.format(h)
            yoff = 0.18 if i == 3 else 0.03
            ax.text(bar.get_x() + bar.get_width() / 2, h + yoff, label, ha="center", va="bottom", fontsize=8)

    # Panel B: deltas only
    ax2 = axes[1]
    colors = []
    for i, d in enumerate(delta):
        if i == 3:
            # Lower jitter is an improvement
            colors.append("#2ca02c" if d < 0 else "#d62728")
        else:
            colors.append("#2ca02c" if d > 0 else "#d62728")

    dbars = ax2.bar(metrics, delta, color=colors)
    ax2.axhline(0.0, color="#333333", linewidth=1.0)
    ax2.set_title("Panel B: Delta (Adapted - Baseline)")
    ax2.set_ylabel("Delta")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    for i, bar in enumerate(dbars):
        h = bar.get_height()
        text = f"{h:.2f}" if i == 3 else f"{h:.3f}"
        va = "bottom" if h >= 0 else "top"
        y = h + (0.03 if h >= 0 else -0.03)
        ax2.text(bar.get_x() + bar.get_width() / 2, y, text, ha="center", va=va, fontsize=9)

    subtitle = "Same-session replay A/B; fixed labels and fixed stabilizer settings."
    footnote = "Footnote: Lower is better for jitter; higher is better for other metrics."
    fig.suptitle("Replay A/B Delta Summary (NR-1)", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.02, subtitle + "  " + footnote, ha="center", fontsize=9)

    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(OUT_DIR / "fig13_replay_ab_delta.png", dpi=300)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_fig12()
    make_fig13()
    print(f"Wrote: {OUT_DIR / 'fig12_dual_gate_decision_flow.png'}")
    print(f"Wrote: {OUT_DIR / 'fig13_replay_ab_delta.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
