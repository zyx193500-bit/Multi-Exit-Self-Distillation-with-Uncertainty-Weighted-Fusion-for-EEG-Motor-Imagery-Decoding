from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


DESKTOP = Path.home() / "Desktop"
PNG_PATH = DESKTOP / "transformer_multi_exit_method_diagram_paper.png"
SVG_PATH = DESKTOP / "transformer_multi_exit_method_diagram_paper.svg"


FORWARD = "#1f1f1f"
LABEL = "#6dbfc4"
DISTILL = "#79b634"
HINT = "#9a9a9a"
ENSEMBLE = "#f2a65a"
BOX = "#f6f3ec"
EDGE = "#666666"
TOP_COLORS = ["#c9d8b6", "#c6dde2", "#d9cfb0", "#dec0c5"]
BOT_COLORS = ["#dfe8d7", "#dbe8ef", "#ece7da"]


def rounded_box(ax, x, y, w, h, text, fc=BOX, ec=EDGE, fontsize=11, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight)
    return patch


def arrow(ax, x1, y1, x2, y2, color=FORWARD, lw=1.8, style="-|>", mutation=12, connectionstyle="arc3"):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(patch)
    return patch


def elbow_arrow(ax, points, color=FORWARD, lw=1.8, mutation=11):
    for (x1, y1), (x2, y2) in zip(points[:-2], points[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, solid_capstyle="round")
    (x1, y1), (x2, y2) = points[-2], points[-1]
    arrow(ax, x1, y1, x2, y2, color=color, lw=lw, mutation=mutation, connectionstyle="arc3")


def stars(ax, x, y, filled, total=5, color=ENSEMBLE):
    text = "".join("★" if i < filled else "☆" for i in range(total))
    ax.text(x, y, text, ha="left", va="center", fontsize=13, color=color, fontweight="bold")


def stage_block(ax, x, y, w, h, title, subtitle, accent):
    rounded_box(ax, x, y, w, h, "", fc="#f8f7f2", ec=EDGE)
    cx = [x + 0.24, x + 0.58, x + 0.92, x + 1.26]
    cy = y + h * 0.52
    for i in range(3):
        arrow(ax, cx[i] + 0.05, cy, cx[i + 1] - 0.05, cy, color="#888888", lw=1.2, style="->", mutation=10)
    for idx, dot_x in enumerate(cx):
        fc = accent if idx > 0 else "#f1f1f1"
        ax.add_patch(Circle((dot_x, cy), 0.08, facecolor=fc, edgecolor="#555555", linewidth=1.0))
    ax.add_patch(Circle((x + 0.58, y + h * 0.77), 0.25, facecolor="none", edgecolor="#555555", linewidth=1.0))
    ax.text(x + w / 2, y + 0.28, title, ha="center", va="center", fontsize=10.2, weight="bold")
    ax.text(x + w / 2, y + 0.1, subtitle, ha="center", va="center", fontsize=8.8)


def fc_box(ax, x, y, w, h, title):
    rounded_box(ax, x, y, w, h, "", fc="#f7f5ef", ec=EDGE)
    cx = [x + 0.22, x + 0.48, x + 0.74, x + 1.0]
    for dot_x in cx:
        ax.add_patch(Circle((dot_x, y + h * 0.66), 0.08, facecolor="#efefef", edgecolor="#666666", linewidth=1.0))
    ax.text(x + w / 2, y + h * 0.38, title, ha="center", va="center", fontsize=9.1)


def bottleneck_stack(ax, x, y, w, h, title, color):
    offsets = [(0.0, 0.0), (0.12, 0.06), (0.24, 0.0)]
    faces = ["#d7e4cd", "#d4e5eb", "#ece5d8"]
    for (dx, dy), fc in zip(offsets, faces):
        rect = Rectangle((x + dx, y + dy), w, h, facecolor=fc, edgecolor=EDGE, linewidth=1.0)
        ax.add_patch(rect)
    ax.add_patch(Rectangle((x + 0.22, y + 0.02), w, h, fill=False, edgecolor=color, linewidth=1.2))
    ax.text(x + w / 2 + 0.12, y - 0.08, title, ha="center", va="top", fontsize=10.3)


def build_figure():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.text(8, 8.5, "Transformer Multi-Exit Self-Distillation", ha="center", va="center", fontsize=18, weight="bold")

    rounded_box(ax, 0.45, 6.9, 0.95, 0.9, "EEG", fc="#ececec", fontsize=11, weight="bold")
    stage_block(ax, 2.0, 6.75, 1.65, 1.05, "Stage 1", "Conv Front-End", TOP_COLORS[0])
    stage_block(ax, 4.2, 6.75, 1.65, 1.05, "Stage 2", "Transformer Mid", TOP_COLORS[1])
    stage_block(ax, 6.4, 6.75, 1.65, 1.05, "Stage 3", "Transformer Deep", TOP_COLORS[2])
    stage_block(ax, 8.6, 6.75, 1.7, 1.05, "Stage 4", "Final High-Level", TOP_COLORS[3])
    rounded_box(ax, 10.7, 6.75, 0.85, 1.05, "FC\nLayer 4", fc="#f3efe8", fontsize=10.5)
    rounded_box(ax, 11.8, 6.75, 0.45, 1.05, "Soft", fc="#f3efe8", fontsize=10.5)
    rounded_box(ax, 12.55, 6.78, 0.52, 0.98, "GT\nLabel", fc="#f6f0f0", fontsize=9.5)
    ax.text(12.82, 7.95, "Final Teacher", ha="center", va="center", fontsize=9.6, rotation=90)
    rounded_box(ax, 13.45, 6.88, 1.95, 0.74, "Primary Output", fc="#fff7ef", ec=ENSEMBLE, fontsize=10.2, weight="bold")

    arrow(ax, 1.4, 7.3, 2.0, 7.3, mutation=13)
    arrow(ax, 3.65, 7.3, 4.2, 7.3, mutation=13)
    arrow(ax, 5.85, 7.3, 6.4, 7.3, mutation=13)
    arrow(ax, 8.05, 7.3, 8.6, 7.3, mutation=13)
    arrow(ax, 10.3, 7.3, 10.7, 7.3, mutation=13)
    arrow(ax, 11.55, 7.3, 11.8, 7.3, mutation=13)
    ax.plot([1.9, 10.55], [6.35, 6.35], linestyle=(0, (4, 3)), color="#9a9a9a", linewidth=1.0)
    ax.text(10.35, 6.14, "backbone forward flow", ha="right", va="top", fontsize=8.8, color="#777777")

    bottleneck_stack(ax, 4.15, 4.8, 0.6, 0.55, "Shallow Exit\n(Bottleneck 1)", BOT_COLORS[0])
    bottleneck_stack(ax, 6.55, 4.8, 0.6, 0.55, "Mid Exit\n(Bottleneck 2)", BOT_COLORS[1])
    bottleneck_stack(ax, 8.95, 4.8, 0.6, 0.55, "Deep Exit\n(Bottleneck 3)", BOT_COLORS[2])
    fc_box(ax, 4.0, 3.75, 1.1, 0.68, "FC Layer 1\n(Shallow)")
    fc_box(ax, 6.4, 3.75, 1.1, 0.68, "FC Layer 2\n(Mid)")
    fc_box(ax, 8.8, 3.75, 1.1, 0.68, "FC Layer 3\n(Deep)")
    rounded_box(ax, 4.0, 3.0, 1.1, 0.38, "Softmax 1", fc="#f7f5ef", fontsize=9.8)
    rounded_box(ax, 6.4, 3.0, 1.1, 0.38, "Softmax 2", fc="#f7f5ef", fontsize=9.8)
    rounded_box(ax, 8.8, 3.0, 1.1, 0.38, "Softmax 3", fc="#f7f5ef", fontsize=9.8)

    arrow(ax, 4.15, 6.75, 4.15, 5.4, mutation=12)
    arrow(ax, 6.55, 6.75, 6.55, 5.4, mutation=12)
    arrow(ax, 8.95, 6.75, 8.95, 5.4, mutation=12)
    arrow(ax, 11.2, 6.75, 11.2, 5.4, mutation=12)
    arrow(ax, 4.45, 4.76, 4.55, 4.43, mutation=12)
    arrow(ax, 6.85, 4.76, 6.95, 4.43, mutation=12)
    arrow(ax, 9.25, 4.76, 9.35, 4.43, mutation=12)
    arrow(ax, 4.55, 3.75, 4.55, 3.38, mutation=12)
    arrow(ax, 6.95, 3.75, 6.95, 3.38, mutation=12)
    arrow(ax, 9.35, 3.75, 9.35, 3.38, mutation=12)

    label_lane_y = [5.9, 5.58, 5.26]
    kd_lane_y = [6.08, 5.76, 5.44]
    softmax_targets = [(4.55, 3.2), (6.95, 3.2), (9.35, 3.2)]
    for lane_y, (tx, ty) in zip(label_lane_y, softmax_targets):
        elbow_arrow(ax, [(12.55, 7.25), (12.28, 7.25), (12.28, lane_y), (tx, lane_y), (tx, ty)], color=LABEL, lw=1.8)
    for lane_y, (tx, ty) in zip(kd_lane_y, softmax_targets):
        elbow_arrow(ax, [(11.98, 7.25), (11.72, 7.25), (11.72, lane_y), (tx - 0.08, lane_y), (tx - 0.08, ty + 0.02)], color=DISTILL, lw=2.0)

    elbow_arrow(ax, [(9.6, 5.14), (8.35, 5.14), (7.15, 5.14)], color=HINT, lw=1.9)
    elbow_arrow(ax, [(7.2, 5.14), (5.95, 5.14), (4.75, 5.14)], color=HINT, lw=1.9)

    ax.text(10.55, 5.2, "Loss source 3", ha="left", va="center", fontsize=10, color="#333333", fontstyle="italic", fontweight="bold")
    ax.text(10.55, 4.98, "L2 loss from hints (features)", ha="left", va="center", fontsize=9.8, color="#333333", fontstyle="italic", fontweight="bold")
    ax.text(10.55, 4.35, "Loss source 2", ha="left", va="center", fontsize=10, color="#333333", fontstyle="italic", fontweight="bold")
    ax.text(10.55, 4.13, "KL divergence loss from distillation", ha="left", va="center", fontsize=9.8, color="#333333", fontstyle="italic", fontweight="bold")
    ax.text(10.55, 3.5, "Loss source 1", ha="left", va="center", fontsize=10, color="#333333", fontstyle="italic", fontweight="bold")
    ax.text(10.55, 3.28, "Cross entropy loss from labels", ha="left", va="center", fontsize=9.8, color="#333333", fontstyle="italic", fontweight="bold")

    rounded_box(ax, 11.25, 1.85, 2.35, 0.86, "Uncertainty Weighting\n3 learnable branch weights", fc="#f7fff0", ec=DISTILL, fontsize=10.0, weight="bold")
    rounded_box(ax, 14.1, 1.85, 1.75, 0.86, "Dynamic Ensemble\nPrimary Output", fc="#fff7ef", ec=ENSEMBLE, fontsize=9.9, weight="bold")
    ensemble_lane_y = [2.86, 2.68, 2.5]
    for lane_y, sx in zip(ensemble_lane_y, [4.55, 6.95, 9.35]):
        elbow_arrow(ax, [(sx, 3.0), (sx, lane_y), (11.25, lane_y), (11.25, 2.28)], color=ENSEMBLE, lw=1.7)
    elbow_arrow(ax, [(12.02, 6.75), (12.02, 2.86), (13.65, 2.86), (13.65, 2.28)], color=ENSEMBLE, lw=1.7)
    arrow(ax, 13.65, 2.28, 14.1, 2.28, color=ENSEMBLE, lw=1.9)

    rounded_box(ax, 0.45, 1.75, 3.0, 2.15, "", fc="#ffffff", ec="#777777")
    arrow(ax, 0.8, 3.45, 1.4, 3.45, color=FORWARD, lw=1.8)
    ax.text(1.58, 3.45, "forward flow", ha="left", va="center", fontsize=9.1)
    arrow(ax, 0.8, 3.05, 1.4, 3.05, color=HINT, lw=1.8)
    ax.text(1.58, 3.05, "hint supervision", ha="left", va="center", fontsize=9.1)
    arrow(ax, 0.8, 2.65, 1.4, 2.65, color=LABEL, lw=1.8)
    ax.text(1.58, 2.65, "label supervision", ha="left", va="center", fontsize=9.1)
    arrow(ax, 0.8, 2.25, 1.4, 2.25, color=DISTILL, lw=1.8)
    ax.text(1.58, 2.25, "distillation supervision", ha="left", va="center", fontsize=9.1)
    arrow(ax, 0.8, 1.9, 1.4, 1.9, color=ENSEMBLE, lw=1.8)
    ax.text(1.58, 1.9, "ensemble aggregation", ha="left", va="center", fontsize=9.1)

    fig.tight_layout(pad=0.6)
    fig.savefig(PNG_PATH, dpi=160)
    fig.savefig(SVG_PATH)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
    print(PNG_PATH)
    print(SVG_PATH)
