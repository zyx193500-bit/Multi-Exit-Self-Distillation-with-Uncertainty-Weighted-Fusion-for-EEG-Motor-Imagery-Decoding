from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path.home() / "Desktop"
PNG_PATH = DESKTOP / "eeg_multi_exit_method_diagram_paper.png"
SVG_PATH = DESKTOP / "eeg_multi_exit_method_diagram_paper.svg"


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


def resblock(ax, x, y, w, h, title, accent):
    rounded_box(ax, x, y, w, h, "", fc="#f8f7f2", ec=EDGE)
    xs = [x + 0.22, x + 0.56, x + 0.90, x + 1.24]
    cy = y + h * 0.52
    for i in range(3):
        arrow(ax, xs[i] + 0.05, cy, xs[i + 1] - 0.05, cy, color="#8a8a8a", lw=1.3, style="->", mutation=10)
    for idx, cx in enumerate(xs):
        fill = accent if idx > 0 else "#f1f1f1"
        circ = Circle((cx, cy), 0.08, facecolor=fill, edgecolor="#555555", linewidth=1.0)
        ax.add_patch(circ)
    ax.add_patch(Circle((x + 0.58, y + h * 0.77), 0.26, facecolor="none", edgecolor="#555555", linewidth=1.0))
    ax.text(x + w / 2, y + 0.18, title, ha="center", va="center", fontsize=10.5)


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
        rect = Rectangle((x + dx, y + dy), w, h, angle=0, facecolor=fc, edgecolor=EDGE, linewidth=1.0)
        ax.add_patch(rect)
    ax.text(x + w / 2 + 0.12, y - 0.08, title, ha="center", va="top", fontsize=10.5)
    ax.add_patch(Rectangle((x + 0.22, y + 0.02), w, h, fill=False, edgecolor=color, linewidth=1.2))


def loss_box(ax, x, y, title, lines, color):
    h = 0.9 + 0.28 * len(lines)
    rounded_box(ax, x, y, 2.5, h, "", fc="#ffffff", ec=color)
    ax.text(x + 1.25, y + h - 0.28, title, ha="center", va="center", fontsize=11, weight="bold", color=color)
    for i, line in enumerate(lines):
        ax.text(x + 0.16, y + h - 0.62 - 0.28 * i, line, ha="left", va="center", fontsize=9.8, color="#333333")


def build_figure():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.text(8, 8.5, "EEG Multi-Exit Self-Distillation", ha="center", va="center", fontsize=18, weight="bold")

    # top flow like the reference figure
    rounded_box(ax, 0.45, 6.9, 0.9, 0.9, "EEG", fc="#ececec", fontsize=11, weight="bold")
    resblock(ax, 2.0, 6.75, 1.55, 1.05, "Block 1", TOP_COLORS[0])
    resblock(ax, 4.05, 6.75, 1.55, 1.05, "Block 2", TOP_COLORS[1])
    resblock(ax, 6.1, 6.75, 1.55, 1.05, "Block 3", TOP_COLORS[2])
    resblock(ax, 8.15, 6.75, 1.55, 1.05, "Block 4", TOP_COLORS[3])
    rounded_box(ax, 10.15, 6.75, 0.8, 1.05, "FC\nLayer 4", fc="#f3efe8", fontsize=10.5)
    rounded_box(ax, 11.2, 6.75, 0.45, 1.05, "Soft", fc="#f3efe8", fontsize=10.5)
    rounded_box(ax, 12.0, 6.78, 0.5, 0.98, "GT\nLabel", fc="#f6f0f0", fontsize=9.5)
    ax.text(12.27, 7.95, "Final Teacher", ha="center", va="center", fontsize=9.6, rotation=90)
    rounded_box(ax, 13.0, 6.88, 1.95, 0.74, "Primary Output", fc="#fff7ef", ec=ENSEMBLE, fontsize=10.2, weight="bold")

    arrow(ax, 1.35, 7.3, 2.0, 7.3, mutation=13)
    arrow(ax, 3.55, 7.3, 4.05, 7.3, mutation=13)
    arrow(ax, 5.6, 7.3, 6.1, 7.3, mutation=13)
    arrow(ax, 7.65, 7.3, 8.15, 7.3, mutation=13)
    arrow(ax, 9.7, 7.3, 10.15, 7.3, mutation=13)
    arrow(ax, 10.95, 7.3, 11.2, 7.3, mutation=13)

    ax.plot([1.85, 9.95], [6.35, 6.35], linestyle=(0, (4, 3)), color="#9a9a9a", linewidth=1.0)
    ax.text(9.75, 6.14, "backbone forward flow", ha="right", va="top", fontsize=8.8, color="#777777")

    # bottom branches
    bottleneck_stack(ax, 4.05, 4.8, 0.6, 0.55, "Shallow Exit\n(Bottleneck 1)", BOT_COLORS[0])
    bottleneck_stack(ax, 6.35, 4.8, 0.6, 0.55, "Mid Exit\n(Bottleneck 2)", BOT_COLORS[1])
    bottleneck_stack(ax, 8.65, 4.8, 0.6, 0.55, "Deep Exit\n(Bottleneck 3)", BOT_COLORS[2])

    fc_box(ax, 3.9, 3.75, 1.1, 0.68, "FC Layer 1\n(Shallow)")
    fc_box(ax, 6.2, 3.75, 1.1, 0.68, "FC Layer 2\n(Mid)")
    fc_box(ax, 8.5, 3.75, 1.1, 0.68, "FC Layer 3\n(Deep)")
    rounded_box(ax, 3.9, 3.0, 1.1, 0.38, "Softmax 1", fc="#f7f5ef", fontsize=9.8)
    rounded_box(ax, 6.2, 3.0, 1.1, 0.38, "Softmax 2", fc="#f7f5ef", fontsize=9.8)
    rounded_box(ax, 8.5, 3.0, 1.1, 0.38, "Softmax 3", fc="#f7f5ef", fontsize=9.8)

    # downward taps
    arrow(ax, 4.05, 6.75, 4.05, 5.4, mutation=12)
    arrow(ax, 6.35, 6.75, 6.35, 5.4, mutation=12)
    arrow(ax, 8.65, 6.75, 8.65, 5.4, mutation=12)
    arrow(ax, 10.95, 6.75, 10.95, 5.4, mutation=12)

    arrow(ax, 4.35, 4.76, 4.45, 4.43, mutation=12)
    arrow(ax, 6.65, 4.76, 6.75, 4.43, mutation=12)
    arrow(ax, 8.95, 4.76, 9.05, 4.43, mutation=12)
    arrow(ax, 4.45, 3.75, 4.45, 3.38, mutation=12)
    arrow(ax, 6.75, 3.75, 6.75, 3.38, mutation=12)
    arrow(ax, 9.05, 3.75, 9.05, 3.38, mutation=12)

    # supervision flows in clean lanes
    label_lane_y = [5.9, 5.58, 5.26]
    kd_lane_y = [6.08, 5.76, 5.44]
    softmax_targets = [(4.45, 3.2), (6.75, 3.2), (9.05, 3.2)]
    for lane_y, (tx, ty) in zip(label_lane_y, softmax_targets):
        elbow_arrow(
            ax,
            [(12.0, 7.25), (11.78, 7.25), (11.78, lane_y), (tx, lane_y), (tx, ty)],
            color=LABEL,
            lw=1.8,
        )
    for lane_y, (tx, ty) in zip(kd_lane_y, softmax_targets):
        elbow_arrow(
            ax,
            [(11.42, 7.25), (11.15, 7.25), (11.15, lane_y), (tx - 0.08, lane_y), (tx - 0.08, ty + 0.02)],
            color=DISTILL,
            lw=2.0,
        )

    # hint arrows between bottlenecks
    elbow_arrow(ax, [(9.3, 5.14), (8.1, 5.14), (6.95, 5.14)], color=HINT, lw=1.9)
    elbow_arrow(ax, [(7.0, 5.14), (5.8, 5.14), (4.65, 5.14)], color=HINT, lw=1.9)
    ax.text(10.05, 5.2, "Loss source 3", ha="left", va="center", fontsize=10, color="#333333", fontstyle="italic", fontweight="bold")
    ax.text(10.05, 4.98, "L2 loss from hints (features)", ha="left", va="center", fontsize=9.8, color="#333333", fontstyle="italic", fontweight="bold")
    ax.text(10.05, 4.35, "Loss source 2", ha="left", va="center", fontsize=10, color="#333333", fontstyle="italic", fontweight="bold")
    ax.text(10.05, 4.13, "KL divergence loss from distillation", ha="left", va="center", fontsize=9.8, color="#333333", fontstyle="italic", fontweight="bold")
    ax.text(10.05, 3.5, "Loss source 1", ha="left", va="center", fontsize=10, color="#333333", fontstyle="italic", fontweight="bold")
    ax.text(10.05, 3.28, "Cross entropy loss from labels", ha="left", va="center", fontsize=9.8, color="#333333", fontstyle="italic", fontweight="bold")

    # uncertainty and ensemble area
    rounded_box(ax, 10.9, 1.85, 2.4, 0.86, "Uncertainty Weighting\n3 learnable branch weights", fc="#f7fff0", ec=DISTILL, fontsize=10.0, weight="bold")
    rounded_box(ax, 13.75, 1.85, 1.95, 0.86, "Dynamic Ensemble\nPrimary Output", fc="#fff7ef", ec=ENSEMBLE, fontsize=10.0, weight="bold")
    ensemble_lane_y = [2.86, 2.68, 2.5]
    for lane_y, sx in zip(ensemble_lane_y, [4.45, 6.75, 9.05]):
        elbow_arrow(ax, [(sx, 3.0), (sx, lane_y), (10.9, lane_y), (10.9, 2.28)], color=ENSEMBLE, lw=1.7)
    elbow_arrow(ax, [(11.42, 6.75), (11.42, 2.86), (13.3, 2.86), (13.3, 2.28)], color=ENSEMBLE, lw=1.7)
    arrow(ax, 13.3, 2.28, 13.75, 2.28, color=ENSEMBLE, lw=1.9)

    rounded_box(ax, 0.45, 1.75, 2.95, 2.15, "", fc="#ffffff", ec="#777777")
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
