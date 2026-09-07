from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_metrics_figure(subjects: list[dict], out_path: Path):
    subject_ids = [s["subject_id"] for s in subjects]
    acc = [s["test_acc"] for s in subjects]
    kappa = [s["test_kappa"] for s in subjects]
    conf = [s["confidence"] for s in subjects]

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    axes[0].bar(subject_ids, acc, color="#4C78A8")
    axes[0].set_title("Test Accuracy by Subject")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_xticks(subject_ids)

    axes[1].bar(subject_ids, kappa, color="#F58518")
    axes[1].set_title("Test Kappa by Subject")
    axes[1].set_ylabel("Kappa")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xticks(subject_ids)

    axes[2].bar(subject_ids, conf, color="#54A24B")
    axes[2].set_title("Selected-Sample Confidence by Subject")
    axes[2].set_ylabel("Confidence")
    axes[2].set_xlabel("Subject")
    axes[2].set_ylim(0, 1.0)
    axes[2].set_xticks(subject_ids)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_electrode_figure(subjects: list[dict], out_path: Path):
    sal_counter = Counter()
    gxi_counter = Counter()
    for subject in subjects:
        sal_counter.update(subject["top_saliency"])
        gxi_counter.update(subject["top_gradxinput"])

    all_names = sorted(set(sal_counter) | set(gxi_counter))
    sal_values = [sal_counter.get(name, 0) for name in all_names]
    gxi_values = [gxi_counter.get(name, 0) for name in all_names]

    order = np.argsort(np.array(sal_values) + np.array(gxi_values))[::-1]
    names = [all_names[i] for i in order]
    sal_values = [sal_values[i] for i in order]
    gxi_values = [gxi_values[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(names))
    width = 0.38
    ax.bar(x - width / 2, sal_values, width=width, label="Top-5 Saliency", color="#4C78A8")
    ax.bar(x + width / 2, gxi_values, width=width, label="Top-5 Grad x Input", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Count Across Subjects")
    ax.set_title("Key Electrode Frequency Across 9 Subjects")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_gallery_figure(collect_dir: Path, subjects: list[dict], out_path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(18, 24))
    axes = axes.flatten()
    for ax, subject in zip(axes, subjects):
        img_path = collect_dir / subject["subject_dir_name"] / "paper_overview_figure.png"
        image = plt.imread(img_path)
        ax.imshow(image)
        ax.set_title(
            f"S{subject['subject_id']} | acc={subject['test_acc']:.3f} | "
            f"{subject['true_name']}->{subject['pred_name']}"
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_dashboard_figure(summary: dict, subjects: list[dict], collect_dir: Path, out_path: Path):
    best_acc = max(subjects, key=lambda s: s["test_acc"])
    best_conf = max(subjects, key=lambda s: s["confidence"])

    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.9, 1.1, 1.5])

    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis("off")
    lines = [
        "TCFormer Original Single-Exit Overall Visualization",
        f"Average Accuracy: {summary['summary'].get('avg_acc', 'N/A')}",
        f"Average Kappa: {summary['summary'].get('avg_kappa', 'N/A')}",
        f"Average Loss: {summary['summary'].get('avg_loss', 'N/A')}",
        f"Total Train Time: {summary['summary'].get('total_train_time', 'N/A')}",
        f"Best Accuracy Subject: S{best_acc['subject_id']} ({best_acc['test_acc']:.4f})",
        f"Highest Confidence Sample: S{best_conf['subject_id']} ({best_conf['confidence']:.4f})",
        "Stages: conv_features, mixed_features, transformer_tokens, reduced_features, tcn_features, logits",
    ]
    ax_text.text(0.01, 0.95, "\n".join(lines), va="top", fontsize=16)

    ax_acc = fig.add_subplot(gs[1, 0:2])
    ax_kappa = fig.add_subplot(gs[1, 2:4])
    subject_ids = [s["subject_id"] for s in subjects]
    acc = [s["test_acc"] for s in subjects]
    kappa = [s["test_kappa"] for s in subjects]
    ax_acc.bar(subject_ids, acc, color="#4C78A8")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title("Accuracy")
    ax_acc.set_xticks(subject_ids)
    ax_kappa.bar(subject_ids, kappa, color="#F58518")
    ax_kappa.set_ylim(0, 1)
    ax_kappa.set_title("Kappa")
    ax_kappa.set_xticks(subject_ids)

    for idx, subject in enumerate(subjects[:4]):
        ax = fig.add_subplot(gs[2, idx])
        img_path = collect_dir / subject["subject_dir_name"] / "paper_overview_figure.png"
        image = plt.imread(img_path)
        ax.imshow(image)
        ax.set_title(f"S{subject['subject_id']} | {subject['pred_name']} | conf={subject['confidence']:.3f}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Build overall visualizations for the collected original single-exit run.")
    parser.add_argument("--collect-dir", type=Path, required=True)
    args = parser.parse_args()

    collect_dir = args.collect_dir.resolve()
    summary_path = collect_dir / "visualization_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary json: {summary_path}")

    summary = load_summary(summary_path)
    subjects = sorted(summary["subjects"], key=lambda x: x["subject_id"])

    build_metrics_figure(subjects, collect_dir / "overall_metrics.png")
    build_electrode_figure(subjects, collect_dir / "overall_electrodes.png")
    build_gallery_figure(collect_dir, subjects, collect_dir / "overall_subject_gallery.png")
    build_dashboard_figure(summary, subjects, collect_dir, collect_dir / "overall_dashboard.png")

    print(f"Saved overall dashboard to: {collect_dir / 'overall_dashboard.png'}")
    print(f"Saved overall metrics to: {collect_dir / 'overall_metrics.png'}")
    print(f"Saved overall electrodes to: {collect_dir / 'overall_electrodes.png'}")
    print(f"Saved overall gallery to: {collect_dir / 'overall_subject_gallery.png'}")


if __name__ == "__main__":
    main()
