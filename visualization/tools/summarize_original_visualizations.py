from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_results(results_path: Path) -> tuple[dict[int, dict], dict[str, str]]:
    text = results_path.read_text(encoding="utf-8", errors="replace")
    subject_metrics: dict[int, dict] = {}
    pattern = re.compile(
        r"Subject\s+(\d+)\s+=>\s+Train Time:\s+([^,]+),\s+Test Time:\s+([^,]+),\s+"
        r"Test Acc:\s+([0-9.]+),\s+Test Loss:\s+([0-9.]+),\s+Test Kappa:\s+([0-9.]+)"
    )
    for match in pattern.finditer(text):
        subject_id = int(match.group(1))
        subject_metrics[subject_id] = {
            "train_time": match.group(2).strip(),
            "test_time": match.group(3).strip(),
            "test_acc": float(match.group(4)),
            "test_loss": float(match.group(5)),
            "test_kappa": float(match.group(6)),
        }

    summary = {}
    summary_patterns = {
        "avg_acc": r"Average Test Accuracy:\s+([^\n]+)",
        "avg_kappa": r"Average Test Kappa:\s+([^\n]+)",
        "avg_loss": r"Average Test Loss:\s+([^\n]+)",
        "total_train_time": r"Total Training Time:\s+([^\n]+)",
        "avg_response_time": r"Average Response Time:\s+([^\n]+)",
    }
    for key, pat in summary_patterns.items():
        m = re.search(pat, text)
        if m:
            summary[key] = m.group(1).strip()
    return subject_metrics, summary


def top_channels(values: list[float], names: list[str], top_k: int = 5) -> list[str]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:top_k]
    return [names[i] for i in order]


def main():
    parser = argparse.ArgumentParser(description="Summarize collected original single-exit visualization outputs.")
    parser.add_argument("--collect-dir", type=Path, required=True)
    parser.add_argument("--single-exit-only", action="store_true")
    args = parser.parse_args()

    collect_dir = args.collect_dir.resolve()
    results_path = collect_dir / "results.txt"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.txt: {results_path}")

    subject_metrics, summary = parse_results(results_path)
    rows = []

    for prediction_path in sorted(collect_dir.glob("*/prediction_summary.json")):
        subject_dir = prediction_path.parent
        pred = json.loads(prediction_path.read_text(encoding="utf-8"))
        importance = json.loads((subject_dir / "input_importance.json").read_text(encoding="utf-8"))
        names = importance["channel_names"]
        subject_id = int(pred["subject_id"])
        metrics = subject_metrics.get(subject_id, {})
        row = {
            "subject_id": subject_id,
            "sample_index": int(pred["sample_index"]),
            "true_name": pred["true_name"],
            "pred_name": pred["pred_name"],
            "confidence": float(pred["confidence"]),
            "test_acc": metrics.get("test_acc"),
            "test_kappa": metrics.get("test_kappa"),
            "test_loss": metrics.get("test_loss"),
            "train_time": metrics.get("train_time"),
            "test_time": metrics.get("test_time"),
            "top_saliency": top_channels(importance["channel_saliency"], names, 5),
            "top_gradxinput": top_channels(importance["channel_grad_times_input"], names, 5),
            "subject_dir_name": subject_dir.name,
        }
        rows.append(row)

    rows.sort(key=lambda x: x["subject_id"])

    md_lines = [
        "# 原始单出口可视化总汇总",
        "",
        f"- 汇总目录：`{collect_dir}`",
        f"- 原始结果文件：`{results_path}`",
        f"- 平均准确率：{summary.get('avg_acc', 'N/A')}",
        f"- 平均 Kappa：{summary.get('avg_kappa', 'N/A')}",
        f"- 平均 Loss：{summary.get('avg_loss', 'N/A')}",
        f"- 总训练时间：{summary.get('total_train_time', 'N/A')}",
        f"- 平均响应时间：{summary.get('avg_response_time', 'N/A')}",
        "",
        "## Subject 汇总表",
        "",
        "| Subject | Sample | True | Pred | Conf | Test Acc | Kappa | Top-5 Saliency | Top-5 Grad x Input | 目录 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    if args.single_exit_only:
        md_lines[10:10] = [
            "## 单出口阶段说明",
            "",
            "- 本次全部按单出口真实阶段命名输出：`conv_features`、`mixed_features`、`transformer_tokens`、`reduced_features`、`tcn_features`、`logits`。",
            "",
        ]
    else:
        md_lines[10:10] = [
            "## 画法对齐说明",
            "",
            "- 本次输出已与之前多出口版本保持相同主文件集合。",
            "- 兼容命名映射：`shallow_routed -> mixed_features`，`mid_routed -> transformer_tokens`，`deep_routed -> reduced_features`，`final_routed -> logits`。",
            "",
        ]

    for row in rows:
        md_lines.append(
            "| {subject_id} | {sample_index} | {true_name} | {pred_name} | {confidence:.4f} | {test_acc:.4f} | {test_kappa:.4f} | {top_sal} | {top_gxi} | `{dirname}` |".format(
                subject_id=row["subject_id"],
                sample_index=row["sample_index"],
                true_name=row["true_name"],
                pred_name=row["pred_name"],
                confidence=row["confidence"],
                test_acc=row["test_acc"] if row["test_acc"] is not None else 0.0,
                test_kappa=row["test_kappa"] if row["test_kappa"] is not None else 0.0,
                top_sal=", ".join(row["top_saliency"]),
                top_gxi=", ".join(row["top_gradxinput"]),
                dirname=row["subject_dir_name"],
            )
        )

    md_lines.extend(["", "## Subject 逐项说明", ""])
    for row in rows:
        dirname = row["subject_dir_name"]
        md_lines.extend(
            [
                f"### Subject {row['subject_id']}",
                "",
                f"- 样本编号：`{row['sample_index']}`",
                f"- 真实类别：`{row['true_name']}`",
                f"- 预测类别：`{row['pred_name']}`",
                f"- 预测置信度：`{row['confidence']:.4f}`",
                f"- 测试准确率：`{row['test_acc']:.4f}`",
                f"- 测试 Kappa：`{row['test_kappa']:.4f}`",
                f"- Top-5 Saliency 电极：`{', '.join(row['top_saliency'])}`",
                f"- Top-5 Grad x Input 电极：`{', '.join(row['top_gradxinput'])}`",
                f"- 子目录：`{dirname}`",
                f"- 主图：`{dirname}/paper_overview_figure.png`",
                f"- 中文说明：`{dirname}/interpretation_notes_zh.md`",
                "",
            ]
        )

    md_path = collect_dir / "visualization_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    csv_path = collect_dir / "visualization_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subject_id",
            "sample_index",
            "true_name",
            "pred_name",
            "confidence",
            "test_acc",
            "test_kappa",
            "test_loss",
            "train_time",
            "test_time",
            "top5_saliency",
            "top5_gradxinput",
            "subject_dir_name",
        ])
        for row in rows:
            writer.writerow([
                row["subject_id"],
                row["sample_index"],
                row["true_name"],
                row["pred_name"],
                f"{row['confidence']:.6f}",
                f"{row['test_acc']:.6f}" if row["test_acc"] is not None else "",
                f"{row['test_kappa']:.6f}" if row["test_kappa"] is not None else "",
                f"{row['test_loss']:.6f}" if row["test_loss"] is not None else "",
                row["train_time"] or "",
                row["test_time"] or "",
                ", ".join(row["top_saliency"]),
                ", ".join(row["top_gradxinput"]),
                row["subject_dir_name"],
            ])

    json_path = collect_dir / "visualization_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "collect_dir": str(collect_dir),
                "results_path": str(results_path),
                "summary": summary,
                **(
                    {}
                    if args.single_exit_only
                    else {
                        "compatibility_aliases": {
                            "shallow_routed": "mixed_features",
                            "mid_routed": "transformer_tokens",
                            "deep_routed": "reduced_features",
                            "final_routed": "logits",
                        },
                    }
                ),
                "stage_layout": "single_exit" if args.single_exit_only else "compat",
                "subjects": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved markdown summary to: {md_path}")
    print(f"Saved csv summary to: {csv_path}")
    print(f"Saved json summary to: {json_path}")


if __name__ == "__main__":
    main()
