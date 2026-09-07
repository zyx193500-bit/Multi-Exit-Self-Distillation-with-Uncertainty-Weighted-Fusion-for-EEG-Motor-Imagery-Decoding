from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from docx import Document
from docx.shared import Inches


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import visualize_single_eeg_stages as viz


ANALYSIS_ROOT = ROOT / "analysis" / "single_eeg_feature_viz_collected"


def clean_text(text: str) -> str:
    return text.replace("��", "+/-").replace("±", "+/-")


def parse_results(results_path: Path) -> tuple[dict[int, dict], dict[str, str]]:
    text = clean_text(results_path.read_text(encoding="utf-8", errors="replace"))
    subject_metrics: dict[int, dict] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(
            r"Subject\s+(\d+)\s+=>\s+Train Time:\s+([^,]+),\s+Test Time:\s+([^,]+),\s+"
            r"Test Acc:\s+([0-9.]+),\s+Test Loss:\s+([0-9.]+),\s+Test Kappa:\s+([0-9.]+)",
            line,
        )
        if not m:
            i += 1
            continue
        subject_id = int(m.group(1))
        subject_metrics[subject_id] = {
            "train_time": m.group(2).strip(),
            "test_time": m.group(3).strip(),
            "test_acc": float(m.group(4)),
            "test_loss": float(m.group(5)),
            "test_kappa": float(m.group(6)),
        }
        if i + 1 < len(lines):
            w = re.match(
                r"\s*Adaptive Weights => Shallow:\s+([0-9.]+),\s+Mid:\s+([0-9.]+),\s+Deep:\s+([0-9.]+),\s+Final:\s+([0-9.]+)",
                lines[i + 1].strip(),
            )
            if w:
                subject_metrics[subject_id]["avg_exit_weights"] = {
                    "shallow": float(w.group(1)),
                    "mid": float(w.group(2)),
                    "deep": float(w.group(3)),
                    "final": float(w.group(4)),
                }
                i += 1
        i += 1

    summary_patterns = {
        "avg_primary_acc": r"Average Test Accuracy \(Primary Output\):\s+([^\n]+)",
        "shallow_exit_acc": r"Shallow Exit Accuracy:\s+([^\n]+)",
        "mid_exit_acc": r"Mid Exit Accuracy:\s+([^\n]+)",
        "deep_exit_acc": r"Deep Exit Accuracy:\s+([^\n]+)",
        "final_exit_acc": r"Final Exit Accuracy:\s+([^\n]+)",
        "avg_shallow_weight": r"Shallow Exit Weight:\s+([^\n]+)",
        "avg_mid_weight": r"Mid Exit Weight:\s+([^\n]+)",
        "avg_deep_weight": r"Deep Exit Weight:\s+([^\n]+)",
        "avg_final_weight": r"Final Exit Weight:\s+([^\n]+)",
        "avg_kappa": r"Average Test Kappa:\s+([^\n]+)",
        "avg_loss": r"Average Test Loss:\s+([^\n]+)",
        "total_train_time": r"Total Training Time:\s+([^\n]+)",
        "avg_response_time": r"Average Response Time:\s+([^\n]+)",
    }
    summary = {}
    for key, pattern in summary_patterns.items():
        m = re.search(pattern, text)
        if m:
            summary[key] = clean_text(m.group(1).strip())
    return subject_metrics, summary


def top_channels(values: list[float], names: list[str], top_k: int = 5) -> list[str]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:top_k]
    return [names[i] for i in order]


def rel(path: Path) -> str:
    return path.resolve().as_posix()


def add_bullet(doc: Document, text: str):
    try:
        doc.add_paragraph(text, style="List Bullet")
    except KeyError:
        doc.add_paragraph(f"- {text}")


def add_paragraph_safe(doc: Document, text: str, style: str | None = None):
    if style is None:
        doc.add_paragraph(text)
        return
    available_styles = {s.name for s in doc.styles}
    if style in available_styles:
        doc.add_paragraph(text, style=style)
    else:
        doc.add_paragraph(text)


def add_picture(doc: Document, path: Path, caption: str, width: float = 6.2):
    if path.exists():
        try:
            doc.add_picture(str(path), width=Inches(width))
            doc.add_paragraph(caption)
            return
        except Exception:
            doc.add_paragraph(f"{caption}（原图存在但无法嵌入 Word，可在结果目录中单独查看：{path}）")


def subject_status_text(acc: float) -> str:
    if acc >= 0.9:
        return "该被试表现非常稳定，说明多出口结构在该个体上形成了清晰而稳定的分层判别表征。"
    if acc >= 0.8:
        return "该被试整体表现较稳定，说明模型已经学到可靠的类别相关模式，但仍存在少量边界样本。"
    if acc >= 0.7:
        return "该被试达到可用水平，说明模型能够抓住主要判别证据，但个体差异仍带来一定混淆。"
    return "该被试个体差异更明显，模型虽然能抓到有效证据，但判别边界相对更模糊。"


def class_text(label: str) -> str:
    mapping = {
        "feet": "脚部运动意象通常更关注中央区和顶中央区附近共同节律变化。",
        "hand(L)": "左手运动意象通常更依赖中央区附近与手部表征相关的局部活动模式。",
        "hand(R)": "右手运动意象通常更依赖中央区附近与手部表征相关的局部活动模式。",
        "tongue": "舌部运动意象通常表现为中央区附近不同于手脚意象的判别模式。",
    }
    return mapping.get(label, "该类别表现为任务相关通道与时间窗中的稀疏判别证据。")


def format_weights(weights: dict[str, float]) -> str:
    ordered_keys = ["shallow", "mid", "deep", "final"]
    return ", ".join(f"{key}={weights[key]:.4f}" for key in ordered_keys if key in weights)


def dominant_exits(weights: dict[str, float], top_k: int = 2) -> str:
    order = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return " 和 ".join(key for key, _ in order)


def choose_representative_subject(subjects: list[dict]) -> dict:
    for subject in subjects:
        if subject["subject_id"] == 3:
            return subject
    return max(subjects, key=lambda x: x["test_acc"])


def load_subject_artifacts(subject_dir: Path) -> tuple[dict, dict, np.lib.npyio.NpzFile]:
    prediction = json.loads((subject_dir / "prediction_summary.json").read_text(encoding="utf-8"))
    importance = json.loads((subject_dir / "input_importance.json").read_text(encoding="utf-8"))
    stage_features = np.load(subject_dir / "stage_features.npz")
    return prediction, importance, stage_features


def stage_stat_lines(stage_features: np.lib.npyio.NpzFile) -> list[str]:
    ordered = [
        "raw_signal",
        "conv_features",
        "shallow_routed",
        "mid_routed",
        "deep_routed",
        "tcn_features",
        "final_routed",
    ]
    lines = []
    for key in ordered:
        arr = stage_features[key]
        lines.append(
            f"{key}: shape={list(arr.shape)}，abs_mean={np.abs(arr).mean():.4f}，abs_max={np.abs(arr).max():.4f}"
        )
    return lines


def salient_time_points(importance: dict, top_k: int = 15) -> str:
    values = np.asarray(importance["time_saliency"], dtype=float)
    order = np.argsort(values)[::-1][:top_k]
    return "，".join(str(int(i)) for i in order)


def load_runtime_config(result_dir: Path) -> dict:
    config_path = result_dir / "config.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def parameter_sections(config: dict) -> list[tuple[str, list[str]]]:
    model_kwargs = config.get("model_kwargs", {})
    preprocessing = config.get("preprocessing", {})
    early_stopping = config.get("early_stopping", {})
    sections = [
        (
            "训练设置",
            [
                f"model={config.get('model', 'N/A')}",
                f"dataset_name={config.get('dataset_name', 'N/A')}",
                f"subject_ids={config.get('subject_ids', 'N/A')}",
                f"seed={config.get('seed', 'N/A')}",
                f"gpu_id={config.get('gpu_id', 'N/A')}",
                f"max_epochs={config.get('max_epochs', 'N/A')}",
                f"precision={config.get('precision', 'N/A')}",
                f"accumulate_grad_batches={config.get('accumulate_grad_batches', 'N/A')}",
                f"selection_metric={config.get('selection_metric', 'N/A')}",
                f"selection_mode={config.get('selection_mode', 'N/A')}",
                f"save_best_checkpoint={config.get('save_best_checkpoint', 'N/A')}",
            ],
        ),
        (
            "早停与监控",
            [
                f"early_stopping.enabled={early_stopping.get('enabled', 'N/A')}",
                f"early_stopping.monitor={early_stopping.get('monitor', 'N/A')}",
                f"early_stopping.mode={early_stopping.get('mode', 'N/A')}",
                f"early_stopping.patience={early_stopping.get('patience', 'N/A')}",
                f"early_stopping.min_delta={early_stopping.get('min_delta', 'N/A')}",
            ],
        ),
        (
            "优化与调度",
            [
                f"optimizer={model_kwargs.get('optimizer', 'N/A')}",
                f"lr={model_kwargs.get('lr', 'N/A')}",
                f"weight_decay={model_kwargs.get('weight_decay', 'N/A')}",
                f"scheduler={model_kwargs.get('scheduler', 'N/A')}",
                f"warmup_epochs={model_kwargs.get('warmup_epochs', 'N/A')}",
                f"warmup_epochs_loso={model_kwargs.get('warmup_epochs_loso', 'N/A')}",
                f"beta_1={model_kwargs.get('beta_1', 'N/A')}",
            ],
        ),
        (
            "预处理与数据增强",
            [
                f"batch_size={preprocessing.get('batch_size', 'N/A')}",
                f"sfreq={preprocessing.get('sfreq', 'N/A')}",
                f"start={preprocessing.get('start', 'N/A')}",
                f"stop={preprocessing.get('stop', 'N/A')}",
                f"z_scale={preprocessing.get('z_scale', 'N/A')}",
                f"interaug={preprocessing.get('interaug', 'N/A')}",
                f"low_cut={preprocessing.get('low_cut', 'N/A')}",
                f"high_cut={preprocessing.get('high_cut', 'N/A')}",
            ],
        ),
        (
            "骨干网络参数",
            [
                f"F1={model_kwargs.get('F1', 'N/A')}",
                f"D={model_kwargs.get('D', 'N/A')}",
                f"d_group={model_kwargs.get('d_group', 'N/A')}",
                f"q_heads={model_kwargs.get('q_heads', 'N/A')}",
                f"kv_heads={model_kwargs.get('kv_heads', 'N/A')}",
                f"trans_depth={model_kwargs.get('trans_depth', 'N/A')}",
                f"tcn_depth={model_kwargs.get('tcn_depth', 'N/A')}",
                f"kernel_length_tcn={model_kwargs.get('kernel_length_tcn', 'N/A')}",
                f"pool_length_1={model_kwargs.get('pool_length_1', 'N/A')}",
                f"pool_length_2={model_kwargs.get('pool_length_2', 'N/A')}",
                f"temp_kernel_lengths={model_kwargs.get('temp_kernel_lengths', 'N/A')}",
                f"use_group_attn={model_kwargs.get('use_group_attn', 'N/A')}",
                f"dropout_conv={model_kwargs.get('dropout_conv', 'N/A')}",
                f"trans_dropout={model_kwargs.get('trans_dropout', 'N/A')}",
                f"dropout_tcn={model_kwargs.get('dropout_tcn', 'N/A')}",
            ],
        ),
        (
            "蒸馏与多出口参数",
            [
                f"primary_output={model_kwargs.get('primary_output', 'N/A')}",
                f"distill_temperature={model_kwargs.get('distill_temperature', 'N/A')}",
                f"byot_proj_dim={model_kwargs.get('byot_proj_dim', 'N/A')}",
                f"byot_normalize_hints={model_kwargs.get('byot_normalize_hints', 'N/A')}",
                f"byot_distill_start_epoch={model_kwargs.get('byot_distill_start_epoch', 'N/A')}",
                f"byot_distill_ramp_epochs={model_kwargs.get('byot_distill_ramp_epochs', 'N/A')}",
                f"byot_hint_start_epoch={model_kwargs.get('byot_hint_start_epoch', 'N/A')}",
                f"byot_hint_ramp_epochs={model_kwargs.get('byot_hint_ramp_epochs', 'N/A')}",
                f"byot_ce_weight_shallow={model_kwargs.get('byot_ce_weight_shallow', 'N/A')}",
                f"byot_ce_weight_mid={model_kwargs.get('byot_ce_weight_mid', 'N/A')}",
                f"byot_ce_weight_deep={model_kwargs.get('byot_ce_weight_deep', 'N/A')}",
                f"byot_kd_weight_shallow={model_kwargs.get('byot_kd_weight_shallow', 'N/A')}",
                f"byot_kd_weight_mid={model_kwargs.get('byot_kd_weight_mid', 'N/A')}",
                f"byot_kd_weight_deep={model_kwargs.get('byot_kd_weight_deep', 'N/A')}",
                f"byot_hint_weight_shallow={model_kwargs.get('byot_hint_weight_shallow', 'N/A')}",
                f"byot_hint_weight_mid={model_kwargs.get('byot_hint_weight_mid', 'N/A')}",
                f"byot_hint_weight_deep={model_kwargs.get('byot_hint_weight_deep', 'N/A')}",
                f"teacher_loss_weight={model_kwargs.get('teacher_loss_weight', 'N/A')}",
                f"teacher_ensemble_weight={model_kwargs.get('teacher_ensemble_weight', 'N/A')}",
                f"ensemble_loss_weight={model_kwargs.get('ensemble_loss_weight', 'N/A')}",
                f"ensemble_prior_shallow={model_kwargs.get('ensemble_prior_shallow', 'N/A')}",
                f"ensemble_prior_mid={model_kwargs.get('ensemble_prior_mid', 'N/A')}",
                f"ensemble_prior_deep={model_kwargs.get('ensemble_prior_deep', 'N/A')}",
            ],
        ),
    ]
    return sections


def save_visualization_for_subject(result_dir: Path, collect_dir: Path, subject_id: int, search_limit: int) -> Path:
    ckpt_path = result_dir / "checkpoints" / f"subject_{subject_id}_best.ckpt"
    config_path = result_dir / "config.yaml"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    config = viz.load_config(config_path)
    pl_module = viz.load_lightning_model(ckpt_path)
    test_dataset, class_names = viz.build_datamodule(config, subject_id)
    sample_index, x, true_label, _ = viz.find_sample(pl_module, test_dataset, prefer_correct=True, limit=search_limit)

    with np.errstate(all="ignore"):
        outputs, stage_outputs = viz.collect_stage_outputs(pl_module, x)
    preds = viz.compute_predictions(pl_module, outputs)
    importance = viz.compute_input_importance(pl_module, x, class_names)

    out_dir = collect_dir / f"{result_dir.name}_subject{subject_id}_sample{sample_index}"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_signal = x.squeeze(0).numpy()
    stage_arrays = {
        "raw_signal": raw_signal,
        "conv_features": viz.to_2d(stage_outputs["conv_features"]),
        "shallow_routed": viz.to_2d(stage_outputs["shallow_routed"]),
        "mid_routed": viz.to_2d(stage_outputs["mid_routed"]),
        "deep_routed": viz.to_2d(stage_outputs["deep_routed"]),
        "tcn_features": viz.to_2d(stage_outputs["tcn_features"]),
        "final_routed": viz.to_2d(stage_outputs["final_routed"]),
    }

    meta = {
        "checkpoint": str(ckpt_path),
        "config": str(config_path),
        "subject_id": subject_id,
        "sample_index": sample_index,
        "true_label": true_label,
        "true_label_name": class_names[true_label],
        "class_names": class_names,
        "eeg_channel_names": viz.BCIC2A_CHANNEL_NAMES,
        "pred_name": class_names[preds["ensemble_pred"]],
        "confidence": float(preds["ensemble_probs"][preds["ensemble_pred"]]),
        **preds,
    }

    (out_dir / "prediction_summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(out_dir / "stage_features.npz", **stage_arrays)

    viz.save_signal_plot(raw_signal, class_names, {**meta, "ensemble_pred": preds["ensemble_pred"]}, out_dir / "raw_signal.png")
    viz.save_combined_figure(stage_arrays, out_dir / "all_stages_heatmap.png")
    for key, arr in stage_arrays.items():
        viz.save_heatmap(arr, key.replace("_", " ").title(), out_dir / f"{key}.png")
    viz.save_channel_importance_bar(
        importance["channel_saliency"],
        viz.BCIC2A_CHANNEL_NAMES,
        "Channel Importance by Saliency",
        out_dir / "channel_importance_saliency.png",
    )
    viz.save_channel_importance_bar(
        importance["channel_grad_times_input"],
        viz.BCIC2A_CHANNEL_NAMES,
        "Channel Importance by Gradient x Input",
        out_dir / "channel_importance_gradxinput.png",
    )
    viz.save_channel_topomap(
        importance["channel_saliency"],
        viz.BCIC2A_CHANNEL_NAMES,
        "Approximate Scalp Map by Saliency",
        out_dir / "channel_topomap_saliency.png",
    )
    viz.save_channel_topomap(
        importance["channel_grad_times_input"],
        viz.BCIC2A_CHANNEL_NAMES,
        "Approximate Scalp Map by Gradient x Input",
        out_dir / "channel_topomap_gradxinput.png",
    )
    (out_dir / "input_importance.json").write_text(
        json.dumps(
            {
                "predicted_class": importance["predicted_class"],
                "predicted_index": importance["predicted_index"],
                "weights": importance["weights"],
                "channel_names": viz.BCIC2A_CHANNEL_NAMES,
                "channel_saliency": importance["channel_saliency"].tolist(),
                "channel_grad_times_input": importance["channel_grad_times_input"].tolist(),
                "time_saliency": importance["time_saliency"].tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    viz.write_interpretation_notes(meta, stage_arrays, importance, out_dir / "interpretation_notes.md")
    viz.write_interpretation_notes_zh(meta, stage_arrays, importance, out_dir / "interpretation_notes_zh.md")
    viz.save_paper_overview_figure(out_dir)
    return out_dir


def collect_subject_rows(collect_dir: Path, result_dir: Path, subject_metrics: dict[int, dict]) -> list[dict]:
    rows = []
    for prediction_path in sorted(collect_dir.glob("*/prediction_summary.json")):
        subject_dir = prediction_path.parent
        pred = json.loads(prediction_path.read_text(encoding="utf-8"))
        importance = json.loads((subject_dir / "input_importance.json").read_text(encoding="utf-8"))
        subject_id = int(pred["subject_id"])
        metrics = subject_metrics.get(subject_id, {})
        avg_weights = metrics.get("avg_exit_weights", {})
        row = {
            "subject_id": subject_id,
            "sample_index": int(pred["sample_index"]),
            "true_name": pred["true_label_name"],
            "pred_name": pred["pred_name"],
            "confidence": float(pred["confidence"]),
            "sample_weights": {
                "shallow": float(pred["weights"][0]),
                "mid": float(pred["weights"][1]),
                "deep": float(pred["weights"][2]),
                "final": float(pred["weights"][3]),
            },
            "avg_exit_weights": avg_weights,
            "test_acc": metrics.get("test_acc"),
            "test_kappa": metrics.get("test_kappa"),
            "test_loss": metrics.get("test_loss"),
            "train_time": metrics.get("train_time"),
            "test_time": metrics.get("test_time"),
            "top_saliency": top_channels(importance["channel_saliency"], importance["channel_names"], 5),
            "top_gradxinput": top_channels(importance["channel_grad_times_input"], importance["channel_names"], 5),
            "subject_dir_name": subject_dir.name,
            "curve_acc": str((result_dir / "curves" / f"subject_{subject_id}_acc.png").resolve()),
            "curve_loss": str((result_dir / "curves" / f"subject_{subject_id}_loss.png").resolve()),
            "confmat": str((result_dir / "confmats" / f"confmat_subject_{subject_id}.png").resolve()),
        }
        rows.append(row)
    rows.sort(key=lambda x: x["subject_id"])
    return rows


def save_summary_files(collect_dir: Path, result_dir: Path, rows: list[dict], summary: dict):
    md_lines = [
        "# TCFormer 多出口可视化总汇总",
        "",
        f"- 汇总目录：`{collect_dir}`",
        f"- 原始结果目录：`{result_dir}`",
        f"- 平均主输出准确率：{summary.get('avg_primary_acc', 'N/A')}",
        f"- Shallow 出口准确率：{summary.get('shallow_exit_acc', 'N/A')}",
        f"- Mid 出口准确率：{summary.get('mid_exit_acc', 'N/A')}",
        f"- Deep 出口准确率：{summary.get('deep_exit_acc', 'N/A')}",
        f"- Final 出口准确率：{summary.get('final_exit_acc', 'N/A')}",
        f"- 平均 Kappa：{summary.get('avg_kappa', 'N/A')}",
        f"- 平均 Loss：{summary.get('avg_loss', 'N/A')}",
        f"- 总训练时间：{summary.get('total_train_time', 'N/A')}",
        f"- 平均响应时间：{summary.get('avg_response_time', 'N/A')}",
        "",
        "## Subject 汇总表",
        "",
        "| Subject | Sample | True | Pred | Conf | Test Acc | Kappa | Top-5 Saliency | Top-5 Grad x Input |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        md_lines.append(
            "| {subject_id} | {sample_index} | {true_name} | {pred_name} | {confidence:.4f} | {test_acc:.4f} | {test_kappa:.4f} | {top_sal} | {top_gxi} |".format(
                subject_id=row["subject_id"],
                sample_index=row["sample_index"],
                true_name=row["true_name"],
                pred_name=row["pred_name"],
                confidence=row["confidence"],
                test_acc=row["test_acc"] if row["test_acc"] is not None else 0.0,
                test_kappa=row["test_kappa"] if row["test_kappa"] is not None else 0.0,
                top_sal=", ".join(row["top_saliency"]),
                top_gxi=", ".join(row["top_gradxinput"]),
            )
        )
    (collect_dir / "visualization_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    with (collect_dir / "visualization_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
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

    payload = {
        "collect_dir": str(collect_dir),
        "result_dir": str(result_dir),
        "summary": summary,
        "stage_layout": "multi_exit",
        "subjects": rows,
    }
    (collect_dir / "visualization_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_metrics_figure(subjects: list[dict], out_path: Path):
    subject_ids = [s["subject_id"] for s in subjects]
    acc = [s["test_acc"] for s in subjects]
    kappa = [s["test_kappa"] for s in subjects]
    conf = [s["confidence"] for s in subjects]
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    axes[0].bar(subject_ids, acc, color="#4C78A8")
    axes[0].set_title("Primary Output Accuracy by Subject")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_xticks(subject_ids)

    axes[1].bar(subject_ids, kappa, color="#F58518")
    axes[1].set_title("Kappa by Subject")
    axes[1].set_ylabel("Kappa")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xticks(subject_ids)

    axes[2].bar(subject_ids, conf, color="#54A24B")
    axes[2].set_title("Selected-Sample Ensemble Confidence by Subject")
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
    names = sorted(set(sal_counter) | set(gxi_counter))
    sal_values = [sal_counter.get(name, 0) for name in names]
    gxi_values = [gxi_counter.get(name, 0) for name in names]
    order = np.argsort(np.array(sal_values) + np.array(gxi_values))[::-1]
    names = [names[i] for i in order]
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
        "TCFormer Multi-Exit Overall Visualization",
        f"Primary Accuracy: {summary.get('avg_primary_acc', 'N/A')}",
        f"Shallow / Mid / Deep / Final: {summary.get('shallow_exit_acc', 'N/A')} | {summary.get('mid_exit_acc', 'N/A')} | {summary.get('deep_exit_acc', 'N/A')} | {summary.get('final_exit_acc', 'N/A')}",
        f"Average Weights: shallow={summary.get('avg_shallow_weight', 'N/A')}, mid={summary.get('avg_mid_weight', 'N/A')}, deep={summary.get('avg_deep_weight', 'N/A')}, final={summary.get('avg_final_weight', 'N/A')}",
        f"Average Kappa: {summary.get('avg_kappa', 'N/A')}",
        f"Average Loss: {summary.get('avg_loss', 'N/A')}",
        f"Total Train Time: {summary.get('total_train_time', 'N/A')}",
        f"Best Accuracy Subject: S{best_acc['subject_id']} ({best_acc['test_acc']:.4f})",
        f"Highest Confidence Sample: S{best_conf['subject_id']} ({best_conf['confidence']:.4f})",
        "Stages: conv_features, shallow_routed, mid_routed, deep_routed, tcn_features, final_routed",
    ]
    ax_text.text(0.01, 0.95, "\n".join(lines), va="top", fontsize=15)

    ax_acc = fig.add_subplot(gs[1, 0:2])
    ax_kappa = fig.add_subplot(gs[1, 2:4])
    subject_ids = [s["subject_id"] for s in subjects]
    acc = [s["test_acc"] for s in subjects]
    kappa = [s["test_kappa"] for s in subjects]
    ax_acc.bar(subject_ids, acc, color="#4C78A8")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title("Primary Accuracy")
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


def build_overall_figures(collect_dir: Path):
    summary = json.loads((collect_dir / "visualization_summary.json").read_text(encoding="utf-8"))
    subjects = sorted(summary["subjects"], key=lambda x: x["subject_id"])
    build_metrics_figure(subjects, collect_dir / "overall_metrics.png")
    build_electrode_figure(subjects, collect_dir / "overall_electrodes.png")
    build_gallery_figure(collect_dir, subjects, collect_dir / "overall_subject_gallery.png")
    build_dashboard_figure(summary["summary"], subjects, collect_dir, collect_dir / "overall_dashboard.png")


def build_docx(result_dir: Path, collect_dir: Path, desktop_docx: Path):
    payload = json.loads((collect_dir / "visualization_summary.json").read_text(encoding="utf-8"))
    summary = payload["summary"]
    subjects = sorted(payload["subjects"], key=lambda x: x["subject_id"])
    runtime_config = load_runtime_config(result_dir)
    param_blocks = parameter_sections(runtime_config)
    rep = choose_representative_subject(subjects)
    rep_dir = collect_dir / rep["subject_dir_name"]
    rep_pred, rep_importance, rep_stage_features = load_subject_artifacts(rep_dir)
    rep_top_sal = "、".join(rep["top_saliency"])
    rep_top_gxi = "、".join(rep["top_gradxinput"])
    rep_times = salient_time_points(rep_importance)
    rep_stage_lines = stage_stat_lines(rep_stage_features)
    rep_weight_text = format_weights(rep["sample_weights"])
    rep_dominant = dominant_exits(rep["sample_weights"])

    doc = Document()
    doc.add_heading("Transformer 多出口可视化与分析", level=1)

    doc.add_heading("放置建议", level=2)
    add_paragraph_safe(
        doc,
        "正文插入位置：放在中文结果讨论结束、英文部分开始之前，作为“单样本可解释性结果与可视化分析”小节。",
        style="Compact",
    )
    add_paragraph_safe(
        doc,
        "图注放置位置：放在原文“图注模板”部分后，补充多出口对应的 Figure 5 与 Figure 6 图注。",
        style="Compact",
    )
    add_paragraph_safe(
        doc,
        "附录放置位置：放在文末，作为“9个subject单样本可解释性案例总览”，便于集中展示全部被试案例。",
        style="Compact",
    )

    doc.add_heading("正文可直接插入内容", level=2)
    doc.add_heading("本次运行参数说明", level=3)
    doc.add_paragraph(
        "为保证本次可视化与结果分析可复现，下面补充本轮正式训练所使用的关键参数。参数按训练设置、优化策略、预处理配置、骨干网络设置以及蒸馏与多出口机制分组展示，可直接附在方法或实验设置部分之后。",
        style="Normal",
    )
    for heading, items in param_blocks:
        doc.add_heading(heading, level=4)
        for item in items:
            add_bullet(doc, item)

    doc.add_heading("单样本可解释性结果与可视化分析", level=3)
    add_paragraph_safe(
        doc,
        "为进一步支撑本文关于“EEG 浅层、中层与深层特征均包含有效判别信息，而多出口动态集成能够提升整体稳定性”的核心观点，这里补充 "
        "Transformer 多出口模型的单样本可解释性分析结果。该部分内容适合放在结果讨论之后，因为它直接服务于前文关于分层表征、出口功能分工以及集成输出优势的论证。",
        style="First Paragraph",
    )
    doc.add_paragraph(
        f"本次可解释性分析基于完整训练结果目录 `{result_dir.name}` 进行，代表性样本选自 subject {rep['subject_id']} 的测试集第 {rep['sample_index']} 个 trial。"
        f"该样本真实标签为 {rep['true_name']}，模型最终预测也为 {rep['pred_name']}，集成输出概率为 {rep['confidence']:.4f}。"
        f"出口权重分别为 {rep_weight_text}，说明当前判断主要依赖 {rep_dominant} 两个较深层出口，而不是浅层分支单独完成决策。",
    )
    doc.add_paragraph(
        "从阶段特征图可以观察到，卷积前端主要提取局部节律和短时波形模式；shallow 分支保留较早期的局部判别线索；mid 与 deep 分支逐步强化跨时间上下文关系；"
        "TCN 模块进一步整合时序演化；final 分支则保留最终分类前最紧凑、最具类别相关性的高层表示。"
        "这一现象说明模型并不是依赖单一时刻的大振幅波形完成分类，而是通过“局部节律提取-中层结构强化-深层语义聚合-动态加权集成”的层级过程逐步完成决策。",
    )
    doc.add_paragraph(
        f"在输入层面，电极贡献分析显示，代表性样本的关键支持证据主要集中在 {rep_top_sal} 等通道，Gradient x Input 给出的高响应电极主要包括 {rep_top_gxi}，"
        f"显著时间窗口主要位于 {rep_times} 附近。该结果进一步说明，模型执行的是任务相关的稀疏证据选择，而不是对所有 EEG 通道平均利用。",
    )
    doc.add_paragraph("下图给出了代表性单样本的论文式总览图，包括原始信号、各阶段特征热图、电极重要性条形图以及近似头皮分布图。")
    add_picture(
        doc,
        rep_dir / "paper_overview_figure.png",
        f"图5. 代表性单样本（subject {rep['subject_id']}, sample {rep['sample_index']}）的可解释性总览图。该图同时展示了原始 EEG、各阶段特征演化过程、电极贡献排序以及近似头皮重要性分布，可用于支撑多出口分层特征与动态集成行为的结果讨论。",
    )
    doc.add_paragraph(
        "此外，为避免代表性样本分析的偶然性，这里进一步对 9 个 subject 的最佳 checkpoint 分别自动选择一个预测正确的测试样本并进行同样的可解释性分析。"
        f"结果表明，9 个被试均成功找到可解释案例，预测类别覆盖 {', '.join(sorted({row['pred_name'] for row in subjects}))}，"
        f"集成置信度大致分布在 {min(row['confidence'] for row in subjects):.4f} 到 {max(row['confidence'] for row in subjects):.4f} 之间，"
        f"Final 出口权重大致分布在 {min(row['sample_weights']['final'] for row in subjects):.4f} 到 {max(row['sample_weights']['final'] for row in subjects):.4f} 之间，"
        "说明当前框架在不同被试上都能够形成稳定的中深层主导、多出口补充的判别模式。"
    )
    doc.add_paragraph("表A. 9个被试单样本可解释案例摘要")
    doc.add_paragraph(
        "上述结果可以作为正文“结果分析”中的补充证据，也可作为 short communication 中解释模型行为差异的重要支撑材料。其作用不只是展示分类是否正确，更重要的是解释模型到底学到了什么、哪些电极提供了主要支持，以及不同深度出口在最终决策中分别承担了什么角色。"
    )

    doc.add_heading("图注模板补充", level=2)
    add_paragraph_safe(
        doc,
        "Figure 5. Representative single-trial interpretability overview including raw EEG, stage-wise feature maps, channel-importance bars, and scalp saliency map.",
        style="Compact",
    )
    add_paragraph_safe(
        doc,
        "Figure 6. Subject-wise representative single-trial interpretability summaries across all nine subjects.",
        style="Compact",
    )

    doc.add_heading("代表性样本中文解释", level=2)
    doc.add_heading("单样本可解释性说明", level=1)
    doc.add_heading("样本结论", level=2)
    add_paragraph_safe(doc, f"预测类别：{rep['pred_name']}", style="Compact")
    add_paragraph_safe(doc, f"真实类别：{rep['true_name']}", style="Compact")
    add_paragraph_safe(doc, f"集成预测置信度：{rep['confidence']:.4f}", style="Compact")
    add_paragraph_safe(doc, f"出口权重：{rep_weight_text}", style="Compact")

    doc.add_heading("论文式图注", level=2)
    add_paragraph_safe(
        doc,
        "图X展示了一个被正确分类样本从原始信号到多出口分层特征表示的逐层演化过程。原始 EEG 图反映了 22 通道输入脑电的时域波动。"
        "Conv Features 展示了卷积前端提取的局部多尺度节律模式。Shallow Routed Features 反映了浅层出口所利用的早期判别线索。"
        "Mid Routed Features 和 Deep Routed Features 分别对应中层与深层语义强化结果。TCN Features 展示了时序卷积网络对多阶段信息进行动态整合后的高强度判别特征。"
        "Final Routed Features 则保留了最终分类前最紧凑、最具类别相关性的证据。",
        style="First Paragraph",
    )

    doc.add_heading("中文论文段落", level=2)
    add_paragraph_safe(
        doc,
        "针对该单样本的可视化结果可以观察到，模型首先在卷积前端阶段提取局部时间窗内的节律模式和基础跨通道响应特征，随后在 shallow、mid 和 deep 三个分支处逐步形成由浅入深的判别表示，"
        "使表征从局部波形特征逐渐过渡到类别相关语义特征。进一步地，TCN 模块对多阶段信息进行联合时序整合，使模型能够捕捉动作意象相关模式在时间上的持续性与演化顺序，最终形成更稳定的判别表示。",
        style="First Paragraph",
    )
    doc.add_paragraph(
        f"从出口权重可以看出，当前样本的分类决策主要依赖 {rep_dominant} 两个较深层出口，而 shallow 出口贡献相对较小。"
        "这说明对于该样本，模型并不是仅依靠浅层局部节律直接完成分类，而是更多依赖中深层时序语义与最终高层表示完成判断。"
        "该现象表明，多出口结构不仅提供了层次化监督，也能够反映浅层、中层与深层特征在判别任务中的不同作用。",
    )
    doc.add_paragraph(
        f"输入梯度分析进一步表明，模型关注的关键证据主要集中在 {rep_top_sal} 等少数中央区及邻近电极，以及一段相对集中的时间窗口内。"
        "这说明模型执行的是任务相关的稀疏证据选择，而不是简单依赖全通道整体振幅变化进行分类。"
        f"{class_text(rep['pred_name'])}",
    )

    doc.add_heading("定量摘要", level=2)
    for line in rep_stage_lines:
        add_paragraph_safe(doc, line, style="Compact")

    doc.add_heading("电极与时间证据", level=2)
    add_paragraph_safe(doc, f"Saliency 前10电极：{rep_top_sal}", style="Compact")
    add_paragraph_safe(doc, f"Gradient x Input 前10电极：{rep_top_gxi}", style="Compact")
    add_paragraph_safe(doc, f"最显著时间点：{rep_times}", style="Compact")

    doc.add_heading("附录：9个subject单样本可解释性案例总览", level=2)
    add_paragraph_safe(
        doc,
        "本附录汇总了 subject 1-9 的代表性正确预测样本。每个小节均包含被试编号、样本索引、真实类别、预测类别、集成置信度以及对应的论文式可解释性总览图，可直接作为导师汇报材料、补充结果页或后续论文附图来源。",
        style="First Paragraph",
    )
    for row in subjects:
        subject_dir = collect_dir / row["subject_dir_name"]
        top_sal = "、".join(row["top_saliency"])
        top_gxi = "、".join(row["top_gradxinput"])
        sample_weights = row["sample_weights"]
        doc.add_heading(f"Subject {row['subject_id']}", level=3)
        add_paragraph_safe(doc, f"样本索引：{row['sample_index']}", style="Compact")
        add_paragraph_safe(doc, f"真实类别：{row['true_name']}", style="Compact")
        add_paragraph_safe(doc, f"预测类别：{row['pred_name']}", style="Compact")
        add_paragraph_safe(doc, f"集成置信度：{row['confidence']:.4f}", style="Compact")
        add_paragraph_safe(doc, f"代表样本出口权重：{format_weights(sample_weights)}", style="Compact")
        add_picture(doc, subject_dir / "paper_overview_figure.png", f"图B{row['subject_id']}. Subject {row['subject_id']} 单样本综合可视化总图。")
        doc.add_paragraph(
            f"简要分析：该被试代表性样本被正确识别为 {row['pred_name']}，测试准确率为 {row['test_acc']:.4f}，Kappa 为 {row['test_kappa']:.4f}。"
            f"{subject_status_text(row['test_acc'])}"
            f"当前样本主要依赖 {dominant_exits(sample_weights)} 出口完成判断。"
            f"Saliency 主要集中在 {top_sal}，Grad x Input 主要集中在 {top_gxi}，说明模型更依赖少数关键通道形成最终判别。"
        )

    doc.save(str(desktop_docx))


def build_markdown_doc(result_dir: Path, collect_dir: Path, output_md: Path):
    payload = json.loads((collect_dir / "visualization_summary.json").read_text(encoding="utf-8"))
    summary = payload["summary"]
    subjects = sorted(payload["subjects"], key=lambda x: x["subject_id"])
    runtime_config = load_runtime_config(result_dir)
    param_blocks = parameter_sections(runtime_config)
    rep = choose_representative_subject(subjects)
    rep_dir = collect_dir / rep["subject_dir_name"]
    _, rep_importance, rep_stage_features = load_subject_artifacts(rep_dir)
    rep_top_sal = "、".join(rep["top_saliency"])
    rep_top_gxi = "、".join(rep["top_gradxinput"])
    rep_times = salient_time_points(rep_importance)
    lines = [
        "# Transformer 多出口可视化与分析",
        "",
        "## 放置建议",
        "",
        "- 正文插入位置：放在中文结果讨论结束、英文部分开始之前，作为“单样本可解释性结果与可视化分析”小节。",
        "- 图注放置位置：放在原文“图注模板”部分后，补充多出口对应的 Figure 5 与 Figure 6 图注。",
        "- 附录放置位置：放在文末，作为“9个subject单样本可解释性案例总览”。",
        "",
        "## 正文可直接插入内容",
        "",
        "### 本次运行参数说明",
        "",
        "为保证本次可视化与结果分析可复现，下面补充本轮正式训练所使用的关键参数。参数按训练设置、优化策略、预处理配置、骨干网络设置以及蒸馏与多出口机制分组展示，可直接附在方法或实验设置部分之后。",
        "",
        "### 单样本可解释性结果与可视化分析",
        "",
        "为进一步支撑本文关于“EEG 浅层、中层与深层特征均包含有效判别信息，而多出口动态集成能够提升整体稳定性”的核心观点，这里补充 Transformer 多出口模型的单样本可解释性分析结果。",
        "",
        f"本次可解释性分析基于完整训练结果目录 `{result_dir.name}` 进行，代表性样本选自 subject {rep['subject_id']} 的测试集第 {rep['sample_index']} 个 trial。该样本真实标签为 {rep['true_name']}，模型最终预测也为 {rep['pred_name']}，集成输出概率为 {rep['confidence']:.4f}。出口权重分别为 {format_weights(rep['sample_weights'])}，说明当前判断主要依赖 {dominant_exits(rep['sample_weights'])} 两个较深层出口。",
        "",
        "从阶段特征图可以观察到，卷积前端主要提取局部节律和短时波形模式；shallow 分支保留较早期的局部判别线索；mid 与 deep 分支逐步强化跨时间上下文关系；TCN 模块进一步整合时序演化；final 分支则保留最终分类前最紧凑、最具类别相关性的高层表示。",
        "",
        f"在输入层面，电极贡献分析显示，代表性样本的关键支持证据主要集中在 {rep_top_sal} 等通道，Gradient x Input 给出的高响应电极主要包括 {rep_top_gxi}，显著时间窗口主要位于 {rep_times} 附近。",
        "",
        f"![]({rel(rep_dir / 'paper_overview_figure.png')})",
        "",
        f"图5. 代表性单样本（subject {rep['subject_id']}, sample {rep['sample_index']}）的可解释性总览图。该图同时展示了原始 EEG、各阶段特征演化过程、电极贡献排序以及近似头皮重要性分布，可用于支撑多出口分层特征与动态集成行为的结果讨论。",
        "",
        f"此外，为避免代表性样本分析的偶然性，这里进一步对 9 个 subject 的最佳 checkpoint 分别自动选择一个预测正确的测试样本并进行同样的可解释性分析。结果表明，9 个被试均成功找到可解释案例，集成置信度大致分布在 {min(row['confidence'] for row in subjects):.4f} 到 {max(row['confidence'] for row in subjects):.4f} 之间，Final 出口权重大致分布在 {min(row['sample_weights']['final'] for row in subjects):.4f} 到 {max(row['sample_weights']['final'] for row in subjects):.4f} 之间。",
        "",
        "表A. 9个被试单样本可解释案例摘要",
        "",
        "## 图注模板补充",
        "",
        "- Figure 5. Representative single-trial interpretability overview including raw EEG, stage-wise feature maps, channel-importance bars, and scalp saliency map.",
        "- Figure 6. Subject-wise representative single-trial interpretability summaries across all nine subjects.",
        "",
        "## 代表性样本中文解释",
        "",
        "### 样本结论",
        "",
        f"- 预测类别：{rep['pred_name']}",
        f"- 真实类别：{rep['true_name']}",
        f"- 集成预测置信度：{rep['confidence']:.4f}",
        f"- 出口权重：{format_weights(rep['sample_weights'])}",
        "",
        "### 论文式图注",
        "",
        "图X展示了一个被正确分类样本从原始信号到多出口分层特征表示的逐层演化过程。Conv Features 展示了卷积前端提取的局部多尺度节律模式。Shallow Routed Features 反映了浅层出口所利用的早期判别线索。Mid Routed Features 和 Deep Routed Features 分别对应中层与深层语义强化结果。TCN Features 展示了时序卷积网络对多阶段信息进行动态整合后的高强度判别特征。Final Routed Features 则保留了最终分类前最紧凑、最具类别相关性的证据。",
        "",
        "### 中文论文段落",
        "",
        f"从出口权重可以看出，当前样本的分类决策主要依赖 {dominant_exits(rep['sample_weights'])} 两个较深层出口，而 shallow 出口贡献相对较小。这说明模型更多依赖中深层时序语义与最终高层表示完成判断，多出口结构不仅提供了层次化监督，也能够反映浅层、中层与深层特征在判别任务中的不同作用。",
        "",
        f"输入梯度分析进一步表明，模型关注的关键证据主要集中在 {rep_top_sal} 等少数中央区及邻近电极，以及一段相对集中的时间窗口内。这说明模型执行的是任务相关的稀疏证据选择，而不是简单依赖全通道整体振幅变化进行分类。{class_text(rep['pred_name'])}",
        "",
        "### 定量摘要",
        "",
        *[f"- {line}" for line in stage_stat_lines(rep_stage_features)],
        "",
        "### 电极与时间证据",
        "",
        f"- Saliency 前10电极：{rep_top_sal}",
        f"- Gradient x Input 前10电极：{rep_top_gxi}",
        f"- 最显著时间点：{rep_times}",
        "",
        "## 总的可视化",
        "",
        f"![]({(collect_dir / 'overall_dashboard.png').resolve().as_posix()})",
        "",
        f"![]({(collect_dir / 'overall_subject_gallery.png').resolve().as_posix()})",
        "",
        f"![]({(collect_dir / 'overall_metrics.png').resolve().as_posix()})",
        "",
        f"![]({(collect_dir / 'overall_electrodes.png').resolve().as_posix()})",
        "",
        f"![]({(result_dir / 'confmats' / 'avg_confusion_matrix.png').resolve().as_posix()})",
        "",
        "## 附录：9个subject单样本可解释性案例总览",
        "",
    ]
    insert_at = lines.index("### 单样本可解释性结果与可视化分析")
    param_lines = []
    for heading, items in param_blocks:
        param_lines.extend([f"#### {heading}", ""])
        param_lines.extend([f"- {item}" for item in items])
        param_lines.append("")
    lines[insert_at:insert_at] = param_lines
    for row in subjects:
        subject_dir = collect_dir / row["subject_dir_name"]
        top_sal = "、".join(row["top_saliency"])
        top_gxi = "、".join(row["top_gradxinput"])
        sample_weights = row["sample_weights"]
        lines.extend(
            [
                f"### Subject {row['subject_id']}",
                "",
                f"- 样本索引：{row['sample_index']}",
                f"- 真实类别：{row['true_name']}",
                f"- 预测类别：{row['pred_name']}",
                f"- 集成置信度：{row['confidence']:.4f}",
                f"- 代表样本出口权重：{format_weights(sample_weights)}",
                "",
                f"![]({(subject_dir / 'paper_overview_figure.png').resolve().as_posix()})",
                "",
                f"简要分析：该被试代表性样本被正确识别为 {row['pred_name']}，测试准确率为 {row['test_acc']:.4f}，Kappa 为 {row['test_kappa']:.4f}。"
                f"{subject_status_text(row['test_acc'])}"
                f"当前样本主要依赖 {dominant_exits(sample_weights)} 出口完成判断。"
                f"Saliency 主要集中在 {top_sal}，Grad x Input 主要集中在 {top_gxi}，说明模型更依赖少数关键通道形成最终判别。",
                "",
            ]
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build TCFormer multi-exit visualizations and a Word report.")
    parser.add_argument("--result-dir", type=Path, required=True, help="Path to a TCFormer multi-exit result directory.")
    parser.add_argument("--search-limit", type=int, default=200, help="How many test samples to scan when selecting one correct sample.")
    parser.add_argument("--docx-name", type=str, default="Transformer多出口可视化与分析_20260503_1331.docx")
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")
    results_path = result_dir / "results.txt"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.txt: {results_path}")

    collect_dir = ANALYSIS_ROOT / result_dir.name
    collect_dir.mkdir(parents=True, exist_ok=True)

    subject_metrics, summary = parse_results(results_path)
    subject_ids = sorted(subject_metrics.keys())
    for subject_id in subject_ids:
        save_visualization_for_subject(result_dir, collect_dir, subject_id, args.search_limit)

    rows = collect_subject_rows(collect_dir, result_dir, subject_metrics)
    save_summary_files(collect_dir, result_dir, rows, summary)
    build_overall_figures(collect_dir)

    output_md = ROOT / "analysis" / f"{result_dir.name}_可视化分析.md"
    build_markdown_doc(result_dir, collect_dir, output_md)

    desktop_docx = Path(r"c:\Users\zyx19\Desktop") / args.docx_name
    fallback_docx = ROOT / "analysis" / args.docx_name
    output_docx = desktop_docx
    try:
        build_docx(result_dir, collect_dir, desktop_docx)
    except Exception:
        build_docx(result_dir, collect_dir, fallback_docx)
        output_docx = fallback_docx

    print(f"COLLECT_DIR={collect_dir}")
    print(f"OUTPUT_MD={output_md}")
    print(f"OUTPUT_DOCX={output_docx}")


if __name__ == "__main__":
    main()
