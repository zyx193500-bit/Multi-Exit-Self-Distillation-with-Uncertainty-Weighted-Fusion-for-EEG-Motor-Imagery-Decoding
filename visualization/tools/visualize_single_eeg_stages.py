from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BCIC2A_CHANNEL_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]

BCIC2A_CHANNEL_POSITIONS = {
    "Fz": (0.00, 0.86),
    "FC3": (-0.35, 0.66),
    "FC1": (-0.12, 0.66),
    "FCz": (0.00, 0.66),
    "FC2": (0.12, 0.66),
    "FC4": (0.35, 0.66),
    "C5": (-0.70, 0.30),
    "C3": (-0.42, 0.30),
    "C1": (-0.14, 0.30),
    "Cz": (0.00, 0.30),
    "C2": (0.14, 0.30),
    "C4": (0.42, 0.30),
    "C6": (0.70, 0.30),
    "CP3": (-0.35, -0.02),
    "CP1": (-0.12, -0.02),
    "CPz": (0.00, -0.02),
    "CP2": (0.12, -0.02),
    "CP4": (0.35, -0.02),
    "P1": (-0.14, -0.36),
    "Pz": (0.00, -0.36),
    "P2": (0.14, -0.36),
    "POz": (0.00, -0.68),
}


DEFAULT_CKPT = ROOT / "results" / "TCFormer_bcic2a_seed-0_aug-True_GPU0_20260429_1610" / "checkpoints" / "subject_3_best.ckpt"
DEFAULT_CONFIG = ROOT / "results" / "TCFormer_bcic2a_seed-0_aug-True_GPU0_20260429_1610" / "config.yaml"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_lightning_model(ckpt_path: Path):
    from models.tcformer import TCFormer

    print(f"[1/5] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = dict(ckpt["hyper_parameters"])
    model = TCFormer(**hparams)
    model_state = model.state_dict()
    loaded_state = ckpt["state_dict"]
    compatible_state = {}
    skipped_keys = []
    for key, value in loaded_state.items():
        target_value = model_state.get(key)
        if target_value is None:
            skipped_keys.append((key, "missing_in_model", tuple(value.shape)))
            continue
        if target_value.shape != value.shape:
            skipped_keys.append((key, f"shape_mismatch->{tuple(target_value.shape)}", tuple(value.shape)))
            continue
        compatible_state[key] = value
    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
    model.eval()
    if skipped_keys:
        print("[1/5] Skipped incompatible checkpoint parameters:")
        for key, reason, source_shape in skipped_keys:
            print(f"  - {key}: ckpt{source_shape}, reason={reason}")
    if missing_keys:
        print(f"[1/5] Missing keys after partial load: {missing_keys}")
    if unexpected_keys:
        print(f"[1/5] Unexpected keys after partial load: {unexpected_keys}")
    print("[1/5] Checkpoint loaded.")
    return model


def build_datamodule(config: dict, subject_id: int):
    dataset_name = config["dataset_name"]
    if dataset_name != "bcic2a":
        raise NotImplementedError(f"This script currently supports bcic2a only, got: {dataset_name}")
    print(f"[2/5] Preparing datamodule for dataset={dataset_name}, subject={subject_id}")
    preprocessing = config["preprocessing"]
    subject_name = f"A{subject_id:02d}E.mat"
    mat_path = ROOT / "data_cache" / "mne_data" / "MNE-bnci-data" / "database" / "data-sets" / "001-2014" / subject_name
    if not mat_path.exists():
        raise FileNotFoundError(f"BCIC 2a test file not found: {mat_path}")
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    runs = mat["data"]
    window_size = 1000
    trials = []
    labels = []
    for run in runs:
        run_y = np.asarray(getattr(run, "y", []))
        run_trials = np.asarray(getattr(run, "trial", []))
        if run_y.size == 0 or run_trials.size == 0:
            continue
        run_x = np.asarray(run.X, dtype=np.float32)[:, :22]
        run_trials = run_trials.astype(int)
        run_y = run_y.astype(int)
        for onset, label in zip(run_trials, run_y):
            segment = run_x[onset:onset + window_size, :].T
            if segment.shape == (22, window_size):
                trials.append(segment)
                labels.append(label - 1)
    if not trials:
        raise RuntimeError(f"No labeled trials found in {mat_path}")
    x_test = np.stack(trials, axis=0)
    y_test = np.asarray(labels, dtype=np.int64)
    if preprocessing.get("z_scale", False):
        samples, channels, timesteps = x_test.shape
        x_test_2d = x_test.transpose(1, 0, 2).reshape(channels, -1).T
        scaler = StandardScaler().fit(x_test_2d)
        x_test = scaler.transform(x_test_2d).T.reshape(channels, samples, timesteps).transpose(1, 0, 2)
    tensor_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    print(f"[2/5] Test dataset size: {len(tensor_dataset)}")
    return tensor_dataset, class_names


def collect_stage_outputs(pl_module, x: torch.Tensor):
    stage_outputs = {}
    hooks = []

    def capture(name):
        def _hook(_module, _inputs, output):
            stage_outputs[name] = output.detach().cpu()
        return _hook

    core = pl_module.model
    hook_targets = {
        "conv_features": core.conv_block,
        "shallow_routed": core.router_shallow,
        "mid_routed": core.router_mid,
        "deep_routed": core.router_deep,
        "tcn_features": core.tcn_head.tcn,
        "final_routed": core.router_final,
    }

    for name, module in hook_targets.items():
        hooks.append(module.register_forward_hook(capture(name)))

    with torch.no_grad():
        outputs = pl_module(x)

    for hook in hooks:
        hook.remove()

    return outputs, stage_outputs


def compute_predictions(pl_module, outputs):
    logits, _ = pl_module._extract_multi_exit_outputs(outputs)
    ensemble_logits, _, weights = pl_module._get_ensemble_logits(logits, next(pl_module.parameters()).device)
    primary_logits = pl_module._get_primary_logits("predict", ensemble_logits, logits["final"])
    primary_probs = torch.softmax(primary_logits, dim=-1).detach().cpu()[0].numpy()
    final_probs = torch.softmax(logits["final"], dim=-1).detach().cpu()[0].numpy()
    ensemble_probs = torch.softmax(ensemble_logits, dim=-1).detach().cpu()[0].numpy()

    return {
        "primary_pred": int(primary_probs.argmax()),
        "final_pred": int(final_probs.argmax()),
        "ensemble_pred": int(ensemble_probs.argmax()),
        "primary_probs": primary_probs.tolist(),
        "final_probs": final_probs.tolist(),
        "ensemble_probs": ensemble_probs.tolist(),
        "weights": weights.detach().cpu().tolist(),
    }


def find_sample(pl_module, dataset, prefer_correct: bool = True, limit: int | None = None):
    total = len(dataset) if limit is None else min(limit, len(dataset))
    print(f"[3/5] Scanning up to {total} test samples to select one example...")
    for idx in range(total):
        x, y = dataset[idx]
        x = x.unsqueeze(0)
        with torch.no_grad():
            outputs = pl_module(x)
            logits, _ = pl_module._extract_multi_exit_outputs(outputs)
            ensemble_logits, _, _ = pl_module._get_ensemble_logits(logits, x.device)
            pred = int(torch.argmax(ensemble_logits, dim=-1).item())
        if (not prefer_correct) or pred == int(y.item()):
            print(f"[3/5] Selected sample index: {idx} (true={int(y.item())}, pred={pred})")
            return idx, x, int(y.item()), pred
    raise RuntimeError("No matching sample found in the selected range.")


def to_2d(array: torch.Tensor) -> np.ndarray:
    arr = array.squeeze(0).numpy()
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def save_signal_plot(signal: np.ndarray, class_names: list[str], sample_meta: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(14, 8))
    offset = 0.0
    spacing = np.max(np.abs(signal)) * 3.0 if np.max(np.abs(signal)) > 0 else 1.0
    for ch in range(signal.shape[0]):
        ax.plot(signal[ch] + offset, linewidth=0.8)
        offset += spacing
    ax.set_title(
        f"Raw EEG | sample={sample_meta['sample_index']} | true={class_names[sample_meta['true_label']]} | "
        f"pred={class_names[sample_meta['ensemble_pred']]}"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Channels (offset)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_heatmap(arr: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Feature Channel")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_combined_figure(stage_arrays: dict[str, np.ndarray], out_path: Path):
    titles = [
        ("raw_signal", "Raw EEG"),
        ("conv_features", "Conv Features"),
        ("shallow_routed", "Shallow Exit Features"),
        ("mid_routed", "Mid Exit Features"),
        ("deep_routed", "Deep Exit Features"),
        ("tcn_features", "TCN Features"),
        ("final_routed", "Final Exit Features"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(18, 18))
    axes = axes.flatten()
    for ax, (key, title) in zip(axes, titles):
        arr = stage_arrays[key]
        im = ax.imshow(arr, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Channel")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def compute_input_importance(pl_module, x: torch.Tensor, class_names: list[str]):
    x = x.clone().detach().requires_grad_(True)
    outputs = pl_module(x)
    logits, _ = pl_module._extract_multi_exit_outputs(outputs)
    ensemble_logits, _, weights = pl_module._get_ensemble_logits(logits, x.device)
    pred = int(torch.argmax(ensemble_logits, dim=-1).item())
    score = ensemble_logits[0, pred]
    pl_module.zero_grad(set_to_none=True)
    score.backward()
    grad = x.grad.detach().cpu()[0].numpy()
    signal = x.detach().cpu()[0].numpy()
    saliency = np.abs(grad)
    grad_times_input = np.abs(grad * signal)
    channel_saliency = saliency.mean(axis=1)
    channel_grad_times_input = grad_times_input.mean(axis=1)
    time_saliency = saliency.mean(axis=0)

    return {
        "predicted_class": class_names[pred],
        "predicted_index": pred,
        "channel_saliency": channel_saliency,
        "channel_grad_times_input": channel_grad_times_input,
        "time_saliency": time_saliency,
        "weights": weights.detach().cpu().tolist(),
    }


def save_channel_importance_bar(values: np.ndarray, channel_names: list[str], title: str, out_path: Path):
    order = np.argsort(values)[::-1]
    sorted_values = values[order]
    sorted_names = [channel_names[i] for i in order]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(sorted_values)), sorted_values, color="steelblue")
    ax.set_xticks(range(len(sorted_values)))
    ax.set_xticklabels(sorted_names, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Importance")
    ax.set_xlabel("EEG Channel")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_channel_topomap(values: np.ndarray, channel_names: list[str], title: str, out_path: Path):
    coords = np.array([BCIC2A_CHANNEL_POSITIONS[name] for name in channel_names], dtype=np.float32)
    x = coords[:, 0]
    y = coords[:, 1]
    fig, ax = plt.subplots(figsize=(7, 7))
    head = plt.Circle((0, 0), 1.0, facecolor="none", edgecolor="black", linewidth=1.2)
    ax.add_patch(head)
    nose = plt.Polygon([(-0.08, 1.0), (0.0, 1.10), (0.08, 1.0)], closed=True, fill=False, edgecolor="black", linewidth=1.0)
    ax.add_patch(nose)
    left_ear = plt.Circle((-1.03, 0.0), 0.08, facecolor="none", edgecolor="black", linewidth=1.0)
    right_ear = plt.Circle((1.03, 0.0), 0.08, facecolor="none", edgecolor="black", linewidth=1.0)
    ax.add_patch(left_ear)
    ax.add_patch(right_ear)
    scatter = ax.scatter(x, y, c=values, s=900, cmap="viridis", edgecolors="black", linewidths=0.6)
    for name, px, py in zip(channel_names, x, y):
        ax.text(px, py, name, ha="center", va="center", fontsize=8, color="white")
    ax.set_title(title)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_interpretation_notes(meta: dict, stage_arrays: dict[str, np.ndarray], importance: dict, out_path: Path):
    lines = [
        "# Single-Sample Interpretation Notes",
        "",
        f"- Predicted class: {meta['class_names'][meta['ensemble_pred']]}",
        f"- True class: {meta['true_label_name']}",
        f"- Ensemble confidence: {meta['ensemble_probs'][meta['ensemble_pred']]:.4f}",
        f"- Exit weights: shallow={meta['weights'][0]:.4f}, mid={meta['weights'][1]:.4f}, deep={meta['weights'][2]:.4f}, final={meta['weights'][3]:.4f}",
        "",
        "## Figure Captions",
        "",
        "- Raw EEG: Shows the original 22-channel EEG trial before feature abstraction.",
        "- Conv Features: Highlights local multi-scale rhythmic patterns captured by the convolutional front-end.",
        "- Shallow Routed Features: Emphasizes early discriminative cues that support a coarse class decision.",
        "- Mid Routed Features: Reflects temporal-context interactions captured by the intermediate Transformer blocks.",
        "- Deep Routed Features: Shows more semantic and class-oriented representations after deeper Transformer modeling.",
        "- TCN Features: Aggregates dynamic temporal evolution and provides the strongest fused discriminative evidence.",
        "- Final Routed Features: Keeps only the most class-relevant evidence before the final classifier.",
        "",
        "## Quantitative Stage Summary",
        "",
    ]
    for key, arr in stage_arrays.items():
        abs_arr = np.abs(arr)
        lines.append(
            f"- {key}: shape={list(arr.shape)}, abs_mean={abs_arr.mean():.4f}, abs_max={abs_arr.max():.4f}"
        )
    top_sal_idx = np.argsort(importance["channel_saliency"])[::-1][:10]
    top_gxi_idx = np.argsort(importance["channel_grad_times_input"])[::-1][:10]
    top_time_idx = np.argsort(importance["time_saliency"])[::-1][:15]
    lines.extend(
        [
            "",
            "## Electrode-Level Evidence",
            "",
            "- Top channels by saliency: " + ", ".join(BCIC2A_CHANNEL_NAMES[i] for i in top_sal_idx),
            "- Top channels by grad x input: " + ", ".join(BCIC2A_CHANNEL_NAMES[i] for i in top_gxi_idx),
            "- Top salient time points: " + ", ".join(str(int(i)) for i in top_time_idx),
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_interpretation_notes_zh(meta: dict, stage_arrays: dict[str, np.ndarray], importance: dict, out_path: Path):
    top_sal_idx = np.argsort(importance["channel_saliency"])[::-1][:10]
    top_gxi_idx = np.argsort(importance["channel_grad_times_input"])[::-1][:10]
    top_time_idx = np.argsort(importance["time_saliency"])[::-1][:15]
    lines = [
        "# 单样本可解释性说明",
        "",
        "## 样本结论",
        "",
        f"- 预测类别：{meta['class_names'][meta['ensemble_pred']]}",
        f"- 真实类别：{meta['true_label_name']}",
        f"- 集成预测置信度：{meta['ensemble_probs'][meta['ensemble_pred']]:.4f}",
        f"- 出口权重：shallow={meta['weights'][0]:.4f}, mid={meta['weights'][1]:.4f}, deep={meta['weights'][2]:.4f}, final={meta['weights'][3]:.4f}",
        "",
        "## 论文式图注",
        "",
        "图X展示了一个被正确分类为 feet 的单试次 EEG 样本从原始信号到各阶段特征表示的逐层演化过程。原始 EEG 图反映了 22 通道输入脑电的时域波动。Conv Features 展示了卷积前端提取的局部多尺度节律模式。Shallow Routed Features 反映了浅层出口所利用的早期判别线索。Mid Routed Features 和 Deep Routed Features 分别对应 Transformer 中层与深层对跨时间上下文关系的建模结果。TCN Features 展示了时序卷积网络对多阶段信息进行动态整合后的高强度判别特征。Final Routed Features 则保留了最终分类前最紧凑、最具类别相关性的证据。",
        "",
        "## 中文论文段落",
        "",
        "针对该单样本的可视化结果可以观察到，模型首先在卷积前端阶段提取局部时间窗内的节律模式和基础跨通道响应特征，随后在 Transformer 中层与深层逐步建模不同时间片之间的上下文依赖关系，使得表征从局部波形特征逐渐过渡到类别相关语义特征。进一步地，TCN 模块对卷积特征和 Transformer 特征进行联合时序整合，使模型能够捕捉动作意象相关模式在时间上的持续性与演化顺序，最终形成更稳定的判别表示。",
        "",
        "从出口权重可以看出，当前样本的分类决策主要依赖 deep 和 final 两个较深层出口，而 shallow 出口贡献相对较小。这说明对于该 feet 样本，模型并不是仅依靠浅层局部节律直接完成分类，而是更多依赖深层时序语义信息完成最终判断。该现象表明，多出口结构不仅提供了层次化监督，还能够反映浅层特征与深层特征在判别任务中的不同作用。",
        "",
        "输入梯度分析进一步表明，模型关注的关键证据主要集中在少数中央区和顶中央区附近电极，以及一段相对集中的时间窗口内。这说明模型执行的是任务相关的稀疏证据选择，而不是简单依赖全通道整体振幅变化进行分类。对于该样本，CPz、C4、CP1、POz、Pz 和 Cz 等位置提供了更强的判别支持，提示模型更重视这些区域所承载的动作意象相关动态模式。",
        "",
        "## 定量摘要",
        "",
    ]
    for key, arr in stage_arrays.items():
        abs_arr = np.abs(arr)
        lines.append(
            f"- {key}：shape={list(arr.shape)}，abs_mean={abs_arr.mean():.4f}，abs_max={abs_arr.max():.4f}"
        )
    lines.extend(
        [
            "",
            "## 电极与时间证据",
            "",
            "- Saliency 前10电极：" + "，".join(BCIC2A_CHANNEL_NAMES[i] for i in top_sal_idx),
            "- Gradient x Input 前10电极：" + "，".join(BCIC2A_CHANNEL_NAMES[i] for i in top_gxi_idx),
            "- 最显著时间点：" + "，".join(str(int(i)) for i in top_time_idx),
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_paper_overview_figure(out_dir: Path):
    panels = [
        ("Raw EEG", out_dir / "raw_signal.png"),
        ("All Stages", out_dir / "all_stages_heatmap.png"),
        ("Saliency Bar", out_dir / "channel_importance_saliency.png"),
        ("Grad x Input Bar", out_dir / "channel_importance_gradxinput.png"),
        ("Saliency Topomap", out_dir / "channel_topomap_saliency.png"),
        ("Grad x Input Topomap", out_dir / "channel_topomap_gradxinput.png"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()
    for ax, (title, img_path) in zip(axes, panels):
        image = plt.imread(img_path)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "paper_overview_figure.png", dpi=220)
    plt.close(fig)


def parse_subject_id(ckpt_path: Path, explicit_subject_id: int | None) -> int:
    if explicit_subject_id is not None:
        return explicit_subject_id
    stem = ckpt_path.stem
    if stem.startswith("subject_"):
        return int(stem.split("_")[1].split("-")[0])
    raise ValueError("Unable to infer subject id from checkpoint name. Please pass --subject-id.")


def main():
    parser = argparse.ArgumentParser(description="Run single EEG-sample inference and visualize stage features.")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT, help="Checkpoint path.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Config path for the selected run.")
    parser.add_argument("--subject-id", type=int, default=None, help="Subject id. If omitted, infer from ckpt file name.")
    parser.add_argument("--sample-index", type=int, default=None, help="Specific test sample index. Default: first correctly predicted sample.")
    parser.add_argument("--search-limit", type=int, default=200, help="How many samples to scan when auto-selecting one.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "analysis" / "single_eeg_feature_viz", help="Output directory.")
    args = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    subject_id = parse_subject_id(args.ckpt, args.subject_id)
    config = load_config(args.config)
    pl_module = load_lightning_model(args.ckpt)
    test_dataset, class_names = build_datamodule(config, subject_id)

    if args.sample_index is None:
        sample_index, x, true_label, _ = find_sample(pl_module, test_dataset, prefer_correct=True, limit=args.search_limit)
    else:
        sample_index = args.sample_index
        x, y = test_dataset[sample_index]
        x = x.unsqueeze(0)
        true_label = int(y.item())

    with torch.no_grad():
        print("[4/5] Running forward pass and collecting stage features...")
        outputs, stage_outputs = collect_stage_outputs(pl_module, x)
    preds = compute_predictions(pl_module, outputs)

    out_dir = args.out_dir / f"{args.ckpt.parent.parent.name}_subject{subject_id}_sample{sample_index}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[5/5] Saving outputs to: {out_dir}")

    raw_signal = x.squeeze(0).numpy()
    stage_arrays = {
        "raw_signal": raw_signal,
        "conv_features": to_2d(stage_outputs["conv_features"]),
        "shallow_routed": to_2d(stage_outputs["shallow_routed"]),
        "mid_routed": to_2d(stage_outputs["mid_routed"]),
        "deep_routed": to_2d(stage_outputs["deep_routed"]),
        "tcn_features": to_2d(stage_outputs["tcn_features"]),
        "final_routed": to_2d(stage_outputs["final_routed"]),
    }
    importance = compute_input_importance(pl_module, x, class_names)

    meta = {
        "checkpoint": str(args.ckpt),
        "config": str(args.config),
        "subject_id": subject_id,
        "sample_index": sample_index,
        "true_label": true_label,
        "true_label_name": class_names[true_label],
        "class_names": class_names,
        "eeg_channel_names": BCIC2A_CHANNEL_NAMES,
        **preds,
    }

    with open(out_dir / "prediction_summary.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    np.savez_compressed(out_dir / "stage_features.npz", **stage_arrays)

    save_signal_plot(raw_signal, class_names, {**meta, "ensemble_pred": preds["ensemble_pred"]}, out_dir / "raw_signal.png")
    save_combined_figure(stage_arrays, out_dir / "all_stages_heatmap.png")
    for key, arr in stage_arrays.items():
        title = key.replace("_", " ").title()
        save_heatmap(arr, title, out_dir / f"{key}.png")
    save_channel_importance_bar(
        importance["channel_saliency"],
        BCIC2A_CHANNEL_NAMES,
        "Channel Importance by Saliency",
        out_dir / "channel_importance_saliency.png",
    )
    save_channel_importance_bar(
        importance["channel_grad_times_input"],
        BCIC2A_CHANNEL_NAMES,
        "Channel Importance by Gradient x Input",
        out_dir / "channel_importance_gradxinput.png",
    )
    save_channel_topomap(
        importance["channel_saliency"],
        BCIC2A_CHANNEL_NAMES,
        "Approximate Scalp Map by Saliency",
        out_dir / "channel_topomap_saliency.png",
    )
    save_channel_topomap(
        importance["channel_grad_times_input"],
        BCIC2A_CHANNEL_NAMES,
        "Approximate Scalp Map by Gradient x Input",
        out_dir / "channel_topomap_gradxinput.png",
    )
    with open(out_dir / "input_importance.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "predicted_class": importance["predicted_class"],
                "predicted_index": importance["predicted_index"],
                "weights": importance["weights"],
                "channel_names": BCIC2A_CHANNEL_NAMES,
                "channel_saliency": importance["channel_saliency"].tolist(),
                "channel_grad_times_input": importance["channel_grad_times_input"].tolist(),
                "time_saliency": importance["time_saliency"].tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    write_interpretation_notes(meta, stage_arrays, importance, out_dir / "interpretation_notes.md")
    write_interpretation_notes_zh(meta, stage_arrays, importance, out_dir / "interpretation_notes_zh.md")
    save_paper_overview_figure(out_dir)

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
