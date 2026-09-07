from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = ROOT / "_upstream_TCFormer"
if not UPSTREAM_ROOT.exists():
    fallback_root = ROOT / "yuanbantcformer"
    if fallback_root.exists():
        UPSTREAM_ROOT = fallback_root

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from author_original_single_exit.run_original_tcformer import install_runtime_compatibility_shims

install_runtime_compatibility_shims()

from models.tcformer import TCFormer


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


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(ckpt_path: Path) -> TCFormer:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = dict(ckpt["hyper_parameters"])
    model = TCFormer(**hparams)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def parse_subject_id(ckpt_path: Path, explicit_subject_id: int | None) -> int:
    if explicit_subject_id is not None:
        return explicit_subject_id
    stem = ckpt_path.stem
    if stem.startswith("subject_"):
        return int(stem.split("_")[1].split("-")[0])
    raise ValueError("Unable to infer subject id from checkpoint path.")


def load_subject_evaluation_trials(subject_id: int, preprocessing: dict):
    mat_path = (
        ROOT / "data_cache" / "mne_data" / "MNE-bnci-data" / "database" / "data-sets" / "001-2014" / f"A{subject_id:02d}E.mat"
    )
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing BCIC2a evaluation file: {mat_path}")

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

    x = np.stack(trials, axis=0)
    y = np.asarray(labels, dtype=np.int64)

    if preprocessing.get("z_scale", False):
        samples, channels, timesteps = x.shape
        x2d = x.transpose(1, 0, 2).reshape(channels, -1).T
        scaler = StandardScaler().fit(x2d)
        x = scaler.transform(x2d).T.reshape(channels, samples, timesteps).transpose(1, 0, 2)

    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def to_2d(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().squeeze(0).numpy()
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def collect_stage_outputs(model: TCFormer, x: torch.Tensor):
    stage_outputs = {}
    hooks = []
    core = model.model

    def capture(name):
        def _hook(_module, _inputs, output):
            stage_outputs[name] = output.detach().cpu()
        return _hook

    hook_targets = {
        "conv_features": core.conv_block,
        "mixed_features": core.mix,
        "transformer_tokens": core.transformer[-1],
        "reduced_features": core.reduce,
        "tcn_features": core.tcn_head.tcn,
        "logits": core.tcn_head,
    }

    for name, module in hook_targets.items():
        hooks.append(module.register_forward_hook(capture(name)))

    with torch.no_grad():
        logits = model(x)

    for hook in hooks:
        hook.remove()

    return logits, stage_outputs


def find_sample(model: TCFormer, x_all: torch.Tensor, y_all: torch.Tensor, limit: int):
    total = min(limit, len(x_all))
    for idx in range(total):
        x = x_all[idx:idx + 1]
        with torch.no_grad():
            pred = int(torch.argmax(model(x), dim=-1).item())
        if pred == int(y_all[idx].item()):
            return idx, x, int(y_all[idx].item()), pred
    x = x_all[0:1]
    with torch.no_grad():
        pred = int(torch.argmax(model(x), dim=-1).item())
    return 0, x, int(y_all[0].item()), pred


def save_heatmap(arr: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Feature Channel")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_signal_plot(signal: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(14, 8))
    offset = 0.0
    spacing = np.max(np.abs(signal)) * 3.0 if np.max(np.abs(signal)) > 0 else 1.0
    for ch in range(signal.shape[0]):
        ax.plot(signal[ch] + offset, linewidth=0.8)
        offset += spacing
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Channels (offset)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_combined_figure(stage_arrays: dict[str, np.ndarray], titles: list[tuple[str, str]], out_path: Path):
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


def compute_input_importance(model: TCFormer, x: torch.Tensor):
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    pred = int(torch.argmax(logits, dim=-1).item())
    score = logits[0, pred]
    model.zero_grad(set_to_none=True)
    score.backward()
    grad = x.grad.detach().cpu()[0].numpy()
    signal = x.detach().cpu()[0].numpy()
    saliency = np.abs(grad)
    grad_times_input = np.abs(grad * signal)
    return {
        "pred": pred,
        "channel_saliency": saliency.mean(axis=1),
        "channel_grad_times_input": grad_times_input.mean(axis=1),
        "time_saliency": saliency.mean(axis=0),
    }


def save_channel_importance_bar(values: np.ndarray, title: str, out_path: Path):
    order = np.argsort(values)[::-1]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(order)), values[order], color="steelblue")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([BCIC2A_CHANNEL_NAMES[i] for i in order], rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_channel_topomap(values: np.ndarray, title: str, out_path: Path):
    coords = np.array([BCIC2A_CHANNEL_POSITIONS[name] for name in BCIC2A_CHANNEL_NAMES], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7, 7))
    head = plt.Circle((0, 0), 1.0, facecolor="none", edgecolor="black", linewidth=1.2)
    ax.add_patch(head)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=900, cmap="viridis", edgecolors="black", linewidths=0.6)
    for name, px, py in zip(BCIC2A_CHANNEL_NAMES, coords[:, 0], coords[:, 1]):
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


def write_notes(meta: dict, stage_arrays: dict[str, np.ndarray], importance: dict, out_path: Path, single_exit_only: bool):
    top_sal = np.argsort(importance["channel_saliency"])[::-1][:10]
    top_gxi = np.argsort(importance["channel_grad_times_input"])[::-1][:10]
    lines = [
        "# 原始单出口可视化说明",
        "",
        f"- 预测类别：{meta['pred_name']}",
        f"- 真实类别：{meta['true_name']}",
        f"- 预测置信度：{meta['confidence']:.4f}",
        "",
        "## 阶段解释",
        "- Conv Features：卷积前端提取的局部时频节律模式。",
        "- Mixed Features：卷积特征经过 1x1 混合后的组间信息重组。",
        "- Transformer Tokens：Transformer 最深层输出的时序上下文表征。",
        "- Reduced Features：将 token 特征压缩回时序特征后的结果。",
        "- TCN Features：融合卷积和 Transformer 信息后的时序卷积表示。",
        "- Logits：最终分类器输出的类别证据。",
        "## 定量摘要",
    ]
    if not single_exit_only:
        lines[lines.index("## 定量摘要"):lines.index("## 定量摘要")] = [
            "",
            "## 与多出口画法的对齐关系",
            "- 为了和之前多出口版本的图保持一致，本次额外输出了兼容命名。",
            "- `shallow_routed.png` 对应单出口的 `mixed_features`。",
            "- `mid_routed.png` 对应单出口的 `transformer_tokens`。",
            "- `deep_routed.png` 对应单出口的 `reduced_features`。",
            "- `final_routed.png` 对应单出口的 `logits`。",
            "",
        ]
    for key, arr in stage_arrays.items():
        abs_arr = np.abs(arr)
        lines.append(f"- {key}: shape={list(arr.shape)}, abs_mean={abs_arr.mean():.4f}, abs_max={abs_arr.max():.4f}")
    lines.extend(
        [
            "",
            "## 关键电极",
            "- Saliency 前10电极：" + "，".join(BCIC2A_CHANNEL_NAMES[i] for i in top_sal),
            "- Grad x Input 前10电极：" + "，".join(BCIC2A_CHANNEL_NAMES[i] for i in top_gxi),
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_notes_en(meta: dict, stage_arrays: dict[str, np.ndarray], importance: dict, out_path: Path, single_exit_only: bool):
    top_sal = np.argsort(importance["channel_saliency"])[::-1][:10]
    top_gxi = np.argsort(importance["channel_grad_times_input"])[::-1][:10]
    lines = [
        "# Original Single-Exit Visualization Notes",
        "",
        f"- Predicted class: {meta['pred_name']}",
        f"- True class: {meta['true_name']}",
        f"- Prediction confidence: {meta['confidence']:.4f}",
        "",
        "## Stage Interpretation",
        "- Conv Features: local multi-scale rhythmic patterns from the convolutional front-end.",
        "- Mixed Features: early channel-mixing representation after the 1x1 projection.",
        "- Transformer Tokens: contextual token representation after deep temporal modeling.",
        "- Reduced Features: compressed deep representation projected back to the temporal axis.",
        "- TCN Features: fused temporal evidence after combining convolutional and Transformer features.",
        "- Logits: final class evidence before the decision.",
        "## Quantitative Summary",
    ]
    if not single_exit_only:
        lines[lines.index("## Quantitative Summary"):lines.index("## Quantitative Summary")] = [
            "",
            "## Compatibility With Multi-Exit Layout",
            "- `shallow_routed.png` is an alias of `mixed_features`.",
            "- `mid_routed.png` is an alias of `transformer_tokens`.",
            "- `deep_routed.png` is an alias of `reduced_features`.",
            "- `final_routed.png` is an alias of `logits`.",
            "",
        ]
    for key, arr in stage_arrays.items():
        abs_arr = np.abs(arr)
        lines.append(f"- {key}: shape={list(arr.shape)}, abs_mean={abs_arr.mean():.4f}, abs_max={abs_arr.max():.4f}")
    lines.extend(
        [
            "",
            "## Key Electrodes",
            "- Top saliency channels: " + ", ".join(BCIC2A_CHANNEL_NAMES[i] for i in top_sal),
            "- Top grad x input channels: " + ", ".join(BCIC2A_CHANNEL_NAMES[i] for i in top_gxi),
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Visualize the author's original single-exit TCFormer stages.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subject-id", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--search-limit", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "analysis" / "single_eeg_feature_viz_original")
    parser.add_argument("--stage-layout", choices=["compat", "single_exit"], default="compat")
    args = parser.parse_args()

    config = load_config(args.config)
    model = load_model(args.ckpt)
    subject_id = parse_subject_id(args.ckpt, args.subject_id)
    x_all, y_all = load_subject_evaluation_trials(subject_id, config["preprocessing"])

    if args.sample_index is None:
        sample_index, x, true_label, pred_label = find_sample(model, x_all, y_all, args.search_limit)
    else:
        sample_index = args.sample_index
        x = x_all[sample_index:sample_index + 1]
        true_label = int(y_all[sample_index].item())
        with torch.no_grad():
            pred_label = int(torch.argmax(model(x), dim=-1).item())

    logits, stage_outputs = collect_stage_outputs(model, x)
    probs = torch.softmax(logits, dim=-1).detach().cpu()[0].numpy()
    importance = compute_input_importance(model, x)

    out_dir = args.out_dir / f"{args.ckpt.parent.parent.name}_subject{subject_id}_sample{sample_index}"
    out_dir.mkdir(parents=True, exist_ok=True)

    original_stage_arrays = {
        "raw_signal": x.squeeze(0).cpu().numpy(),
        "conv_features": to_2d(stage_outputs["conv_features"]),
        "mixed_features": to_2d(stage_outputs["mixed_features"]),
        "transformer_tokens": to_2d(stage_outputs["transformer_tokens"]).T,
        "reduced_features": to_2d(stage_outputs["reduced_features"]),
        "tcn_features": to_2d(stage_outputs["tcn_features"]),
        "logits": to_2d(stage_outputs["logits"]),
    }
    compat_stage_arrays = {
        "raw_signal": original_stage_arrays["raw_signal"],
        "conv_features": original_stage_arrays["conv_features"],
        "shallow_routed": original_stage_arrays["mixed_features"],
        "mid_routed": original_stage_arrays["transformer_tokens"],
        "deep_routed": original_stage_arrays["reduced_features"],
        "tcn_features": original_stage_arrays["tcn_features"],
        "final_routed": original_stage_arrays["logits"],
    }

    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    meta = {
        "checkpoint": str(args.ckpt),
        "config": str(args.config),
        "subject_id": subject_id,
        "sample_index": sample_index,
        "true_label": true_label,
        "pred_label": pred_label,
        "true_name": class_names[true_label],
        "pred_name": class_names[pred_label],
        "confidence": float(probs[pred_label]),
        "probabilities": probs.tolist(),
        "channel_names": BCIC2A_CHANNEL_NAMES,
    }

    with open(out_dir / "prediction_summary.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.stage_layout == "single_exit":
        stage_arrays = original_stage_arrays
        combined_titles = [
            ("raw_signal", "Raw EEG"),
            ("conv_features", "Conv Features"),
            ("mixed_features", "Mixed Features"),
            ("transformer_tokens", "Transformer Tokens"),
            ("reduced_features", "Reduced Features"),
            ("tcn_features", "TCN Features"),
            ("logits", "Logits"),
        ]
        np.savez_compressed(out_dir / "stage_features.npz", **original_stage_arrays)
    else:
        stage_arrays = compat_stage_arrays
        combined_titles = [
            ("raw_signal", "Raw EEG"),
            ("conv_features", "Conv Features"),
            ("shallow_routed", "Shallow Exit Features"),
            ("mid_routed", "Mid Exit Features"),
            ("deep_routed", "Deep Exit Features"),
            ("tcn_features", "TCN Features"),
            ("final_routed", "Final Exit Features"),
        ]
        np.savez_compressed(out_dir / "stage_features.npz", **compat_stage_arrays)
        np.savez_compressed(out_dir / "stage_features_original_names.npz", **original_stage_arrays)

    save_signal_plot(
        original_stage_arrays["raw_signal"],
        f"Raw EEG | sample={sample_index} | true={meta['true_name']} | pred={meta['pred_name']}",
        out_dir / "raw_signal.png",
    )
    save_combined_figure(stage_arrays, combined_titles, out_dir / "all_stages_heatmap.png")
    for key, arr in stage_arrays.items():
        save_heatmap(arr, key.replace("_", " ").title(), out_dir / f"{key}.png")
    if args.stage_layout != "single_exit":
        for key, arr in original_stage_arrays.items():
            save_heatmap(arr, key.replace("_", " ").title(), out_dir / f"{key}.png")
    save_channel_importance_bar(
        importance["channel_saliency"],
        "Channel Importance by Saliency",
        out_dir / "channel_importance_saliency.png",
    )
    save_channel_importance_bar(
        importance["channel_grad_times_input"],
        "Channel Importance by Gradient x Input",
        out_dir / "channel_importance_gradxinput.png",
    )
    save_channel_topomap(
        importance["channel_saliency"],
        "Approximate Scalp Map by Saliency",
        out_dir / "channel_topomap_saliency.png",
    )
    save_channel_topomap(
        importance["channel_grad_times_input"],
        "Approximate Scalp Map by Gradient x Input",
        out_dir / "channel_topomap_gradxinput.png",
    )
    with open(out_dir / "input_importance.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "predicted_class": class_names[importance["pred"]],
                "predicted_index": importance["pred"],
                "channel_names": BCIC2A_CHANNEL_NAMES,
                "channel_saliency": importance["channel_saliency"].tolist(),
                "channel_grad_times_input": importance["channel_grad_times_input"].tolist(),
                "time_saliency": importance["time_saliency"].tolist(),
                **(
                    {
                        "compatibility_aliases": {
                            "shallow_routed": "mixed_features",
                            "mid_routed": "transformer_tokens",
                            "deep_routed": "reduced_features",
                            "final_routed": "logits",
                        },
                    }
                    if args.stage_layout != "single_exit"
                    else {}
                ),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    write_notes(meta, stage_arrays, importance, out_dir / "interpretation_notes_zh.md", args.stage_layout == "single_exit")
    write_notes_en(meta, stage_arrays, importance, out_dir / "interpretation_notes.md", args.stage_layout == "single_exit")
    save_paper_overview_figure(out_dir)

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
