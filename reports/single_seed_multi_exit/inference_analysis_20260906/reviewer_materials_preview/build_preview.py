"""Generate an author-facing analysis preview without changing the manuscript."""

import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import html
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
import numpy as np
from scipy.special import log_softmax
from scipy.stats import permutation_test, rankdata, spearmanr
from sklearn.metrics import log_loss


OUT = Path(__file__).resolve().parent
INPUT = OUT.parent
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
DATASETS = ("bcic2a", "bcic2b")
DATASET_NAMES = {"bcic2a": "IV-2a", "bcic2b": "IV-2b"}
EXITS = ("shallow", "mid", "deep", "final", "ensemble")
EXIT_NAMES = {"shallow": "S", "mid": "M", "deep": "D",
              "final": "F", "ensemble": "Fusion"}
COLORS = {"shallow": "#8a91a1", "mid": "#6994af", "deep": "#9b80a8",
          "final": "#b47739", "ensemble": "#237f7a"}
CORR_METRICS = ("error_rate", "ece_15", "nll", "brier_sum")
CORR_LABELS = ("Error rate", "ECE", "NLL", "Brier")
PARAMS = tuple([("s", k) for k in EXITS[:3]] + [("w", k) for k in EXITS[:4]])
WIDTH = 183 / 25.4
ARTIFACTS = []
QA = []
PDF = None

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.3,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_two_decimals(value, signed=False):
    rounded = Decimal(str(round(float(value), 10))).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP)
    return format(rounded, "+.2f" if signed else ".2f")


def reliability(y, p):
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    groups = np.minimum((confidence * 15).astype(int), 14)
    rows = []
    for i in range(15):
        selected = groups == i
        n = int(selected.sum())
        rows.append({
            "bin": i, "lower": i / 15, "upper": (i + 1) / 15, "n_trials": n,
            "mean_confidence": float(confidence[selected].mean()) if n else None,
            "accuracy": float(correct[selected].mean()) if n else None,
            "trial_fraction": n / len(y),
        })
    ece = sum(r["trial_fraction"] * abs(r["accuracy"] - r["mean_confidence"])
              for r in rows if r["n_trials"])
    return rows, ece


def load_data():
    assert read_json(INPUT / "verification.json")["complete"]
    data, metrics, parameters, comparisons = {}, [], [], []
    for dataset in DATASETS:
        data[dataset] = []
        for subject in range(1, 10):
            folder = INPUT / "results" / dataset / f"subject_{subject}"
            report = read_json(folder / "report.json")
            with np.load(folder / "predictions.npz", allow_pickle=False) as archive:
                arrays = {k: archive[k].copy() for k in archive.files}
            y = arrays["y_true"]
            hp = report["checkpoint_hparams"]
            clamped_s = np.clip(arrays["exit_log_vars"],
                                hp.get("exit_log_var_min", -3),
                                hp.get("exit_log_var_max", 3))
            assert len(clamped_s) == 3
            weights = arrays["exit_weights"]
            priors = np.array([hp.get(f"ensemble_prior_{k}", v)
                               for k, v in zip(EXITS[:3], (0.45, 0.9, 1.15))])
            raw = np.r_[np.exp(-clamped_s) * priors, hp.get("teacher_ensemble_weight", 3)]
            np.testing.assert_allclose(weights, raw / raw.sum(), atol=1e-7)
            by_exit = {}
            for key in EXITS:
                p = arrays[f"{key}_probs"]
                z = arrays[f"{key}_logits"].astype(np.float64)
                np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-10)
                bins, ece = reliability(y, p)
                nll = float(-log_softmax(z, axis=1)[np.arange(len(y)), y].mean())
                np.testing.assert_allclose(nll, log_loss(y, p, labels=np.arange(p.shape[1])))
                row = {
                    "dataset": DATASET_NAMES[dataset], "subject": subject,
                    "exit": EXIT_NAMES[key], "n_trials": len(y),
                    "accuracy_pct": float(100 * (p.argmax(1) == y).mean()),
                    "error_rate": float((p.argmax(1) != y).mean()),
                    "ece_15": ece, "nll": nll,
                    "brier_sum": float(np.square(p - np.eye(p.shape[1])[y]).sum(1).mean()),
                }
                old = next(r for r in report["metrics"] if r["exit"] == key)
                for name in ("accuracy_pct", "ece_15", "nll", "brier_sum"):
                    np.testing.assert_allclose(row[name], old[name], atol=1e-10)
                by_exit[key] = row
                metrics.append(row)
            for i, key in enumerate(EXITS[:4]):
                parameters.append({
                    "dataset": DATASET_NAMES[dataset], "subject": subject,
                    "exit": EXIT_NAMES[key],
                    "s_raw": float(arrays["exit_log_vars"][i]) if i < 3 else None,
                    "s_effective": float(clamped_s[i]) if i < 3 else None,
                    "precision_exp_minus_s": float(np.exp(-clamped_s[i])) if i < 3 else None,
                    "fixed_prior_or_final_coefficient": float(priors[i]) if i < 3 else float(raw[-1]),
                    "normalized_fusion_weight": float(weights[i]),
                })
            f_ok = arrays["final_probs"].argmax(1) == y
            fused_ok = arrays["ensemble_probs"].argmax(1) == y
            comparisons.append({
                "dataset": DATASET_NAMES[dataset], "subject": subject, "n_trials": len(y),
                "final_accuracy_pct": by_exit["final"]["accuracy_pct"],
                "fusion_accuracy_pct": by_exit["ensemble"]["accuracy_pct"],
                "accuracy_change_pp": float(100 * (int(fused_ok.sum()) - int(f_ok.sum())) / len(y)),
                "rescued_errors": int((~f_ok & fused_ok).sum()),
                "introduced_errors": int((f_ok & ~fused_ok).sum()),
                "ece_change": by_exit["ensemble"]["ece_15"] - by_exit["final"]["ece_15"],
                "nll_change": by_exit["ensemble"]["nll"] - by_exit["final"]["nll"],
                "brier_change": by_exit["ensemble"]["brier_sum"] - by_exit["final"]["brier_sum"],
            })
            data[dataset].append({
                "subject": subject, "arrays": arrays, "metrics": by_exit,
                "s": dict(zip(EXITS[:3], map(float, clamped_s))),
                "w": dict(zip(EXITS[:4], map(float, weights))),
            })
    write_csv(TABLES / "subject_metrics.csv", metrics)
    write_csv(TABLES / "learned_parameters_and_weights.csv", parameters)
    write_csv(TABLES / "subject_fusion_gains.csv", comparisons)
    return data, comparisons


def calibration_rows(data):
    rows = []
    for dataset in DATASETS:
        for key in EXITS:
            row = {"dataset": DATASET_NAMES[dataset], "exit": EXIT_NAMES[key], "n_subjects": 9}
            for metric in ("accuracy_pct", "ece_15", "nll", "brier_sum"):
                row[metric] = float(np.mean([s["metrics"][key][metric] for s in data[dataset]]))
            rows.append(row)
    write_csv(TABLES / "calibration_summary.csv", rows)
    return rows


def correlation_analysis(data):
    # The 56-test family is fixed before results are inspected: 2 datasets x 7 x 4.
    rows = []
    for dataset in DATASETS:
        for kind, key in PARAMS:
            x = np.array([subject[kind][key] for subject in data[dataset]])
            xr = rankdata(x)
            xr = (xr - xr.mean()) / np.linalg.norm(xr - xr.mean())
            for metric in CORR_METRICS:
                y = np.array([subject["metrics"][key][metric] for subject in data[dataset]])
                yr = rankdata(y)
                yr = (yr - yr.mean()) / np.linalg.norm(yr - yr.mean())
                rho = float(xr @ yr)
                np.testing.assert_allclose(rho, spearmanr(x, y).statistic, atol=1e-12)

                def absolute_correlation(permuted_x, axis=-1):
                    return np.abs(np.sum(permuted_x * yr, axis=axis))

                result = permutation_test(
                    (xr,), absolute_correlation, permutation_type="pairings",
                    vectorized=True, n_resamples=np.inf, batch=8192, alternative="greater")
                assert len(result.null_distribution) == 362880
                rows.append({
                    "dataset": DATASET_NAMES[dataset], "parameter": f"{kind}({EXIT_NAMES[key]})",
                    "exit": EXIT_NAMES[key], "metric": metric, "n_subjects": 9,
                    "spearman_rho": rho, "p_exact_two_sided": float(result.pvalue),
                    "n_permutations": 362880,
                    "supportive_direction": "positive" if kind == "s" else "negative",
                })
            print(f"CORRELATIONS {DATASET_NAMES[dataset]} {kind}({EXIT_NAMES[key]})", flush=True)
    p = np.array([r["p_exact_two_sided"] for r in rows])
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    for position, index in enumerate(order):
        rows[index]["p_holm_56"] = float(adjusted[position])
    write_csv(TABLES / "weight_reliability_correlations.csv", rows)
    return rows


def title_and_note(fig, title, note):
    fig.suptitle(title, x=0.08, ha="left", y=0.975, fontsize=11, fontweight="bold")
    fig.text(0.08, 0.025, note, ha="left", va="bottom", fontsize=7, color="#444444")


def save(fig, name, caption_en, caption_zh):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        point_labels = [a for a in ax.texts if getattr(a, "_subject_point_label", False)]
        if not point_labels:
            continue
        scale = fig.dpi / 72
        points = [ax.transData.transform(a.xy) for a in point_labels]
        point_boxes = [Bbox.from_bounds(x - 3 * scale, y - 3 * scale, 6 * scale, 6 * scale)
                       for x, y in points]
        occupied = []
        for label in point_labels:
            for dx, dy in [(5, 5), (5, -10), (-10, 5), (-10, -10),
                           (0, 12), (0, -15), (12, 3), (-16, 3)]:
                label.set_position((dx, dy))
                bbox = label.get_window_extent(renderer).expanded(1.15, 1.15)
                within = ax.bbox.contains(bbox.x0, bbox.y0) and ax.bbox.contains(bbox.x1, bbox.y1)
                if within and not any(bbox.overlaps(b) for b in occupied + point_boxes):
                    occupied.append(bbox)
                    break
            else:
                raise RuntimeError(f"Could not place subject label in {name}")
    fig.canvas.draw()
    outside = []
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        bbox = artist.get_window_extent(renderer)
        if bbox.width and bbox.height and (
            bbox.x0 < -1 or bbox.y0 < -1 or
            bbox.x1 > fig.bbox.width + 1 or bbox.y1 > fig.bbox.height + 1
        ):
            # Matplotlib keeps invisible tick-label objects outside view limits.
            if artist.get_clip_on():
                continue
            outside.append(artist.get_text())
    if outside:
        raise RuntimeError(f"Text outside canvas in {name}: {outside}")
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=300)
    PDF.savefig(fig)
    ARTIFACTS.append({"name": name, "caption_en": caption_en, "caption_zh": caption_zh})
    QA.append({"figure": name, "canvas_text_bounds": "passed",
               "panels": len(fig.axes), "width_mm": round(fig.get_figwidth() * 25.4, 2),
               "height_mm": round(fig.get_figheight() * 25.4, 2)})
    plt.close(fig)
    print("FIGURE", name, flush=True)


def plot_calibration(rows):
    fig, ax = plt.subplots(figsize=(WIDTH, 4.2))
    ax.axis("off")
    title_and_note(
        fig, "Calibration and discrimination across exits",
        "Equal-weight subject means; n = 9 per dataset. Lower ECE, NLL and Brier are better.\n"
        "F is the final exit of the same multi-exit model, not an independently trained baseline.")
    cols = ["Dataset", "Output", "Accuracy (%)", "ECE", "NLL", "Brier"]
    body = [[r["dataset"], r["exit"], f'{r["accuracy_pct"]:.2f}',
             f'{r["ece_15"]:.4f}', f'{r["nll"]:.4f}', f'{r["brier_sum"]:.4f}'] for r in rows]
    table = ax.table(cellText=body, colLabels=cols, loc="center", cellLoc="center",
                     colWidths=[0.14, 0.14, 0.20, 0.17, 0.17, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.75)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dfe0")
        cell.set_linewidth(0.45)
        if r == 0:
            cell.set_facecolor("#eaf0f2")
            cell.get_text().set_fontweight("bold")
        elif r in (5, 10):
            cell.set_facecolor("#e8f2ef")
        else:
            cell.set_facecolor("white")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.21, top=0.84)
    save(fig, "01_calibration_summary",
         "Calibration and discrimination. Each entry is the mean of nine subject-level metrics. "
         "ECE uses 15 equal-width top-confidence bins; NLL uses hard labels and natural logarithms; "
         "Brier is summed over classes before averaging over trials. Fusion rows are shaded only "
         "to identify the evaluated output, not to indicate statistical significance.",
         "校准与分类性能表。每项为9位被试各自指标的等权平均。ECE使用15个等宽置信度分箱；"
         "NLL使用真实硬标签和自然对数；Brier先在类别维度求和，再对试次取平均。"
         "浅色背景仅用于标识融合输出，不代表统计显著。2a与2b的类别数不同，不据此直接比较任务难度。")


def pooled_reliability(data):
    pooled, rows = {}, []
    for dataset in DATASETS:
        pooled[dataset] = {}
        y = np.concatenate([s["arrays"]["y_true"] for s in data[dataset]])
        for key in EXITS:
            p = np.concatenate([s["arrays"][f"{key}_probs"] for s in data[dataset]])
            bins, ece = reliability(y, p)
            pooled[dataset][key] = {"bins": bins, "n": len(y), "ece": ece}
            rows.extend(dict(dataset=DATASET_NAMES[dataset], exit=EXIT_NAMES[key], **r) for r in bins)
    write_csv(TABLES / "reliability_bins_pooled.csv", rows)
    return pooled


def reliability_curve(ax, series, key):
    bins = series["bins"]
    xx = np.array([r["mean_confidence"] if r["n_trials"] else np.nan for r in bins])
    yy = np.array([r["accuracy"] if r["n_trials"] else np.nan for r in bins])
    ax.plot(xx, yy, marker="o" if key == "ensemble" else "s", markersize=3,
            color=COLORS[key], label=EXIT_NAMES[key])
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Observed accuracy")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(alpha=0.18, linewidth=0.5)


def plot_reliability(pooled):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.9),
                             gridspec_kw={"height_ratios": [1.65, 1]})
    title_and_note(fig, "Reliability of final-exit and fused predictions",
                   "All test trials pooled for visualization. Empty bins are not connected.\n"
                   "Calibration-table ECE is a subject mean, not the pooled-diagram ECE.")
    for j, dataset in enumerate(DATASETS):
        ax = axes[0, j]
        ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=0.9, label="Perfect calibration")
        for key in ("final", "ensemble"):
            reliability_curve(ax, pooled[dataset][key], key)
        ax.set_title(f"{chr(97+j)}  {DATASET_NAMES[dataset]} | {pooled[dataset]['final']['n']:,} trials",
                     loc="left")
        ax.legend(loc="upper left")
        bottom = axes[1, j]
        for key in ("final", "ensemble"):
            values = [r["trial_fraction"] * 100 for r in pooled[dataset][key]["bins"]]
            bottom.stairs(values, np.linspace(0, 1, 16), color=COLORS[key],
                          linewidth=1.4, label=EXIT_NAMES[key])
        bottom.set(xlim=(0, 1), ylim=(0, 85), xlabel="Confidence", ylabel="Trials (%)")
        bottom.set_title(f"{chr(99+j)}  Confidence distribution", loc="left")
        bottom.grid(axis="y", alpha=0.18, linewidth=0.5)
    fig.subplots_adjust(left=0.095, right=0.98, top=0.86, bottom=0.18,
                        wspace=0.35, hspace=0.6)
    save(fig, "02_reliability_overview",
         "Reliability diagrams and confidence distributions for F and fusion. Test trials are pooled "
         "within each dataset for descriptive visualization; the lower panels expose bin support. "
         "The diagonal denotes perfect calibration. These pooled curves are not subject-level "
         "confidence intervals, and their ECE is distinct from the subject-averaged table values.",
         "F出口与融合输出的可靠性曲线及置信度分布。每个数据集的测试试次在此图中合并，"
         "只作描述性展示；下排显示置信度分布，避免忽略稀疏分箱。对角线表示理想校准。"
         "这里没有置信区间，且合并试次后的ECE不等于校准表中按被试平均的ECE。")

    for dataset in DATASETS:
        fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 5.4))
        title_and_note(fig, f"Exit-wise reliability | {DATASET_NAMES[dataset]}",
                       "Pooled test trials; 15 equal-width confidence bins. Unoccupied bins remain empty.")
        for i, key in enumerate(EXITS):
            ax = axes.flat[i]
            ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=0.8)
            reliability_curve(ax, pooled[dataset][key], key)
            ax.set_title(f"{chr(97+i)}  {EXIT_NAMES[key]}\n"
                         f"Pooled ECE = {pooled[dataset][key]['ece']:.3f}",
                         loc="left", fontsize=8.5, linespacing=1.25)
        axes.flat[5].axis("off")
        axes.flat[5].text(
            0, 0.92,
            "All outputs shown\n\n"
            "S / M / D: auxiliary exits\nF: final exit\nFusion: weighted logits\n\n"
            f"{pooled[dataset]['final']['n']:,} test trials\n9 subjects\n\n"
            "Table ECE uses subject means.\nThis figure shows pooled ECE.",
            va="top", fontsize=8, linespacing=1.45)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.85,
                            wspace=0.48, hspace=0.65)
        save(fig, f"03_reliability_all_exits_{dataset}",
             f"All five prediction outputs on {DATASET_NAMES[dataset]}. Pooled ECE values are explicitly "
             "labeled and are not substituted for the subject-level mean calibration results.",
             f"{DATASET_NAMES[dataset]}的全部5种输出，未选择性省略任何出口。图内明确标注的是合并试次后的"
             "Pooled ECE，不能直接替换表中按被试平均的ECE。")


def plot_correlations(rows):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 5.5))
    title_and_note(fig, "Association of exit parameters with predictive reliability",
                   "Spearman rho across 9 subjects, separately for each exit and dataset.\n"
                   "s: learned log-scale; w: normalized fusion weight. Lower column metrics are better.")
    cmap = LinearSegmentedColormap.from_list("association", ["#477fa5", "#f8f8f6", "#b96849"])
    for ax, dataset in zip(axes, DATASETS):
        matrix = np.empty((7, 4))
        adjusted = np.empty_like(matrix)
        for i, (kind, key) in enumerate(PARAMS):
            for j, metric in enumerate(CORR_METRICS):
                r = next(r for r in rows if r["dataset"] == DATASET_NAMES[dataset]
                         and r["parameter"] == f"{kind}({EXIT_NAMES[key]})" and r["metric"] == metric)
                matrix[i, j] = r["spearman_rho"]
                adjusted[i, j] = r["p_holm_56"]
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(4), CORR_LABELS)
        ax.set_yticks(range(7), [f"{kind}({EXIT_NAMES[key]})" for kind, key in PARAMS])
        ax.set_title(DATASET_NAMES[dataset], loc="left")
        ax.tick_params(length=0)
        for i in range(7):
            for j in range(4):
                ax.text(j, i, f"{matrix[i,j]:+.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(matrix[i,j]) > 0.8 else "#222222")
                if adjusted[i, j] <= 0.05:
                    ax.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                           fill=False, linewidth=1.1, edgecolor="#161616"))
        ax.axhline(2.5, color="white", linewidth=4)
    fig.subplots_adjust(left=0.12, right=0.92, top=0.86, bottom=0.23, wspace=0.28)
    cax = fig.add_axes([0.3, 0.13, 0.46, 0.023])
    fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[-1, -0.5, 0, 0.5, 1])
    n_sig = sum(r["p_holm_56"] <= 0.05 for r in rows)
    save(fig, "04_weight_reliability_correlations",
         "Exit-wise association across nine subjects. The upper rows show the effective clamped "
         "learned log-scales for S, M and D; the lower rows show normalized fusion weights. F has "
         "no separately learned log-scale in these checkpoints. Each association uses nine "
         "subject-level pairs. Exact permutation tests enumerate all 9! pairings using absolute "
         "Spearman correlation as the two-sided statistic. Holm adjustment covers all 56 tests. "
         f"Outlined cells pass adjusted P <= 0.05 ({n_sig}/56). "
         "A positive log-scale association or a negative weight association is directionally "
         "consistent with the reliability interpretation; neither establishes causality or "
         "sample-level uncertainty. Normalized weights also depend on fixed priors and other exits.",
         "以被试为单位的出口相关分析。上3行为S/M/D实际使用的学习对数尺度，下4行为归一化融合权重。"
         "这些checkpoint中F没有独立学习的s参数，因此不构造s(F)。每项相关均使用9对被试数据，"
         "不是将全局权重重复到每个试次。P值使用全部9!配对排列的精确双侧检验，56项比较统一做Holm校正。"
         f"带边框单元格表示校正后P≤0.05，本次为{n_sig}/56项。"
         "s与误差指标正相关、w与误差指标负相关才与其可靠性解释方向一致；"
         "但相关性不等于因果机制或试次级不确定性，归一化权重还受固定先验和其他出口影响。")


def plot_gains(comparisons):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 5.0), sharex=True)
    title_and_note(fig, "Subject-level change from final exit to fusion",
                   "Each bar is one subject's paired accuracy change. Zero means unchanged accuracy.\n"
                   "The reference is F within the same multi-exit model, not an independent single-exit baseline.")
    counts = {}
    for ax, dataset in zip(axes, DATASETS):
        selected = [r for r in comparisons if r["dataset"] == DATASET_NAMES[dataset]]
        values = np.array([r["accuracy_change_pp"] for r in selected])
        counts[dataset] = {
            "improved": int((values > 1e-10).sum()),
            "unchanged": int((np.abs(values) <= 1e-10).sum()),
            "decreased": int((values < -1e-10).sum()),
            "mean_gain_pp": float(values.mean()),
            "rescued_errors": sum(r["rescued_errors"] for r in selected),
            "introduced_errors": sum(r["introduced_errors"] for r in selected),
        }
        ax.axvline(0, color="#555555", linewidth=0.8, zorder=0)
        colors = ["#237f7a" if value >= 0 else "#b96849" for value in values]
        ax.barh(np.arange(9), values, color=colors, height=0.55)
        for i, value in enumerate(values):
            if abs(value) < 1e-10:
                ax.plot(0, i, "o", color="#777777", markersize=3)
            ax.text(value + (0.12 if value >= 0 else -0.12), i,
                    display_two_decimals(value, signed=True) if abs(value) > 1e-10 else "0.00",
                    va="center", ha="left" if value >= 0 else "right", fontsize=8)
        ax.set_yticks(range(9), [f"S{i}" for i in range(1, 10)])
        ax.invert_yaxis()
        ax.set(xlim=(-1.8, 3.9), xlabel="Fusion - F accuracy (percentage points)")
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        ax.set_title(f"{DATASET_NAMES[dataset]} | Mean {values.mean():+.2f} pp", loc="left")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.22, wspace=0.30)
    save(fig, "05_subject_fusion_gains",
         "Paired changes in accuracy for all nine subjects per dataset. Bars show individual subjects, "
         "not averages with omitted error bars. The reference is the final exit of the same fitted "
         "multi-exit model. IV-2a has eight improvements and one tie; IV-2b has six improvements, "
         "one tie and two decreases. Corrections and newly introduced errors are retained in the source table.",
         "每个数据集9位被试的配对准确率变化，全部被试均保留。每条柱代表一位被试，不是均值柱，"
         "所以不添加误差线。参照为同一多出口模型的F出口。IV-2a有8人提升、1人持平；"
         "IV-2b有6人提升、1人持平、2人下降。纠正错误与新增错误的数量保留在配套表格中。")
    return counts


def plot_logscale_scatter(data, correlations):
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 7.5))
    title_and_note(fig, "Learned exit log-scales and test error rates",
                   "Every point is one subject; labels identify subjects 1-9. No fitted trend is imposed.\n"
                   "P values and the full 56-test correction are provided in the correlation table.")
    for row, key in enumerate(EXITS[:3]):
        for column, dataset in enumerate(DATASETS):
            ax = axes[row, column]
            x = [s["s"][key] for s in data[dataset]]
            y = [100 * s["metrics"][key]["error_rate"] for s in data[dataset]]
            ax.scatter(x, y, s=20, color=COLORS[key], edgecolor="white", linewidth=0.5, zorder=3)
            for subject, xi, yi in zip(range(1, 10), x, y):
                annotation = ax.annotate(str(subject), (xi, yi), xytext=(5, 5),
                                         textcoords="offset points", fontsize=6.5)
                annotation._subject_point_label = True
            ax.margins(x=0.24, y=0.2)
            r = next(r for r in correlations if r["dataset"] == DATASET_NAMES[dataset]
                     and r["parameter"] == f"s({EXIT_NAMES[key]})" and r["metric"] == "error_rate")
            ax.set_title(f"{DATASET_NAMES[dataset]} | {EXIT_NAMES[key]} | rho = {r['spearman_rho']:+.2f}",
                         loc="left")
            ax.set(xlabel=f"Learned log-scale s({EXIT_NAMES[key]})", ylabel="Test error (%)")
            ax.grid(alpha=0.18, linewidth=0.5)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.9, bottom=0.15, hspace=0.63, wspace=0.38)
    save(fig, "06_logscale_error_scatter",
         "Raw subject-level points underlying log-scale/error-rate correlations. All auxiliary "
         "exits and both datasets are shown without selecting the strongest associations. Labels "
         "are subject IDs. A fitted regression line is omitted because the displayed statistic is "
         "a rank correlation and the evidence is exploratory.",
         "学习对数尺度与错误率相关性的原始散点。两数据集、三个辅助出口全部展示，"
         "不只挑选最强相关。数字表示被试编号。这里展示的是秩相关，且属于探索性分析，"
         "因此没有强行拟合直线，也不能据此直接推断不确定性的生理或因果来源。")


def plot_complementarity(data):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 6.5))
    title_and_note(fig, "Prediction differences and shared errors across exits",
                   "Equal-weight subject means. Disagreement is not, by itself, evidence of useful complementarity.\n"
                   "All six exit pairs and all subjects are included; diagonal cells are not comparisons.")
    rows = []
    matrices = {}
    for dataset in DATASETS:
        disagreement = np.zeros((9, 4, 4))
        error_corr = np.full((9, 4, 4), np.nan)
        for n, subject in enumerate(data[dataset]):
            y = subject["arrays"]["y_true"]
            predictions = [subject["arrays"][f"{key}_probs"].argmax(1) for key in EXITS[:4]]
            for i in range(4):
                for j in range(i + 1, 4):
                    err_i, err_j = predictions[i] != y, predictions[j] != y
                    d = float(100 * (predictions[i] != predictions[j]).mean())
                    r = (float(np.corrcoef(err_i, err_j)[0, 1])
                         if err_i.std() > 0 and err_j.std() > 0 else None)
                    disagreement[n, i, j] = disagreement[n, j, i] = d
                    error_corr[n, i, j] = error_corr[n, j, i] = np.nan if r is None else r
                    rows.append({"dataset": DATASET_NAMES[dataset], "subject": subject["subject"],
                                 "exit_1": EXIT_NAMES[EXITS[i]], "exit_2": EXIT_NAMES[EXITS[j]],
                                 "prediction_disagreement_pct": d, "binary_error_correlation": r})
        mean_corr = np.zeros((4, 4))
        for i in range(4):
            for j in range(i + 1, 4):
                assert np.isfinite(error_corr[:, i, j]).all()
                mean_corr[i, j] = mean_corr[j, i] = error_corr[:, i, j].mean()
        matrices[dataset] = (disagreement.mean(0), mean_corr)
    for column, dataset in enumerate(DATASETS):
        for row, values in enumerate(matrices[dataset]):
            ax = axes[row, column]
            mask = np.eye(4, dtype=bool)
            shown = np.ma.masked_array(values, mask=mask)
            image = ax.imshow(shown, cmap="Blues" if row == 0 else "Greys",
                              vmin=0, vmax=16 if row == 0 else 1)
            ax.set_xticks(range(4), ["S", "M", "D", "F"])
            ax.set_yticks(range(4), ["S", "M", "D", "F"])
            ax.tick_params(length=0)
            ax.set_title(f"{DATASET_NAMES[dataset]} | " +
                         ("Disagreement (%)" if row == 0 else "Error correlation"), loc="left")
            for i in range(4):
                for j in range(4):
                    if i == j:
                        ax.text(j, i, "-", ha="center", va="center", color="#888888")
                    else:
                        ax.text(j, i, f"{values[i,j]:.1f}" if row == 0 else f"{values[i,j]:.2f}",
                                ha="center", va="center", fontsize=9,
                                color="white" if values[i,j] > (9 if row == 0 else 0.6) else "#222222")
            fig.colorbar(image, ax=ax, shrink=0.8, fraction=0.055, pad=0.035)
    fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.15, hspace=0.35, wspace=0.30)
    write_csv(TABLES / "exit_complementarity.csv", rows)
    save(fig, "07_exit_complementarity",
         "Mean subject-level prediction disagreement and correlation of binary error indicators "
         "for all exit pairs. These summaries describe differences and shared failures, not causal "
         "evidence that a particular training component created complementary representations. "
         "No undefined subject correlation was excluded.",
         "各出口对的预测分歧比例及二元错误指示变量相关性，先在每位被试内计算，再等权平均。"
         "它们说明出口预测存在差异及共享错误，但不能仅凭此证明某个训练模块产生了互补表征。"
         "所有出口对均展示，本次不存在因相关系数未定义而剔除被试的情况。")


def build_author_preview(calibration, correlations, counts):
    sig = [r for r in correlations if r["p_holm_56"] <= 0.05]
    sections = []
    for i, artifact in enumerate(ARTIFACTS, 1):
        sections.append(
            f'<section><h2>{i}. {html.escape(artifact["caption_zh"].split("。")[0])}</h2>'
            f'<img src="figures/{artifact["name"]}.png" alt="{html.escape(artifact["name"])}">'
            f'<p>{html.escape(artifact["caption_zh"])}</p>'
            f'<details><summary>English caption</summary><p>{html.escape(artifact["caption_en"])}</p></details>'
            f'<p class="files"><a href="figures/{artifact["name"]}.pdf">PDF</a> · '
            f'<a href="figures/{artifact["name"]}.svg">SVG</a></p></section>')
    table_rows = "".join(
        f'<tr><td>{r["dataset"]}</td><td>{r["exit"]}</td><td>{r["accuracy_pct"]:.2f}%</td>'
        f'<td>{r["ece_15"]:.4f}</td><td>{r["nll"]:.4f}</td><td>{r["brier_sum"]:.4f}</td></tr>'
        for r in calibration)
    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>审稿补充分析预览</title>
<style>
body{{margin:0;background:#fff;color:#24282a;font:16px/1.75 "Microsoft YaHei",Arial,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:32px 24px 64px}}
h1{{font-size:28px;margin:0 0 8px}}h2{{font-size:21px;margin:0 0 20px}}
p{{margin:12px 0}}.note{{padding:14px 0;border-top:2px solid #237f7a;border-bottom:1px solid #dce2e4}}
section{{padding:32px 0;border-top:1px solid #dce2e4}}
img{{display:block;width:100%;height:auto;max-width:900px;margin:0 auto}}
table{{width:100%;border-collapse:collapse;font-size:15px;margin:24px 0}}
td,th{{padding:9px 12px;border-bottom:1px solid #dce2e4;text-align:right}}
td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){{text-align:left}}
th{{background:#edf2f4}}a{{color:#176862}}.files,summary{{font-size:14px}}
.scroll{{overflow-x:auto}}details p{{font-family:Arial,sans-serif}}
@media(max-width:600px){{main{{padding:20px 12px}}h1{{font-size:24px}}h2{{font-size:18px}}td,th{{padding:7px}}}}
</style></head><body><main>
<h1>审稿补充分析预览</h1>
<p>这是独立的图表与统计结果预览，没有修改论文或回复信。</p>
<div class="note">每个数据集9位被试。表中指标按被试等权平均，可靠性曲线按图注合并试次展示。
F指同一多出口模型内的最终出口，不是独立训练的单出口基线。</div>
<h2 style="margin-top:24px">先看结论</h2>
<p>融合相较F在两个数据集上提高了平均准确率，并降低平均NLL和Brier。
IV-2a的ECE下降，IV-2b的ECE略升，因此不能写“所有校准指标均改善”。</p>
<p>IV-2a有8位被试提升、1位持平；IV-2b有6位提升、1位持平、2位下降。
相关分析按出口分别使用9对被试数据；56项比较做Holm校正后，{len(sig)}项达到P≤0.05。
相关结果不能单独确立试次级不确定性，也不能排除被试难度等共同影响。</p>
<div class="scroll"><table><thead><tr><th>数据集</th><th>输出</th><th>准确率</th>
<th>ECE</th><th>NLL</th><th>Brier</th></tr></thead><tbody>{table_rows}</tbody></table></div>
<p><a href="reviewer_analysis_preview.pdf">打开合并PDF图册</a> ·
<a href="tables/weight_reliability_correlations.csv">完整相关分析</a> ·
<a href="tables/subject_metrics.csv">逐被试指标</a> ·
<a href="tables/subject_fusion_gains.csv">逐被试收益</a></p>
{''.join(sections)}
<section><h2>可回应的范围</h2>
<p>校准表、可靠性曲线和权重相关分析可用于回应出口可靠性的质疑；
逐被试收益和共享错误分析可支持收益不一致与局限性讨论。
这里没有增加参数匹配训练、新方法对照、在线闭环或跨数据集迁移实验，也没有计算跨运行波动。</p>
<p>数值推理采用FP32。校准分析中未做额外温度校准或测试集拟合。
相关检验是探索性的，被试内的不同出口未当作相互独立的样本合并检验。</p>
</section></main></body></html>"""
    (OUT / "preview.html").write_text(doc, encoding="utf-8")
    captions = "\n\n".join(
        f'{a["name"]}\nEN: {a["caption_en"]}\n中文: {a["caption_zh"]}' for a in ARTIFACTS)
    (OUT / "figure_captions.txt").write_text(captions, encoding="utf-8")
    write_json(OUT / "analysis_summary.json", {
        "aggregation": "9 subjects per dataset; equal-weight subject means unless pooled is stated",
        "comparison": "Fusion versus F within the same multi-exit model",
        "calibration": calibration, "subject_gains": counts,
        "correlations_holm_significant": sig,
        "correlations_total": len(correlations),
        "n_test_trials": {"IV-2a": 2592, "IV-2b": 2840},
        "no_manuscript_changes": True,
    })


def main():
    global PDF
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    write_json(OUT / "figure_contract.json", {
        "core_conclusion": "Evaluate output calibration, between-subject parameter associations, and paired fusion gains without assuming uniform benefit.",
        "archetype": "quantitative grid",
        "backend": "Python/matplotlib",
        "width_mm": 183, "min_text_pt": 6.5,
        "exports": ["PNG", "SVG", "PDF", "CSV"],
        "statistical_units": "9 subjects per dataset for correlations and paired subject changes",
        "correlation_family": "56 tests: 2 datasets x (3 learned log-scales + 4 normalized weights) x 4 error metrics",
        "correlation_test": "Exact permutation of pairings; absolute Spearman rho; Holm adjustment across 56 tests",
        "null_assumption": "Subject-level pairings are exchangeable under independence; this is not a causal test.",
        "reliability_diagrams": "All test trials pooled, descriptive; no bin intervals or significance claims",
        "exclusions": "None",
        "baseline_boundary": "The final exit is not an independently trained single-exit model",
        "public_label_policy": "No random-initialization identifier or run count in figures",
        "sources": ["provided predicted probabilities", "provided checkpoint parameters"],
    })
    data, comparisons = load_data()
    calibration = calibration_rows(data)
    pooled = pooled_reliability(data)
    correlations = correlation_analysis(data)
    with PdfPages(OUT / "reviewer_analysis_preview.pdf") as pdf:
        PDF = pdf
        plot_calibration(calibration)
        plot_reliability(pooled)
        plot_correlations(correlations)
        counts = plot_gains(comparisons)
        plot_logscale_scatter(data, correlations)
        plot_complementarity(data)
    build_author_preview(calibration, correlations, counts)
    write_json(OUT / "qa_computational.json", {
        "input_verification": "complete; 18 models and 5432 test trials",
        "metric_recomputation": "all match independently recomputed saved values",
        "weight_reconstruction": "all match exp(-s), fixed priors, and final coefficient",
        "permutation_test": "362880 exact pairings for each of 56 associations",
        "p_value_family": "Holm across all 56 associations, no favorable subset",
        "between_subject_error_bars": "not drawn; individual subjects retained in paired plots",
        "all_exits_and_subjects_retained": True,
        "figure_checks": QA,
        "manuscript_untouched": True,
    })
    write_json(OUT / "source_fingerprints.json", {
        "generator_sha256": file_hash(Path(__file__)),
        "input_summary_sha256": file_hash(INPUT / "summary.json"),
        "input_verification_sha256": file_hash(INPUT / "verification.json"),
    })
    print("COMPLETE", flush=True)
    print(json.dumps({"figures": len(ARTIFACTS),
                      "holm_significant": sum(r["p_holm_56"] <= 0.05 for r in correlations)},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
