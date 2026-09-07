"""Re-evaluate archived seed-0 TCFormer checkpoints without training."""

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback
import zipfile

import numpy as np
from scipy.special import log_softmax
from sklearn.metrics import cohen_kappa_score, confusion_matrix, log_loss
import torch
from torch.utils.data import DataLoader
import yaml


ROOT = Path(__file__).resolve().parent
RUNS = {
    "bcic2a": "TCFormer_bcic2a_seed-0_aug-True_GPU0_20260503_1331",
    "bcic2b": "TCFormer_bcic2b_seed-0_aug-True_GPU0_20260507_183904",
}
EXITS = ("shallow", "mid", "deep", "final", "ensemble")
VERSION = 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False),
                    encoding="utf-8")
    os.replace(temp, path)


def write_csv(path, rows):
    if not rows:
        return
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def metrics(y, logits, n_bins=15):
    """Top-label ECE, hard-label NLL, and sum-over-classes Brier."""
    lp = log_softmax(np.asarray(logits, dtype=np.float64), axis=1)
    p = np.exp(lp)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-10)
    predicted = p.argmax(axis=1)
    correct = predicted == y
    confidence = p.max(axis=1)
    assignment = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    bins = []
    ece = 0.0
    for index in range(n_bins):
        mask = assignment == index
        count = int(mask.sum())
        acc = float(correct[mask].mean()) if count else None
        conf = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(acc - conf)
        bins.append({
            "bin": index, "lower": index / n_bins,
            "upper": (index + 1) / n_bins, "n": count,
            "accuracy": acc, "confidence": conf,
        })
    nll = float(-lp[np.arange(len(y)), y].mean())
    # Independent check against the library's hard-label log loss.
    np.testing.assert_allclose(nll, log_loss(y, p, labels=np.arange(p.shape[1])),
                               rtol=1e-6, atol=1e-7)
    target = np.eye(p.shape[1])[y]
    return {
        "n_trials": len(y), "accuracy_pct": float(100 * correct.mean()),
        "kappa": float(cohen_kappa_score(y, predicted)),
        "ece_15": float(ece), "nll": nll,
        "brier_sum": float(np.mean(np.sum((p - target) ** 2, axis=1))),
        "mean_confidence": float(confidence.mean()),
    }, bins, p, confusion_matrix(y, predicted, labels=np.arange(p.shape[1]))


def self_test():
    y = np.array([0, 1])
    m, bins, _, _ = metrics(y, np.zeros((2, 2)))
    np.testing.assert_allclose(
        [m["accuracy_pct"], m["ece_15"], m["nll"], m["brier_sum"]],
        [50, 0, np.log(2), 0.5])
    assert sum(b["n"] for b in bins) == 2
    m, _, _, _ = metrics(y, np.log([[0.9, 0.1], [0.1, 0.9]]))
    np.testing.assert_allclose([m["ece_15"], m["brier_sum"]], [0.1, 0.02])


def install_local_raw_resolver(project):
    """Redirect only file lookup; retain MOABB parsing and project preprocessing."""
    import moabb.datasets.bnci as bnci

    original = bnci.data_path
    found = {}
    for root in (project, Path("E:/EEG/data_cache/mne_data")):
        for path in root.rglob("*.mat"):
            if re.fullmatch(r"[AB]\d{2}[TE]\.mat", path.name):
                found.setdefault(path.name, path)
    used = {}

    def resolve(url, path=None, force_update=False, update_path=None, verbose=None):
        name = url.rsplit("/", 1)[-1]
        if name in found and not force_update:
            chosen = str(found[name])
            used[name] = {"path": chosen, "sha256": digest(chosen)}
            return [chosen]
        result = original(url, path, force_update, update_path, verbose)
        used[name] = {"path": str(result[0]), "sha256": digest(result[0])}
        return result

    bnci.data_path = resolve
    return used


def get_test_data(project, dataset, subject, preproc, used):
    from datamodules.bcic4_2a import BCICIV2a
    from datamodules.bcic4_2b import BCICIV2b

    preproc = dict(preproc, num_workers=0)
    signature = hashlib.sha256(json.dumps({
        "preprocessing": preproc,
        "sources": {
            name: digest(project / name) for name in [
                "datamodules/base.py", "datamodules/bcic4_2a.py",
                "datamodules/bcic4_2b.py", "utils/load_bcic4.py",
            ]
        },
    }, sort_keys=True).encode()).hexdigest()
    cache_dir = ROOT / "preprocessed_cache"
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"{dataset}_subject_{subject}_{signature[:12]}.npz"
    meta_path = cache.with_suffix(".json")
    if cache.exists() and meta_path.exists():
        try:
            with np.load(cache, allow_pickle=False) as values:
                x = torch.from_numpy(values["x"])
                y = torch.from_numpy(values["y"])
            return x, y, json.loads(meta_path.read_text(encoding="utf-8"))
        except zipfile.BadZipFile:
            cache.rename(cache.with_suffix(f".invalid_{time.time_ns()}"))

    dm = (BCICIV2a if dataset == "bcic2a" else BCICIV2b)(preproc, subject)
    used.clear()
    dm.setup("test")
    x, y = dm.test_dataset.tensors
    assert torch.isfinite(x).all()
    metadata = {
        "preprocessing": preproc, "cache_signature": signature,
        "train_n": len(dm.train_dataset), "test_n": len(y),
        "test_shape": list(x.shape),
        "class_names": dm.class_names, "raw_files": dict(used),
        "normalization": "Project BaseDataModule._z_scale, fitted on training data",
        "test_augmentation": False,
    }
    if hasattr(dm.dataset, "get_metadata"):
        desc = dm.dataset.get_metadata()
        metadata["session_counts_all"] = {
            str(k): int(v) for k, v in desc.groupby("session").size().items()
        }
    temp = cache.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, x=x.numpy(), y=y.numpy())
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"Cache integrity check failed: {temp}")
    os.replace(temp, cache)
    write_json(meta_path, metadata)
    return x, y, metadata


def evaluate(project, dataset, subject, device, used, precision, batch_size):
    from models.tcformer import TCFormer

    output = ROOT / "results" / dataset / f"subject_{subject}"
    output.mkdir(parents=True, exist_ok=True)
    run = project / "TCFormer result" / RUNS[dataset]
    ckpt_path = run / "checkpoints" / f"subject_{subject}_best.ckpt"
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint_hash = digest(ckpt_path)
    old_report = output / "report.json"
    if old_report.exists():
        old = json.loads(old_report.read_text(encoding="utf-8"))
        if (old.get("version") == VERSION and
                old.get("checkpoint_sha256") == checkpoint_hash and
                old.get("precision") == precision and
                (output / "predictions.npz").exists()):
            print(f"RESUME {dataset} S{subject}: completed", flush=True)
            return old
        raise RuntimeError(f"Existing incompatible output: {output}")

    start = time.perf_counter()
    with (output / "preprocessing.log").open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            x, y_tensor, data_meta = get_test_data(
                project, dataset, subject, config["preprocessing"], used)
    prep_seconds = time.perf_counter() - start
    gc.collect()
    model = TCFormer(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().to(device)
    loader = DataLoader(torch.utils.data.TensorDataset(x, y_tensor),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    collected = {key: [] for key in EXITS}
    batch_weights = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=precision == "16-mixed"):
                logits, _ = model._extract_multi_exit_outputs(model(batch_x))
                fused, _, weights = model._get_ensemble_logits(logits, device)
            logits = dict(logits, ensemble=fused)
            for key in EXITS:
                collected[key].append(logits[key].float().cpu().numpy())
            batch_weights.append(weights.cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    logits = {key: np.concatenate(value) for key, value in collected.items()}
    y = y_tensor.numpy()
    weights = batch_weights[0]
    np.testing.assert_allclose(batch_weights, np.tile(weights, (len(batch_weights), 1)))
    assert len(y) == len(logits["ensemble"])
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-6)
    rows, reliability, cm = [], [], {}
    arrays = {"y_true": y, "trial_index": np.arange(len(y)),
              "exit_weights": weights, "exit_log_vars": model.exit_log_vars.detach().cpu().numpy()}
    for key in EXITS:
        m, bins, probs, matrix = metrics(y, logits[key])
        rows.append(dict(dataset=dataset, seed=0, subject=subject, exit=key, **m))
        reliability.extend(dict(dataset=dataset, subject=subject, exit=key, **b) for b in bins)
        arrays[f"{key}_logits"] = logits[key]
        arrays[f"{key}_probs"] = probs
        cm[key] = matrix.tolist()
    final_correct = logits["final"].argmax(axis=1) == y
    fused_correct = logits["ensemble"].argmax(axis=1) == y
    comparisons = {
        "reference": "Final exit in the same multi-exit model; NOT independent single-exit baseline",
        "fusion_minus_final_pp": float(100 * (fused_correct.mean() - final_correct.mean())),
        "rescued_final_errors": int((~final_correct & fused_correct).sum()),
        "introduced_errors": int((final_correct & ~fused_correct).sum()),
        "prediction_disagreement_pct": float(100 * (
            logits["final"].argmax(1) != logits["ensemble"].argmax(1)).mean()),
    }
    archive_text = (run / "results.txt").read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"Subject {subject} =>[^\n]*Test Acc: ([0-9.]+)", archive_text)
    archived_acc = float(match[1]) * 100 if match else None
    report = {
        "version": VERSION, "dataset": dataset, "seed": 0, "subject": subject,
        "checkpoint_sha256": checkpoint_hash, "checkpoint_path": str(ckpt_path),
        "checkpoint_epoch_zero_based": checkpoint.get("epoch"),
        "strict_load": True, "precision": precision, "inference_batch_size": batch_size,
        "archived_config": config, "checkpoint_hparams": dict(checkpoint["hyper_parameters"]),
        "data": data_meta, "weights": dict(zip(EXITS[:4], map(float, weights))),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "preprocessing_seconds": prep_seconds,
        "inference_seconds_including_transfer_and_initialization": inference_seconds,
        "timing_note": "Whole test pass, not a steady-state latency benchmark",
        "archived_accuracy_pct": archived_acc,
        "accuracy_minus_archived_pp": (
            rows[-1]["accuracy_pct"] - archived_acc if archived_acc is not None else None),
        "metrics": rows, "confusion_matrices": cm,
        "fusion_vs_final": comparisons,
    }
    temp = output / "predictions.npz.tmp"
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temp, output / "predictions.npz")
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "reliability_bins.csv", reliability)
    write_json(output / "report.json", report)
    print(f"DONE {dataset} S{subject}: fused={rows[-1]['accuracy_pct']:.4f}% "
          f"F={rows[-2]['accuracy_pct']:.4f}% ECE={rows[-1]['ece_15']:.5f} "
          f"NLL={rows[-1]['nll']:.5f} Brier={rows[-1]['brier_sum']:.5f} "
          f"prep={prep_seconds:.1f}s inference={inference_seconds:.1f}s", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def summarize():
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((ROOT / "results").glob("bcic*/subject_*/report.json"))
    ]
    rows = [row for report in reports for row in report["metrics"]]
    write_csv(ROOT / "all_subject_metrics.csv", rows)
    summary = []
    for dataset in RUNS:
        for key in EXITS:
            chosen = [row for row in rows if row["dataset"] == dataset and row["exit"] == key]
            if not chosen:
                continue
            row = dict(dataset=dataset, seed=0, exit=key, n_subjects=len(chosen),
                       n_trials=sum(r["n_trials"] for r in chosen))
            for metric in ("accuracy_pct", "kappa", "ece_15", "nll", "brier_sum"):
                row[f"macro_{metric}"] = float(np.mean([r[metric] for r in chosen]))
            summary.append(row)
    write_csv(ROOT / "dataset_summary.csv", summary)
    write_csv(ROOT / "fusion_vs_final.csv", [
        dict(dataset=r["dataset"], subject=r["subject"], **r["fusion_vs_final"])
        for r in reports
    ])
    write_csv(ROOT / "exit_weights.csv", [
        dict(dataset=r["dataset"], subject=r["subject"], **r["weights"])
        for r in reports
    ])
    write_json(ROOT / "summary.json", {
        "scope": "Archived seed-0 multi-exit checkpoint inference only; no training or tuning",
        "aggregation": "Equal-weight mean of per-subject metrics; no between-seed SD",
        "ece": "15 fixed equal-width bins of maximum softmax probability, [left,right), last inclusive",
        "nll": "Mean hard-label negative log probability, natural logarithm, no smoothing",
        "brier": "Mean sum of squared probability errors over all classes",
        "fusion": "Weighted logits then softmax, using checkpoint global exit weights",
        "interpretation": "F is within the multi-exit model, not independently trained single-exit",
        "summary": summary,
    })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=list(RUNS), default=list(RUNS))
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--precision", choices=["32", "16-mixed"], default="32")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--quiet-summary", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.project))
    torch.set_num_threads(4)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if args.preprocess_only:
        used = install_local_raw_resolver(args.project)
        for dataset in args.datasets:
            run = args.project / "TCFormer result" / RUNS[dataset]
            cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
            for subject in args.subjects:
                print(f"CACHE START {dataset} S{subject}", flush=True)
                log_dir = ROOT / "results" / dataset / f"subject_{subject}"
                log_dir.mkdir(parents=True, exist_ok=True)
                with (log_dir / "cache_build.log").open("w", encoding="utf-8") as stream:
                    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                        x, y, _ = get_test_data(
                            args.project, dataset, subject, cfg["preprocessing"], used)
                print(f"CACHE DONE {dataset} S{subject}: {len(y)} trials", flush=True)
                del x, y
                gc.collect()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this run")
    device = torch.device("cuda:0")
    self_test()
    used = install_local_raw_resolver(args.project)
    write_json(ROOT / "environment.json", {
        "python": sys.executable, "gpu": torch.cuda.get_device_name(device),
        "versions": {name: importlib.metadata.version(name) for name in [
            "torch", "pytorch-lightning", "braindecode", "moabb", "mne",
            "numpy", "scipy", "scikit-learn",
        ]},
        "precision": args.precision, "seed": 0, "metric_self_tests": "passed",
        "script_sha256": digest(__file__),
    })
    failed = []
    for dataset in args.datasets:
        for subject in args.subjects:
            print(f"START {dataset} S{subject}", flush=True)
            try:
                evaluate(args.project, dataset, subject, device, used,
                         args.precision, args.batch_size)
            except Exception as error:
                traceback.print_exc()
                failed.append(dict(dataset=dataset, subject=subject, error=str(error)))
            gc.collect()
            torch.cuda.empty_cache()
            summarize()
            write_json(ROOT / "status.json", {
                "requested_datasets": args.datasets, "requested_subjects": args.subjects,
                "failures": failed, "last_attempt": f"{dataset}/S{subject}",
                "finished": False,
            })
    summary = summarize()
    write_json(ROOT / "status.json", {
        "requested_datasets": args.datasets, "requested_subjects": args.subjects,
        "failures": failed, "finished": True, "summary": summary,
    })
    if not args.quiet_summary:
        print(json.dumps(summary, indent=2), flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
