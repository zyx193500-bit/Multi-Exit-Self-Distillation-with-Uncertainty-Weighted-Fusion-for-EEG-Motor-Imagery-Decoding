"""Check exported predictions and checkpoint identities independently of inference."""

import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from scipy.special import softmax
from sklearn.metrics import log_loss


ROOT = Path(__file__).resolve().parent
EXITS = ("shallow", "mid", "deep", "final", "ensemble")
EXPECTED_SUMMARY = {"bcic2a": 87.04, "bcic2b": 89.99}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    checks, pairs, reports = [], [], []
    for dataset, expected_mean in EXPECTED_SUMMARY.items():
        accuracy = []
        for subject in range(1, 10):
            directory = ROOT / "results" / dataset / f"subject_{subject}"
            report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
            ckpt = Path(report["checkpoint_path"])
            assert digest(ckpt) == report["checkpoint_sha256"]
            assert report["strict_load"] and report["seed"] == 0
            assert report["archived_config"]["preprocessing"]["interaug"] is True
            text = (ckpt.parent.parent / "results.txt").read_text(
                encoding="utf-8", errors="replace")
            summary_match = re.search(r"Average Test Accuracy \(Primary Output\): ([\d.]+)", text)
            assert float(summary_match[1]) == expected_mean
            weight_match = re.search(
                rf"Subject {subject} =>[^\n]*\n\s*Adaptive Weights => "
                r"Shallow: ([\d.]+), Mid: ([\d.]+), Deep: ([\d.]+), Final: ([\d.]+)", text)
            np.testing.assert_allclose(
                [report["weights"][k] for k in EXITS[:4]],
                [float(x) for x in weight_match.groups()], atol=0.000051, rtol=0)
            with zipfile.ZipFile(directory / "predictions.npz") as archive:
                assert archive.testzip() is None
            with np.load(directory / "predictions.npz", allow_pickle=False) as values:
                y = values["y_true"]
                assert len(y) == report["data"]["test_n"]
                for key, row in zip(EXITS, report["metrics"]):
                    p = values[f"{key}_probs"]
                    assert np.isfinite(p).all()
                    assert (p >= 0).all() and (p <= 1).all()
                    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-10)
                    np.testing.assert_allclose(
                        p, softmax(values[f"{key}_logits"].astype(np.float64), axis=1))
                    np.testing.assert_allclose(
                        100 * (p.argmax(1) == y).mean(), row["accuracy_pct"])
                    np.testing.assert_allclose(
                        log_loss(y, p, labels=np.arange(p.shape[1])), row["nll"])
                fused = values["ensemble_probs"].argmax(1)
                acc = float(100 * (fused == y).mean())
                assert abs(acc - report["archived_accuracy_pct"]) < 0.0051
                accuracy.append(acc)
                weighted_logits = sum(
                    values[f"{key}_logits"] * report["weights"][key] for key in EXITS[:4])
                np.testing.assert_allclose(
                    weighted_logits, values["ensemble_logits"], rtol=1e-5, atol=1e-6)
                for i, left in enumerate(EXITS[:4]):
                    for right in EXITS[:4][i + 1:]:
                        pred_l = values[f"{left}_probs"].argmax(1)
                        pred_r = values[f"{right}_probs"].argmax(1)
                        err_l, err_r = pred_l != y, pred_r != y
                        corr = (float(np.corrcoef(err_l, err_r)[0, 1])
                                if err_l.std() and err_r.std() else None)
                        pairs.append({
                            "dataset": dataset, "subject": subject, "seed": 0,
                            "exit_1": left, "exit_2": right,
                            "disagreement_pct": float(100 * (pred_l != pred_r).mean()),
                            "both_wrong_pct": float(100 * (err_l & err_r).mean()),
                            "error_correlation": corr,
                        })
            reports.append(report)
            checks.append({
                "dataset": dataset, "subject": subject,
                "accuracy_pct": acc,
                "archived_rounded_accuracy_pct": report["archived_accuracy_pct"],
                "checkpoint_hash_unchanged": True,
                "exit_weights_match_log": True,
                "probability_and_metric_checks": "passed",
            })
        np.testing.assert_allclose(np.mean(accuracy), expected_mean, atol=0.0051, rtol=0)
        print(f"VERIFIED {dataset}: {np.mean(accuracy):.6f}% vs archive {expected_mean:.2f}%")

    write_csv(ROOT / "verification.csv", checks)
    write_csv(ROOT / "exit_disagreement.csv", pairs)
    status = {
        "finished": True, "complete": True, "seed": 0,
        "completed_models": len(checks), "requested_models": 18, "failures": [],
        "total_test_trials": sum(r["data"]["test_n"] for r in reports),
        "checkpoint_hashes": "all unchanged",
        "accuracy_and_exit_weights": "all match archived logs within rounding",
        "precision": "FP32 on CUDA; archived training config was 16-mixed",
        "metric_checks": "passed",
    }
    (ROOT / "verification.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (ROOT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
