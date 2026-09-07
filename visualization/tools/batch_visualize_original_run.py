from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def main():
    parser = argparse.ArgumentParser(description="Batch visualize one original single-exit TCFormer run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory that contains checkpoints and config.yaml")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "analysis" / "single_eeg_feature_viz_original_collected",
        help="Centralized output directory",
    )
    parser.add_argument("--subjects", type=str, default="all", help="Comma-separated subject ids, or 'all'")
    parser.add_argument("--stage-layout", choices=["compat", "single_exit"], default="compat")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config.yaml"
    ckpt_dir = run_dir / "checkpoints"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints dir: {ckpt_dir}")

    if args.subjects.lower() == "all":
        subjects = list(range(1, 10))
    else:
        subjects = [int(x.strip()) for x in args.subjects.split(",") if x.strip()]

    run_name = run_dir.name
    collect_dir = args.out_root.resolve() / run_name
    collect_dir.mkdir(parents=True, exist_ok=True)

    for subject_id in subjects:
        ckpt_path = ckpt_dir / f"subject_{subject_id}_model.ckpt"
        if not ckpt_path.exists():
            print(f"[skip] subject {subject_id}: missing {ckpt_path}")
            continue
        cmd = [
            str(PYTHON),
            str(ROOT / "tools" / "visualize_single_eeg_stages_original.py"),
            "--ckpt",
            str(ckpt_path),
            "--config",
            str(config_path),
            "--subject-id",
            str(subject_id),
            "--out-dir",
            str(collect_dir),
            "--stage-layout",
            args.stage_layout,
        ]
        print(f"[run] subject {subject_id}")
        subprocess.run(cmd, check=True)

    summary_src = run_dir / "results.txt"
    if summary_src.exists():
        shutil.copy2(summary_src, collect_dir / "results.txt")
    shutil.copy2(config_path, collect_dir / "config.yaml")
    print(f"[done] collected outputs in: {collect_dir}")


if __name__ == "__main__":
    main()
