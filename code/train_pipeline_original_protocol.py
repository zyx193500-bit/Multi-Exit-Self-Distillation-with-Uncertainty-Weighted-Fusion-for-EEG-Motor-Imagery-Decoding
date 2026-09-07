"""Run the current model under the original TCFormer evaluation protocol.

This entry point intentionally keeps the existing model and preprocessing code,
but changes the evaluation protocol only:

* BCIC IV-2a: session 1 (``session_T``) for training, session 2
  (``session_E``) for testing.
* BCIC IV-2b: sessions 1-3 (``01T``, ``02T``, ``03T``) for training and
  sessions 4-5 (``04E``, ``05E``) for testing.
* No validation loader is exposed during fitting.
* No early stopping and no validation-based checkpoint selection.
* The in-memory model after the fixed epoch schedule is evaluated directly.

The original training pipeline is not modified. This script evaluates the
currently selected model (including the current multi-exit TCFormer) under the
original paper's session/checkpoint protocol; it does not restore the original
single-exit architecture.
"""

from __future__ import annotations

import copy
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from utils.get_datamodule_cls import get_datamodule_cls
from utils.get_model_cls import get_model_cls
from utils.latency import measure_latency
from utils.metrics import MetricsCallback, write_summary
from utils.plotting import plot_confusion_matrix
from utils.seed import seed_everything


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_DIR / "configs"
SUPPORTED_DATASETS = {"bcic2a", "bcic2b", "hgd"}


def _metric_to_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ProgressFileCallback(Callback):
    """Persist progress without assuming that validation is available."""

    def __init__(self, result_dir: Path, subject_id: int, max_epochs: int):
        super().__init__()
        self.subject_id = subject_id
        self.max_epochs = max_epochs
        self.subject_progress_path = result_dir / f"subject_{subject_id}_progress.txt"
        self.live_progress_path = result_dir / "live_progress.txt"

    def _write_progress(self, trainer, status: str):
        metrics = trainer.callback_metrics
        current_epoch = int(trainer.current_epoch) + 1
        if status == "finished":
            current_epoch = min(current_epoch, self.max_epochs)
        lines = [
            f"status={status}",
            f"subject_id={self.subject_id}",
            f"epoch={current_epoch}",
            f"max_epochs={self.max_epochs}",
            f"global_step={trainer.global_step}",
        ]
        for key in ("train_loss", "train_acc"):
            metric = _metric_to_float(metrics.get(key))
            if metric is not None:
                lines.append(f"{key}={metric:.6f}")
        content = "\n".join(lines) + "\n"
        self.subject_progress_path.write_text(content, encoding="utf-8")
        self.live_progress_path.write_text(content, encoding="utf-8")

    def on_fit_start(self, trainer, pl_module):
        self._write_progress(trainer, status="starting")

    def on_train_epoch_end(self, trainer, pl_module):
        self._write_progress(trainer, status="training")

    def on_fit_end(self, trainer, pl_module):
        self._write_progress(trainer, status="finished")


def _strict_protocol_datamodule(base_cls):
    """Return a data-module subclass that never exposes test data as val data."""

    class StrictProtocolDataModule(base_cls):
        def val_dataloader(self):
            return None

    StrictProtocolDataModule.__name__ = f"{base_cls.__name__}StrictProtocol"
    StrictProtocolDataModule.__qualname__ = StrictProtocolDataModule.__name__
    return StrictProtocolDataModule


def parse_subject_ids(subjects_arg):
    if subjects_arg is None:
        return None
    value = subjects_arg.strip().lower()
    if value == "all":
        return "all"
    return [int(item.strip()) for item in subjects_arg.split(",") if item.strip()]


def _protocol_description(dataset_name: str) -> dict:
    descriptions = {
        "bcic2a": {
            "train_sessions": ["S1", "session_T"],
            "test_sessions": ["S2", "session_E"],
        },
        "bcic2b": {
            "train_sessions": ["S1", "S2", "S3", "01T", "02T", "03T"],
            "test_sessions": ["S4", "S5", "04E", "05E"],
        },
        "hgd": {
            "train_sessions": ["S1"],
            "test_sessions": ["S2"],
        },
    }
    return {
        "name": "original_tcformer_session_protocol",
        "dataset": dataset_name,
        "train_sessions": descriptions[dataset_name]["train_sessions"],
        "test_sessions": descriptions[dataset_name]["test_sessions"],
        "validation": "disabled",
        "checkpoint_selection": "none",
        "checkpoint_used_for_test": "last_epoch_in_memory",
        "early_stopping": False,
    }


def _prepare_config(args):
    config_path = Path(args.config) if args.config else CONFIG_DIR / f"{args.model}.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config = copy.deepcopy(yaml.safe_load(handle))

    dataset_name = args.dataset.lower()
    if dataset_name not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(
            f"This strict protocol entry point supports only {supported}; "
            f"got {dataset_name!r}."
        )
    if args.loso:
        raise ValueError(
            "LOSO is a different protocol. Omit --loso for the subject-dependent "
            "session split used by the original main experiments."
        )

    config["dataset_name"] = dataset_name
    if dataset_name == "bcic2b":
        config["max_epochs"] = config["max_epochs_2b"]
        config["accumulate_grad_batches"] = config.get(
            "accumulate_grad_batches_2b",
            config.get("accumulate_grad_batches", 1),
        )
        config["model_kwargs"].update(config.get("model_kwargs_2b", {}))

    config["preprocessing"] = copy.deepcopy(config["preprocessing"][dataset_name])
    config["preprocessing"]["z_scale"] = config["z_scale"]
    if args.interaug:
        config["preprocessing"]["interaug"] = True
    elif args.no_interaug:
        config["preprocessing"]["interaug"] = False
    else:
        config["preprocessing"]["interaug"] = config["interaug"]

    config["gpu_id"] = args.gpu_id
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    parsed_subjects = parse_subject_ids(args.subjects)
    if parsed_subjects is not None:
        config["subject_ids"] = parsed_subjects

    # These settings are recorded explicitly so a result directory is
    # self-describing even if it is copied away from this source tree.
    config["save_best_checkpoint"] = False
    config["selection_metric"] = None
    config["early_stopping"] = {"enabled": False}
    config["evaluation_protocol"] = _protocol_description(dataset_name)
    config["original_config_path"] = str(config_path)
    return config


def train_and_test(config):
    model_name = config["model"]
    dataset_name = config["dataset_name"]
    seed = config["seed"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = (
        PROJECT_DIR
        / "results"
        / "original_protocol"
        / f"{model_name}_{dataset_name}_seed-{seed}_aug-"
        f"{config['preprocessing']['interaug']}_GPU{config['gpu_id']}_{timestamp}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("checkpoints", "confmats", "curves"):
        (result_dir / subdir).mkdir(parents=True, exist_ok=True)

    with (result_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    (result_dir / "protocol.txt").write_text(
        "Original TCFormer session protocol\n"
        f"Dataset: {dataset_name}\n"
        f"Train sessions: {config['evaluation_protocol']['train_sessions']}\n"
        f"Test sessions: {config['evaluation_protocol']['test_sessions']}\n"
        "Validation: disabled\n"
        "Checkpoint selection: none\n"
        "Checkpoint used for test: final last-epoch model\n"
        "Early stopping: disabled\n",
        encoding="utf-8",
    )

    model_cls = get_model_cls(model_name)
    base_datamodule_cls = get_datamodule_cls(dataset_name)
    datamodule_cls = _strict_protocol_datamodule(base_datamodule_cls)
    config["model_kwargs"]["n_channels"] = datamodule_cls.channels
    config["model_kwargs"]["n_classes"] = datamodule_cls.classes

    subject_cfg = config["subject_ids"]
    subject_ids = (
        datamodule_cls.all_subject_ids
        if subject_cfg == "all"
        else [subject_cfg]
        if isinstance(subject_cfg, int)
        else subject_cfg
    )

    test_accs, test_losses, test_kappas = [], [], []
    train_times, test_times, response_times = [], [], []
    all_confmats = []
    test_accs_exits = {"shallow": [], "mid": [], "deep": [], "final": []}
    exit_weights = []

    for subject_id in subject_ids:
        print(f"\n>>> Strict original-protocol training on subject: {subject_id}")
        seed_everything(config["seed"])
        metrics_callback = MetricsCallback()
        progress_callback = ProgressFileCallback(
            result_dir=result_dir,
            subject_id=subject_id,
            max_epochs=int(config["max_epochs"]),
        )

        trainer_precision = config.get("precision", "32-true")
        if not torch.cuda.is_available():
            trainer_precision = "32-true"
        if config.get("gpu_id", 0) == -1 and torch.cuda.is_available():
            devices = "auto"
        elif torch.cuda.is_available():
            devices = [config.get("gpu_id", 0)]
        else:
            devices = 1

        trainer = Trainer(
            max_epochs=config["max_epochs"],
            devices=devices,
            accelerator="auto",
            accumulate_grad_batches=int(config.get("accumulate_grad_batches", 1)),
            precision=trainer_precision,
            logger=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            callbacks=[metrics_callback, progress_callback],
        )

        datamodule = datamodule_cls(config["preprocessing"], subject_id=subject_id)
        model = model_cls(
            **config["model_kwargs"],
            max_epochs=config["max_epochs"],
        )
        param_count = sum(parameter.numel() for parameter in model.parameters())

        train_start = time.time()
        trainer.fit(model, datamodule=datamodule)
        train_times.append((time.time() - train_start) / 60.0)

        test_start = time.time()
        # ckpt_path=None deliberately evaluates the final in-memory model.
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path=None)
        test_times.append(time.time() - test_start)

        sample_x, _ = datamodule.test_dataset[0]
        input_shape = (1, *sample_x.shape)
        response_times.append(measure_latency(model, input_shape, device="cpu"))

        metrics = test_results[0]
        test_accs.append(metrics["test_acc"])
        test_losses.append(metrics["test_loss"])
        test_kappas.append(metrics["test_kappa"])

        if "test_acc_shallow" in metrics:
            active_exits = model.get_active_exits() if hasattr(model, "get_active_exits") else {
                "shallow": True,
                "mid": True,
                "deep": True,
                "final": True,
            }
            for exit_name in ("shallow", "mid", "deep", "final"):
                value = metrics.get(f"test_acc_{exit_name}", float("nan"))
                if not active_exits.get(exit_name, True):
                    value = float("nan")
                test_accs_exits[exit_name].append(value)
        if hasattr(model, "get_exit_weights"):
            exit_weights.append(model.get_exit_weights())

        if model.test_confmat is None:
            raise RuntimeError("The test confusion matrix was not produced.")
        cm = model.test_confmat.numpy()
        all_confmats.append(cm)
        plot_confusion_matrix(
            cm,
            save_path=result_dir / f"confmats/confmat_subject_{subject_id}.png",
            class_names=datamodule_cls.class_names,
            title=f"Subject {subject_id} - final last-epoch model",
        )

        if config.get("save_last_checkpoint", False):
            trainer.save_checkpoint(result_dir / "checkpoints" / f"subject_{subject_id}_last.ckpt")

    write_summary(
        result_dir,
        model_name,
        dataset_name,
        subject_ids,
        param_count,
        test_accs,
        test_losses,
        test_kappas,
        train_times,
        test_times,
        response_times,
        test_accs_exits,
        exit_weights=exit_weights,
    )

    if all_confmats:
        plot_confusion_matrix(
            np.mean(np.stack(all_confmats), axis=0),
            save_path=result_dir / "confmats/avg_confusion_matrix.png",
            class_names=datamodule_cls.class_names,
            title="Average final last-epoch confusion matrix",
        )
    print(f"\nStrict-protocol results written to: {result_dir}")


def parse_arguments():
    parser = ArgumentParser(
        description="Evaluate the current model with the original TCFormer session protocol."
    )
    parser.add_argument("--model", default="tcformer", help="Model config stem, e.g. tcformer")
    parser.add_argument("--dataset", default="bcic2a", choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID; -1 uses all visible GPUs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--interaug", action="store_true")
    parser.add_argument("--no_interaug", action="store_true")
    parser.add_argument("--subjects", default=None, help='"all", "1", or "1,2,3"')
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--config", default=None, help="Optional YAML config path")
    parser.add_argument("--loso", action="store_true", help="Rejected: this script is subject-dependent")
    parser.add_argument(
        "--save_last_checkpoint",
        action="store_true",
        help="Optionally save the final checkpoint; it is never used for model selection.",
    )
    return parser.parse_args()


def run():
    args = parse_arguments()
    config = _prepare_config(args)
    config["save_last_checkpoint"] = bool(args.save_last_checkpoint)
    train_and_test(config)


if __name__ == "__main__":
    run()
