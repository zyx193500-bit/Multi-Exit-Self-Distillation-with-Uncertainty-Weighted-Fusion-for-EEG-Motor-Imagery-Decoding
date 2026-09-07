import os, time, yaml
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
# from torchviz import make_dot  # optional for graph visualization

from utils.plotting import plot_confusion_matrix, plot_curve
from utils.metrics  import MetricsCallback, write_summary
from utils.latency  import measure_latency
from utils.misc     import visualize_model_graph, show_gpu_info

from utils.get_datamodule_cls import get_datamodule_cls
from utils.get_model_cls import get_model_cls
from utils.seed import seed_everything

# Set visible GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# Define the path to the configuration directory
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


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
        for key in ["train_loss", "train_acc", "val_loss", "val_acc"]:
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

    def on_validation_epoch_end(self, trainer, pl_module):
        self._write_progress(trainer, status="validating")

    def on_fit_end(self, trainer, pl_module):
        self._write_progress(trainer, status="finished")


def parse_subject_ids(subjects_arg):
    if subjects_arg is None:
        return None
    value = subjects_arg.strip().lower()
    if value == "all":
        return "all"
    return [int(item.strip()) for item in subjects_arg.split(",") if item.strip()]

# Main training and testing pipeline
def train_and_test(config):
     # Create result and checkpoints directories
    model_name = config["model"]
    dataset_name = config["dataset_name"]
    seed = config["seed"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = str(config.get("experiment_name", model_name))
    result_group = str(config.get("result_group", ""))
    result_root = Path(__file__).resolve().parent / "results"
    if result_group:
        result_root = result_root / result_group
    result_dir = (
        result_root / experiment_name /
        f"{model_name}_{dataset_name}_seed-{seed}_aug-{config['preprocessing']['interaug']}"
        f"_GPU{config['gpu_id']}_{timestamp}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["checkpoints", "confmats", "curves"]: (result_dir / sub).mkdir(parents=True, exist_ok=True)

    # Save config to the result directory
    with open(result_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Retrieve model and datamodule classes
    model_cls = get_model_cls(model_name)
    datamodule_cls = get_datamodule_cls(dataset_name)

    config["model_kwargs"]["n_channels"] = datamodule_cls.channels
    config["model_kwargs"]["n_classes"] = datamodule_cls.classes

    # Parse subject IDs from config
    subj_cfg = config["subject_ids"]
    subject_ids = datamodule_cls.all_subject_ids if subj_cfg == "all" else \
                  [subj_cfg] if isinstance(subj_cfg, int) else \
                  subj_cfg
  
    # Initialize containers for tracking metrics across subjects
    test_accs, test_losses, test_kappas = [], [], []
    train_times, test_times, response_times = [], [], []
    all_confmats = []
    test_accs_exits = {"shallow": [], "mid": [], "deep": [], "final": []}
    exit_weights = []

    # Loop through each subject ID for training and testing   
    for subject_id in subject_ids:
        print(f"\n>>> Training on subject: {subject_id}")

        # Set seed for reproducibility
        seed_everything(config["seed"])
        metrics_callback = MetricsCallback()
   
        progress_callback = ProgressFileCallback(
            result_dir=result_dir,
            subject_id=subject_id,
            max_epochs=int(config["max_epochs"]),
        )
        callbacks = [metrics_callback, progress_callback]
        checkpoint_callback = None
        monitor_metric = str(config.get("selection_metric", "val_acc"))

        if config.get("save_best_checkpoint", False):
            checkpoint_callback = ModelCheckpoint(
                dirpath=result_dir / "checkpoints",
                filename=f"subject_{subject_id}_best",
                monitor=monitor_metric,
                mode=str(config.get("selection_mode", "max")),
                save_top_k=1,
                save_last=False,
            )
            callbacks.append(checkpoint_callback)

        early_stopping_cfg = config.get("early_stopping", {})
        if early_stopping_cfg.get("enabled", False):
            callbacks.append(
                EarlyStopping(
                    monitor=early_stopping_cfg.get("monitor", monitor_metric),
                    mode=early_stopping_cfg.get("mode", "max"),
                    patience=int(early_stopping_cfg.get("patience", 40)),
                    min_delta=float(early_stopping_cfg.get("min_delta", 0.0)),
                    verbose=bool(early_stopping_cfg.get("verbose", True)),
                )
            )

        trainer_precision = config.get("precision", "32-true")
        if not torch.cuda.is_available():
            trainer_precision = "32-true"

        # Initialize PyTorch Lightning Trainer
        trainer = Trainer(
            max_epochs=config["max_epochs"],
            devices = "auto" if (config.get("gpu_id", 0) == -1 and torch.cuda.is_available()) else (1 if not torch.cuda.is_available() else [config.get("gpu_id", 0)]),
            num_sanity_val_steps=0,
            accelerator="auto",
            strategy = "auto" if config.get("gpu_id", 0) != -1 
                else DDPStrategy(find_unused_parameters=True), 
            accumulate_grad_batches=int(config.get("accumulate_grad_batches", 1)),
            precision=trainer_precision,
            logger=False,
            enable_checkpointing=checkpoint_callback is not None,
            callbacks=callbacks
        )

        # Instantiate datamodule and model
        datamodule = datamodule_cls(config["preprocessing"], subject_id=subject_id)
        model = model_cls(**config["model_kwargs"], max_epochs=config["max_epochs"])

        # Optionally visualize the model graph
        # visualize_model_graph(model, model_name=model_name)
        
        # Count total number of model parameters
        param_count = sum(p.numel() for p in model.parameters())

        # ---------------- TRAIN ----------------
        st_train = time.time()
        trainer.fit(model, datamodule=datamodule)
        train_times.append((time.time() - st_train) / 60) # minutes

        # ---------------- TEST -----------------
        st_test = time.time()
        test_ckpt_path = "best" if checkpoint_callback is not None else None
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path=test_ckpt_path)
        test_duration = time.time() - st_test
        test_times.append(test_duration)

        # ---------------- LATENCY --------------
        # Deduce input shape from one sample of the test dataset
        sample_x, _ = datamodule.test_dataset[0]
        input_shape = (1, *sample_x.shape)  # prepend batch dim
        # device_str = "cuda:" + str(config["gpu_id"]) if config["gpu_id"] != -1 else "cpu"
        device_str = "cpu"
        lat_ms = measure_latency(model, input_shape, device=device_str)
        response_times.append(lat_ms)  # convert to seconds for summary helper

        # ---------------- METRICS --------------
        # Test metrics follow the configured primary output, which defaults to ensemble.
        acc_key = "test_acc"
        test_accs.append(test_results[0][acc_key])
        test_losses.append(test_results[0]["test_loss"])
        test_kappas.append(test_results[0]["test_kappa"])
        
        if "test_acc_shallow" in test_results[0]:
            active_exits = model.get_active_exits() if hasattr(model, "get_active_exits") else {
                "shallow": True,
                "mid": True,
                "deep": True,
                "final": True,
            }
            for exit_name in ["shallow", "mid", "deep", "final"]:
                metric_name = f"test_acc_{exit_name}"
                value = test_results[0].get(metric_name, float("nan"))
                if not active_exits.get(exit_name, True):
                    value = float("nan")
                test_accs_exits[exit_name].append(value)
        if hasattr(model, "get_exit_weights"):
            exit_weights.append(model.get_exit_weights())

        # compute & store this subject's confusion matrix
        # The [C × C] tensor is inside the LightningModule:
        cm = model.test_confmat.numpy()
        all_confmats.append(cm)

        # plot per-subject if requested
        if config.get("plot_cm_per_subject", False):
            plot_confusion_matrix(
                cm, save_path=result_dir / f"confmats/confmat_subject_{subject_id}.png",
                class_names=datamodule_cls.class_names,
                title=f"Confusion Matrix – Subject {subject_id}",
            )            

        # Plot and save loss and accuracy curves if available
        if metrics_callback.train_loss and metrics_callback.val_loss:
            plot_curve(metrics_callback.train_loss, metrics_callback.val_loss,
                        "Loss", subject_id, result_dir / f"curves/subject_{subject_id}_loss.png")
        if metrics_callback.train_acc and metrics_callback.val_acc:
            plot_curve(metrics_callback.train_acc, metrics_callback.val_acc,
                        "Accuracy", subject_id, result_dir / f"curves/subject_{subject_id}_acc.png")

        # Legacy full-checkpoint export hook
        if config.get("save_checkpoint", False):
            ckpt_path = result_dir / f"checkpoints/subject_{subject_id}_model.ckpt"
            trainer.save_checkpoint(ckpt_path)
   
    # Summarize and save final results
    write_summary(result_dir, model_name, dataset_name, subject_ids, param_count,
        test_accs, test_losses, test_kappas, train_times, test_times, response_times,
        test_accs_exits, exit_weights=exit_weights)
    
    # plot the average if requested
    if config.get("plot_cm_average", True) and all_confmats:
        avg_cm = np.mean(np.stack(all_confmats), axis=0)
        plot_confusion_matrix(
            avg_cm, save_path=result_dir / "confmats/avg_confusion_matrix.png",
            class_names= datamodule_cls.class_names,
            title="Average Confusion Matrix",
        )     


# Command-line argument parsing
def parse_arguments():
    """Parses command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="tcformer",
        help = "Name of the model to use. Options:\n"
               "tcformer, atcnet, d-atcnet, atcnet_2_0, eegnet, shallownet, basenet\n"
                "eegtcnet, eegconformer, tsseffnet, eegdeformer, sst_dpn, ctnet, mscformer"
    )        
    parser.add_argument("--dataset", type=str, default="bcic2a", 
        help="Name of the dataset to use."
                        "Options: bcic2a, bcic2b, hgd, reh_mi, bcic3"
    )
    parser.add_argument("--loso", action="store_true", default=False, 
        help="Enable subject-independent (LOSO) mode"
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID to use")
    
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed value (overrides config if specified)")
    parser.add_argument("--interaug", action="store_true", 
                        help="Enable inter-trial augmentation (overrides config if specified)")
    parser.add_argument("--no_interaug", action="store_true", 
                        help="Disable inter-trial augmentation (overrides config if specified)")
    parser.add_argument("--subjects", type=str, default=None,
                        help='Subjects to run: "all", "1", or comma-separated like "1,2,3"')
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override max epochs after dataset-specific adjustment")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional path to a config YAML file")
    return parser.parse_args()

# ----------------------------------------------
# Main function to run the training and testing pipeline
# ----------------------------------------------
def run():
    # show_gpu_info()     # Uncomment to display GPU info
    args = parse_arguments()
     
    # load config
    config_path = args.config if args.config is not None else os.path.join(CONFIG_DIR, f"{args.model}.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Adjust training parameters based on LOSO setting
    if args.loso:
        config["dataset_name"] = args.dataset + "_loso" 
        config["max_epochs"] = config["max_epochs_loso_hgd"] if args.dataset == "hgd" else config["max_epochs_loso"]
        config["model_kwargs"]["warmup_epochs"] = config["model_kwargs"]["warmup_epochs_loso"]
    else:
        config["dataset_name"] = args.dataset
        config["max_epochs"] = config["max_epochs_2b"] if args.dataset == "bcic2b" else config["max_epochs"]
        if args.dataset == "bcic2b":
            config["accumulate_grad_batches"] = config.get(
                "accumulate_grad_batches_2b",
                config.get("accumulate_grad_batches", 1),
            )
            config["model_kwargs"].update(config.get("model_kwargs_2b", {}))

    config["preprocessing"] = config["preprocessing"][args.dataset]
    config["preprocessing"]["z_scale"] = config["z_scale"]
    # Override interaug if specified
    if args.interaug:
        config["preprocessing"]["interaug"] = True
    elif args.no_interaug:
        config["preprocessing"]["interaug"] = False
    else:
        config["preprocessing"]["interaug"] = config["interaug"]
    config.pop("interaug", None)

    config["gpu_id"] = args.gpu_id
    # Override seed if specified
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    parsed_subjects = parse_subject_ids(args.subjects)
    if parsed_subjects is not None:
        config["subject_ids"] = parsed_subjects

    # set to True to plot confusion matrices
    config["plot_cm_per_subject"] = True # set to True to plot per-subject confusion matrices
    config["plot_cm_average"]     = True # set to True to plot average confusion matrix

    train_and_test(config)

if __name__ == "__main__":
    run()
