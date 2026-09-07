
from pytorch_lightning import Callback
import numpy as np

# Custom Callback to track train/val loss and accuracy each epoch
class MetricsCallback(Callback):
    """Custom PyTorch Lightning callback to track training and validation metrics."""
    def __init__(self):
        super().__init__()
        self.train_loss, self.val_loss = [], []
        self.train_acc, self.val_acc = [], []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.train_loss.append(metrics["train_loss"].cpu().item())
        if "train_acc" in metrics:
            self.train_acc.append(metrics["train_acc"].cpu().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val_loss" in metrics:
            self.val_loss.append(metrics["val_loss"].cpu().item())
        if "val_acc" in metrics:
            self.val_acc.append(metrics["val_acc"].cpu().item())


# Helper to write summary results to a text file
def write_summary(result_dir, model_name, dataset_name, subject_ids,
                   param_count, test_accs, test_losses, test_kappas,
                   train_times, test_times, response_times, test_accs_exits=None,
                   exit_weights=None):
    avg_test_acc = float(np.mean(test_accs))
    std_test_acc = float(np.std(test_accs))
    avg_test_kappa = float(np.mean(test_kappas))   # 🆕  average κ
    std_test_kappa = float(np.std(test_kappas))
    avg_test_loss = float(np.mean(test_losses))
    std_test_loss = float(np.std(test_losses))
    avg_exit_weights = None
    if exit_weights:
        avg_exit_weights = {
            key: float(np.mean([weights[key] for weights in exit_weights]))
            for key in ["shallow", "mid", "deep", "final"]
        }

    total_train_time = float(np.sum(train_times))
    avg_response_time = float(np.mean(response_times))   # milliseconds

    def _finite_stats(values):
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return None, None
        return float(np.mean(finite)), float(np.std(finite))


    with open(result_dir / "results.txt", "w") as f:
        f.write(f"Results for model: {model_name}\n")
        f.write(f"#Params: {param_count}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Subject IDs: {subject_ids}\n\n")
        f.write("Results for each subject:\n")

        for i, subject_id in enumerate(subject_ids):
            f.write(
                f"Subject {subject_id} => Train Time: {train_times[i]:.2f}m, "
                f"Test Time: {test_times[i]:.2f}s, "
                f"Test Acc: {test_accs[i]:.4f}, "
                f"Test Loss: {test_losses[i]:.4f}, "
                f"Test Kappa: {test_kappas[i]:.4f}\n"   # 🆕 κ output
            )
            if exit_weights and i < len(exit_weights):
                weights = exit_weights[i]
                f.write(
                    "  Adaptive Weights => "
                    f"Shallow: {weights['shallow']:.4f}, "
                    f"Mid: {weights['mid']:.4f}, "
                    f"Deep: {weights['deep']:.4f}, "
                    f"Final: {weights['final']:.4f}\n"
                )

        f.write("\n--- Summary Statistics ---\n")
        f.write(f"Average Test Accuracy (Primary Output): {avg_test_acc * 100:.2f} ± {std_test_acc * 100:.2f}\n")
        
        if test_accs_exits and len(test_accs_exits["shallow"]) > 0:
            shallow_stats = _finite_stats(test_accs_exits["shallow"])
            mid_stats = _finite_stats(test_accs_exits["mid"])
            deep_stats = _finite_stats(test_accs_exits["deep"])
            final_stats = _finite_stats(test_accs_exits["final"])

            if shallow_stats[0] is not None:
                f.write(f"  - Shallow Exit Accuracy: {shallow_stats[0] * 100:.2f} ± {shallow_stats[1] * 100:.2f}\n")
            if mid_stats[0] is not None:
                f.write(f"  - Mid Exit Accuracy:     {mid_stats[0] * 100:.2f} ± {mid_stats[1] * 100:.2f}\n")
            if deep_stats[0] is not None:
                f.write(f"  - Deep Exit Accuracy:    {deep_stats[0] * 100:.2f} ± {deep_stats[1] * 100:.2f}\n")
            if final_stats[0] is not None:
                f.write(f"  - Final Exit Accuracy:   {final_stats[0] * 100:.2f} ± {final_stats[1] * 100:.2f}\n")
        if avg_exit_weights is not None:
            f.write("Average Adaptive Exit Weights:\n")
            f.write(f"  - Shallow Exit Weight: {avg_exit_weights['shallow']:.4f}\n")
            f.write(f"  - Mid Exit Weight:     {avg_exit_weights['mid']:.4f}\n")
            f.write(f"  - Deep Exit Weight:    {avg_exit_weights['deep']:.4f}\n")
            f.write(f"  - Final Exit Weight:   {avg_exit_weights['final']:.4f}\n")
            
        f.write(f"Average Test Kappa:    {avg_test_kappa:.3f} ± {std_test_kappa:.3f}\n")
        f.write(f"Average Test Loss:     {avg_test_loss:.3f} ± {std_test_loss:.3f}\n")
        f.write(f"Total Training Time: {total_train_time:.2f} min\n")
        f.write(f"Average Response Time: {avg_response_time:.2f} ms\n")

    print("\n=== Summary ===")
    print(f"Average Test Accuracy (Primary Output): {avg_test_acc * 100:.2f} ± {std_test_acc * 100:.2f}")
    if test_accs_exits and len(test_accs_exits["shallow"]) > 0:
        if shallow_stats[0] is not None:
            print(f"  - Shallow Exit Accuracy: {shallow_stats[0] * 100:.2f} ± {shallow_stats[1] * 100:.2f}")
        if mid_stats[0] is not None:
            print(f"  - Mid Exit Accuracy:     {mid_stats[0] * 100:.2f} ± {mid_stats[1] * 100:.2f}")
        if deep_stats[0] is not None:
            print(f"  - Deep Exit Accuracy:    {deep_stats[0] * 100:.2f} ± {deep_stats[1] * 100:.2f}")
        if final_stats[0] is not None:
            print(f"  - Final Exit Accuracy:   {final_stats[0] * 100:.2f} ± {final_stats[1] * 100:.2f}")
    if avg_exit_weights is not None:
        print("Average Adaptive Exit Weights:")
        print(f"  - Shallow Exit Weight: {avg_exit_weights['shallow']:.4f}")
        print(f"  - Mid Exit Weight:     {avg_exit_weights['mid']:.4f}")
        print(f"  - Deep Exit Weight:    {avg_exit_weights['deep']:.4f}")
        print(f"  - Final Exit Weight:   {avg_exit_weights['final']:.4f}")
    print(f"Average Test Kappa:    {avg_test_kappa:.3f} ± {std_test_kappa:.3f}")
    print(f"Average Test Loss:     {avg_test_loss:.3f} ± {std_test_loss:.3f}")
    print(f"Total Training Time: {total_train_time:.2f} min")
    print(f"Average Response Time: {avg_response_time:.2f} ms")
