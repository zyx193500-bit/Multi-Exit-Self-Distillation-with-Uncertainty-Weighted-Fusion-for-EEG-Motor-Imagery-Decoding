import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy
from torchmetrics.classification import (
    MulticlassCohenKappa, MulticlassConfusionMatrix
)

import pytorch_lightning as pl
from utils.lr_scheduler import linear_warmup_cosine_decay
import random

# Helper: Randomly selects a subset of EEG channels (augmentations)
def select_random_channels(x, keep_ratio=0.9):
    """
    Select a subset of EEG channels.
    Args:
        x: Tensor of shape [B, C, T]
        keep_ratio: fraction of channels to keep
    Returns:
        Tensor of shape [B, C_selected, T]
    """
    B, C, T = x.shape
    keep_chs = int(C * keep_ratio)
    keep_indices = sorted(random.sample(range(C), keep_chs))
    return x[:, keep_indices, :], keep_indices

# Helper: Randomly masks EEG channels (augmentations)
def random_channel_mask(x, keep_ratio=0.9):
    """
    Randomly keeps a subset of EEG channels.
    Args:
        x: Tensor of shape [B, C, T]
        keep_ratio: Float, ratio of channels to keep (e.g., 0.9 to keep 90%).
    Returns:
        Augmented tensor with masked channels set to 0.
    """
    B, C, T = x.shape
    keep_chs = int(C * keep_ratio)
    keep_indices = sorted(random.sample(range(C), keep_chs))
    mask = torch.zeros_like(x)
    mask[:, keep_indices, :] = 1
    return x * mask

# Lightning module
class ClassificationModule(pl.LightningModule):
    def __init__(
            self,
            model,
            n_classes,
            lr=0.001,
            weight_decay=0.0,
            optimizer="adam",
            scheduler=False,
            max_epochs=1000,
            warmup_epochs=20,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        # Kendall et al. (CVPR 2018): learn one homoscedastic uncertainty per BYOT student branch.
        self.exit_log_vars = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))

        # ── metrics ───────────────────────────────────────
        self.test_kappa = MulticlassCohenKappa(num_classes=n_classes)        
        self.test_cm = MulticlassConfusionMatrix(num_classes=n_classes)  
        # will hold the final cm on CPU after test
        self.test_confmat = None

    # forward
    def forward(self, x):
        return self.model(x)

    def _get_uncertainty_statistics(self, device):
        min_log_var = float(self.hparams.get("exit_log_var_min", -3.0))
        max_log_var = float(self.hparams.get("exit_log_var_max", 3.0))
        log_vars = self.exit_log_vars.clamp(min=min_log_var, max=max_log_var).to(device)
        precisions = torch.exp(-log_vars)
        return log_vars, precisions

    def _get_exit_enabled(self, device, dtype=torch.float32):
        enabled = torch.tensor(
            [
                float(self.hparams.get("enable_exit_shallow", True)),
                float(self.hparams.get("enable_exit_mid", True)),
                float(self.hparams.get("enable_exit_deep", True)),
                float(self.hparams.get("enable_exit_final", True)),
            ],
            device=device,
            dtype=dtype,
        )
        if enabled.sum() <= 0:
            enabled[-1] = 1.0
        return enabled

    def _get_schedule_factor(self, mode: str, start_epoch: int, ramp_epochs: int):
        if mode != "train":
            return 1.0
        return self._ramp_value(int(self.current_epoch), start_epoch, ramp_epochs)

    def _get_byot_factors(self, mode: str):
        kd_factor = self._get_schedule_factor(
            mode,
            int(self.hparams.get("byot_distill_start_epoch", 20)),
            int(self.hparams.get("byot_distill_ramp_epochs", 20)),
        )
        hint_factor = self._get_schedule_factor(
            mode,
            int(self.hparams.get("byot_hint_start_epoch", 40)),
            int(self.hparams.get("byot_hint_ramp_epochs", 20)),
        )
        return kd_factor, hint_factor

    def _get_teacher_student_weights(self, device, gates=None):
        _, precisions = self._get_uncertainty_statistics(device)
        enabled = self._get_exit_enabled(device, dtype=precisions.dtype)
        if gates is None:
            gates = torch.ones(3, device=device, dtype=precisions.dtype)
        else:
            gates = gates.to(device=device, dtype=precisions.dtype)

        ensemble_weight_mode = str(self.hparams.get("ensemble_weight_mode", "dynamic")).lower()
        student_mask = gates * enabled[:3]
        teacher_mask = enabled[3:4]

        if ensemble_weight_mode == "uniform":
            raw_weights = torch.cat([student_mask, teacher_mask])
        elif ensemble_weight_mode == "manual":
            manual_weights = torch.tensor(
                [
                    float(self.hparams.get("fixed_weight_shallow", 1.0)),
                    float(self.hparams.get("fixed_weight_mid", 1.0)),
                    float(self.hparams.get("fixed_weight_deep", 1.0)),
                    float(self.hparams.get("fixed_weight_final", 1.0)),
                ],
                device=device,
                dtype=precisions.dtype,
            )
            raw_weights = manual_weights * torch.cat([student_mask, teacher_mask])
        else:
            branch_priors = torch.tensor(
                [
                    float(self.hparams.get("ensemble_prior_shallow", 0.45)),
                    float(self.hparams.get("ensemble_prior_mid", 0.90)),
                    float(self.hparams.get("ensemble_prior_deep", 1.15)),
                ],
                device=device,
                dtype=precisions.dtype,
            )
            teacher_scale = float(self.hparams.get("teacher_ensemble_weight", 3.0))
            raw_weights = torch.cat(
                [
                    student_mask * precisions * branch_priors,
                    teacher_mask * torch.tensor([teacher_scale], device=device, dtype=precisions.dtype),
                ]
            )

        if raw_weights.sum().abs() <= 1e-6:
            raw_weights = torch.cat([torch.zeros(3, device=device, dtype=precisions.dtype), torch.ones(1, device=device, dtype=precisions.dtype)])
        weights = raw_weights / raw_weights.sum().clamp_min(1e-6)
        return precisions, weights

    def _get_ensemble_logits(self, logits, device, gates=None):
        out_shallow = logits["shallow"]
        out_mid = logits["mid"]
        out_deep = logits["deep"]
        out_final = logits["final"]
        branch_precisions, weights = self._get_teacher_student_weights(device, gates)
        w_shallow, w_mid, w_deep, w_final = weights
        ensemble_logits = (
            w_shallow * out_shallow
            + w_mid * out_mid
            + w_deep * out_deep
            + w_final * out_final
        )
        return ensemble_logits, branch_precisions, weights

    def _get_primary_logits(self, mode: str, ensemble_logits, out_final):
        primary_output = str(self.hparams.get("primary_output", "ensemble")).lower()
        if primary_output == "ensemble":
            return ensemble_logits
        if primary_output == "final":
            return out_final
        if mode == "train":
            return ensemble_logits
        return out_final

    def _hint_loss(self, student_feat, teacher_feat):
        if self.hparams.get("byot_normalize_hints", True):
            student_feat = F.normalize(student_feat, dim=-1)
            teacher_feat = F.normalize(teacher_feat, dim=-1)
        return F.mse_loss(student_feat, teacher_feat)

    def _resolve_hint_targets(self, features, enabled):
        feature_names = ["shallow", "mid", "deep", "final"]
        enabled_map = {
            name: bool(enabled[idx].item() > 0)
            for idx, name in enumerate(feature_names)
        }
        targets = {}
        for idx, student_name in enumerate(feature_names[:-1]):
            student_feature = features.get(student_name)
            if student_feature is None or not enabled_map[student_name]:
                targets[student_name] = None
                continue
            target_feature = None
            for teacher_name in feature_names[idx + 1:]:
                teacher_feature = features.get(teacher_name)
                if enabled_map[teacher_name] and teacher_feature is not None:
                    target_feature = teacher_feature.detach()
                    break
            targets[student_name] = target_feature
        return targets

    def _extract_multi_exit_outputs(self, outputs):
        if isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
            features = outputs.get("distill_features", {})
            return logits, features

        if isinstance(outputs, tuple) and len(outputs) == 5:
            _, out_shallow, out_mid, out_deep, out_final = outputs
            logits = {
                "shallow": out_shallow,
                "mid": out_mid,
                "deep": out_deep,
                "final": out_final,
            }
            return logits, {}

        return None, None

    def _ramp_value(self, epoch, start_epoch, ramp_epochs):
        if epoch < start_epoch:
            return 0.0
        if ramp_epochs <= 0:
            return 1.0
        return min((epoch - start_epoch + 1) / ramp_epochs, 1.0)

    def _get_exit_loss_gates(self, mode: str, device):
        if mode != "train" or not self.hparams.get("use_exit_curriculum", True):
            return torch.ones(4, device=device)

        warmup_epochs = int(self.hparams.get("exit_warmup_epochs", 10))
        ramp_epochs = int(self.hparams.get("exit_ramp_epochs", 5))
        epoch = int(self.current_epoch)

        gates = torch.tensor(
            [
                self._ramp_value(epoch, warmup_epochs + 2 * ramp_epochs, ramp_epochs),
                self._ramp_value(epoch, warmup_epochs + ramp_epochs, ramp_epochs),
                self._ramp_value(epoch, warmup_epochs, ramp_epochs),
                1.0,
            ],
            device=device,
            dtype=torch.float32,
        )
        return gates

    # optimiser / scheduler
    def configure_optimizers(self):
        betas = self.hparams.get("beta_1", 0.9), self.hparams.get("beta_2", 0.999)
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                         betas=betas,
                                         weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "adamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                          betas=betas,
                                          weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
                                        weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError
        if self.hparams.scheduler:
            scheduler = LambdaLR(optimizer,
                                 linear_warmup_cosine_decay(self.hparams.warmup_epochs,
                                                            self.hparams.max_epochs))
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    # steps
    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="val")
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="test")
        return {"test_loss": loss, "test_acc": acc}

    # common logic
    def shared_step(self, batch, batch_idx, mode: str = "train"):
        x, y = batch
        if mode == "train":
            if self.hparams.get("random_channel_masking", False):
                x = random_channel_mask(x, self.hparams.get("keep_ratio",0.9))
            if self.hparams.get("random_channel_selection", False):
                x, _ = select_random_channels(x, self.hparams.get("keep_ratio",0.9))

        outputs = self.forward(x)

        logits, features = self._extract_multi_exit_outputs(outputs)

        if logits is not None:
            out_shallow = logits["shallow"]
            out_mid = logits["mid"]
            out_deep = logits["deep"]
            out_final = logits["final"]
            enabled = self._get_exit_enabled(out_final.device, dtype=out_final.dtype)

            label_smoothing = float(self.hparams.get("label_smoothing", 0.05))
            loss_shallow = F.cross_entropy(out_shallow, y, label_smoothing=label_smoothing)
            loss_mid = F.cross_entropy(out_mid, y, label_smoothing=label_smoothing)
            loss_deep = F.cross_entropy(out_deep, y, label_smoothing=label_smoothing)
            loss_final = F.cross_entropy(out_final, y, label_smoothing=label_smoothing)
            acc_shallow = accuracy(out_shallow, y, task="multiclass", num_classes=self.hparams.n_classes)
            acc_mid = accuracy(out_mid, y, task="multiclass", num_classes=self.hparams.n_classes)
            acc_deep = accuracy(out_deep, y, task="multiclass", num_classes=self.hparams.n_classes)
            acc_final = accuracy(out_final, y, task="multiclass", num_classes=self.hparams.n_classes)

            temperature = float(self.hparams.get("distill_temperature", 3.0))
            kd_factor, hint_factor = self._get_byot_factors(mode)
            kd_factor_tensor = torch.tensor(kd_factor, device=out_final.device, dtype=out_final.dtype)
            hint_factor_tensor = torch.tensor(hint_factor, device=out_final.device, dtype=out_final.dtype)
            teacher_logits = out_final.detach()
            zero = torch.zeros((), device=out_final.device, dtype=out_final.dtype)

            kd_enabled = {
                "shallow": bool(self.hparams.get("enable_kd_shallow", True)) and bool(enabled[0].item() > 0),
                "mid": bool(self.hparams.get("enable_kd_mid", True)) and bool(enabled[1].item() > 0),
                "deep": bool(self.hparams.get("enable_kd_deep", True)) and bool(enabled[2].item() > 0),
            }

            kd_shallow = zero if not kd_enabled["shallow"] else F.kl_div(
                F.log_softmax(out_shallow / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)
            kd_mid = zero if not kd_enabled["mid"] else F.kl_div(
                F.log_softmax(out_mid / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)
            kd_deep = zero if not kd_enabled["deep"] else F.kl_div(
                F.log_softmax(out_deep / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)

            feature_shallow = features.get("shallow")
            feature_mid = features.get("mid")
            feature_deep = features.get("deep")
            feature_final = features.get("final")
            hint_targets = self._resolve_hint_targets(features, enabled)
            hint_enabled = {
                "shallow": bool(self.hparams.get("enable_hint_shallow", True)),
                "mid": bool(self.hparams.get("enable_hint_mid", True)),
                "deep": bool(self.hparams.get("enable_hint_deep", True)),
            }
            hint_shallow = zero if (feature_shallow is None or hint_targets["shallow"] is None or not hint_enabled["shallow"]) else self._hint_loss(feature_shallow, hint_targets["shallow"])
            hint_mid = zero if (feature_mid is None or hint_targets["mid"] is None or not hint_enabled["mid"]) else self._hint_loss(feature_mid, hint_targets["mid"])
            hint_deep = zero if (feature_deep is None or hint_targets["deep"] is None or not hint_enabled["deep"]) else self._hint_loss(feature_deep, hint_targets["deep"])

            student_losses = torch.stack(
                [
                    float(self.hparams.get("byot_ce_weight_shallow", 0.8)) * loss_shallow
                    + kd_factor_tensor * float(self.hparams.get("byot_kd_weight_shallow", 0.5)) * kd_shallow
                    + hint_factor_tensor * float(self.hparams.get("byot_hint_weight_shallow", 0.15)) * hint_shallow,
                    float(self.hparams.get("byot_ce_weight_mid", 0.9)) * loss_mid
                    + kd_factor_tensor * float(self.hparams.get("byot_kd_weight_mid", 0.6)) * kd_mid
                    + hint_factor_tensor * float(self.hparams.get("byot_hint_weight_mid", 0.15)) * hint_mid,
                    float(self.hparams.get("byot_ce_weight_deep", 1.0)) * loss_deep
                    + kd_factor_tensor * float(self.hparams.get("byot_kd_weight_deep", 0.7)) * kd_deep
                    + hint_factor_tensor * float(self.hparams.get("byot_hint_weight_deep", 0.10)) * hint_deep,
                ]
            )

            student_gates = self._get_exit_loss_gates(mode, out_final.device)[:3]
            log_vars, branch_precisions = self._get_uncertainty_statistics(student_losses.device)
            student_gates = student_gates * enabled[:3]
            uncertainty_terms = student_gates * (branch_precisions * student_losses + log_vars)
            ensemble_logits, branch_precisions, weights = self._get_ensemble_logits(
                logits, out_final.device, student_gates
            )
            loss_ensemble = F.cross_entropy(ensemble_logits, y, label_smoothing=label_smoothing)
            student_loss_weight_mode = str(self.hparams.get("student_loss_weight_mode", "uncertainty")).lower()
            if student_loss_weight_mode == "uniform":
                active_student_count = student_gates.sum().clamp_min(1.0)
                student_loss_term = (student_gates * student_losses).sum() / active_student_count
            else:
                student_loss_term = uncertainty_terms.sum()
            loss = (
                float(self.hparams.get("ensemble_loss_weight", 1.2)) * loss_ensemble
                + float(self.hparams.get("teacher_loss_weight", 0.8)) * loss_final
                + student_loss_term
            )

            w_shallow, w_mid, w_deep, w_final = weights
            p_shallow, p_mid, p_deep = branch_precisions
            lv_shallow, lv_mid, lv_deep = log_vars
            acc_ensemble = accuracy(ensemble_logits, y, task="multiclass", num_classes=self.hparams.n_classes)
            primary_logits = self._get_primary_logits(mode, ensemble_logits, out_final)
            primary_loss = loss_ensemble if primary_logits is ensemble_logits else loss_final
            primary_acc = accuracy(primary_logits, y, task="multiclass", num_classes=self.hparams.n_classes)

            self.log(f"{mode}_loss", primary_loss, prog_bar=True, on_step=False, on_epoch=True)
            if mode == "train":
                self.log(f"{mode}_loss_total", loss, prog_bar=False, on_step=False, on_epoch=True)
            elif mode in {"val", "test"}:
                self.log(f"{mode}_loss_total", loss, prog_bar=False, on_step=False, on_epoch=True)
            self.log(
                f"{mode}_loss_ce",
                torch.stack([loss_shallow, loss_mid, loss_deep, loss_final]).mean(),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(f"{mode}_loss_shallow", loss_shallow, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_loss_mid", loss_mid, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_loss_deep", loss_deep, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_loss_final", loss_final, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_loss_kd_shallow", kd_shallow, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_loss_kd_mid", kd_mid, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_loss_kd_deep", kd_deep, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_loss_hint_shallow", hint_shallow, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_loss_hint_mid", hint_mid, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_loss_hint_deep", hint_deep, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_loss_student_shallow", student_losses[0], prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_loss_student_mid", student_losses[1], prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_loss_student_deep", student_losses[2], prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_loss_ensemble", loss_ensemble, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_factor_kd", kd_factor_tensor, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_factor_hint", hint_factor_tensor, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_weight_shallow", w_shallow, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_weight_mid", w_mid, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_weight_deep", w_deep, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_weight_final", w_final, prog_bar=False, on_step=False, on_epoch=True)
            if mode == "train":
                self.log(f"{mode}_precision_shallow", p_shallow, prog_bar=False, on_step=False, on_epoch=True)
                self.log(f"{mode}_precision_mid", p_mid, prog_bar=False, on_step=False, on_epoch=True)
                self.log(f"{mode}_precision_deep", p_deep, prog_bar=False, on_step=False, on_epoch=True)
                self.log(f"{mode}_log_var_shallow", lv_shallow, prog_bar=False, on_step=False, on_epoch=True)
                self.log(f"{mode}_log_var_mid", lv_mid, prog_bar=False, on_step=False, on_epoch=True)
                self.log(f"{mode}_log_var_deep", lv_deep, prog_bar=False, on_step=False, on_epoch=True)
                self.log(f"{mode}_gate_shallow", student_gates[0], prog_bar=False, on_step=False, on_epoch=True)
                self.log(f"{mode}_gate_mid", student_gates[1], prog_bar=False, on_step=False, on_epoch=True)
                self.log(f"{mode}_gate_deep", student_gates[2], prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_acc", primary_acc, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{mode}_acc_shallow", acc_shallow, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_acc_mid", acc_mid, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_acc_deep", acc_deep, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_acc_final", acc_final, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{mode}_acc_ensemble", acc_ensemble, prog_bar=False, on_step=False, on_epoch=True)
            
            if mode == "test":
                preds = torch.argmax(primary_logits, dim=-1)
                self.test_kappa.update(preds, y)
                self.test_cm.update(preds, y)
                self.log("test_kappa", self.test_kappa, prog_bar=False, on_step=False, on_epoch=True)
                
            return loss if mode == "train" else primary_loss, primary_acc
        else:
            y_hat = outputs
            loss = F.cross_entropy(y_hat, y)
            acc = accuracy(y_hat, y, task="multiclass", num_classes=self.hparams.n_classes)
            self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{mode}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

            if mode == "test":
                preds = torch.argmax(y_hat, dim=-1)
                self.test_kappa.update(preds, y)                       
                self.test_cm.update(preds, y)
                self.log("test_kappa", self.test_kappa, prog_bar=False, on_step=False, on_epoch=True)

            return loss, acc
    
    # grab confusion matrix once per test epoch
    def on_test_epoch_end(self):
        # 1) raw counts  ───────────────────────────────────────────
        cm_counts = self.test_cm.compute()   # shape [C, C]
        self.test_cm.reset()

        # 2) row-normalise → %  (handle rows with 0 samples safely)
        with torch.no_grad():
            row_sums = cm_counts.sum(dim=1, keepdim=True).clamp_min(1)
            cm_percent = cm_counts.float() / row_sums * 100.0

        self.test_confmat = cm_percent.cpu()        # stash for plotting

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        outputs = self.forward(x)
        logits, _ = self._extract_multi_exit_outputs(outputs)
        if logits is not None:
            out_shallow = logits["shallow"]
            out_mid = logits["mid"]
            out_deep = logits["deep"]
            out_final = logits["final"]
            ensemble_logits, _, weights = self._get_ensemble_logits(logits, out_shallow.device)
            final_probs = F.softmax(out_final, dim=-1)
            ensemble_probs = F.softmax(ensemble_logits, dim=-1)
            primary_logits = self._get_primary_logits("predict", ensemble_logits, out_final)
            return {
                "preds": torch.argmax(primary_logits, dim=-1),
                "final_preds": torch.argmax(out_final, dim=-1),
                "ensemble_preds": torch.argmax(ensemble_logits, dim=-1),
                "ensemble_probs": ensemble_probs,
                "shallow_probs": F.softmax(out_shallow, dim=-1),
                "mid_probs": F.softmax(out_mid, dim=-1),
                "deep_probs": F.softmax(out_deep, dim=-1),
                "final_probs": final_probs,
                "weights": weights.detach().cpu()
            }
        else:
            return torch.argmax(outputs, dim=-1)

    def get_exit_weights(self):
        _, weights = self._get_teacher_student_weights(self.exit_log_vars.device)
        weights = weights.detach().cpu()
        return {
            "shallow": float(weights[0].item()),
            "mid": float(weights[1].item()),
            "deep": float(weights[2].item()),
            "final": float(weights[3].item()),
        }

    def get_active_exits(self):
        return {
            "shallow": bool(self.hparams.get("enable_exit_shallow", True)),
            "mid": bool(self.hparams.get("enable_exit_mid", True)),
            "deep": bool(self.hparams.get("enable_exit_deep", True)),
            "final": bool(self.hparams.get("enable_exit_final", True)),
        }
    
