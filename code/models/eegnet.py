import torch
from einops.layers.torch import Rearrange

from torch import Tensor, nn

from .classification_module import ClassificationModule
from .modules import Conv2dWithConstraint
from utils.weight_initialization import glorot_weight_zero_bias


class EEGNetModule(nn.Module):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            input_window_samples: int,
            pool_mode: str = "mean",
            F1: int = 8,
            D: int = 2,
            F2: int = 16,
            kernel_length: int = 32,
            drop_prob: float = 0.5,
            pool_time_length: int = 4,
            pool_time_stride: int = 4,
            kernel_length_dw_sep: int = 16
    ):
        super(EEGNetModule, self).__init__()
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]


        self.rearrange_input = Rearrange("b c t -> b 1 c t")
        self.conv_temporal = nn.Conv2d(1, F1, (1, kernel_length), bias=False,
                                       padding=(0, kernel_length // 2))
        self.bnorm_temporal = nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3)
        self.conv_spatial = Conv2dWithConstraint(F1, F1 * D, (n_channels, 1),
                                                 bias=False, groups=F1, max_norm=1)
        self.bnorm_1 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3)
        self.elu_1 = nn.ELU()
        self.pool_1 = pool_class((1, pool_time_length), stride=(1, pool_time_stride))
        self.drop1 = nn.Dropout(drop_prob)

        self.dw_sep_conv = nn.Sequential(
            nn.Conv2d(F2, F2, (1, kernel_length_dw_sep), bias=False, groups=F2,
                      padding=(0, kernel_length_dw_sep // 2)),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3),
            nn.ELU(),
            pool_class((1, 8)),
            nn.Dropout(drop_prob),
        )

        out = input_window_samples + 2 * (kernel_length // 2) - kernel_length + 1
        out = int((out - pool_time_length) / pool_time_stride + 1)
        out = out + 2 * (kernel_length_dw_sep // 2) - kernel_length_dw_sep + 1
        out = int((out - 8) / 8 + 1)

        self.classifier = nn.Sequential(
            nn.Conv2d(F2, n_classes, (1, out)),
            Rearrange("b n_classes 1 1 -> b n_classes"))

        glorot_weight_zero_bias(self)

    def forward(self, x):
        x = self.rearrange_input(x)
        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)
        x = self.conv_spatial(x)
        x = self.bnorm_1(x)
        x = self.elu_1(x)
        x = self.pool_1(x)
        x = self.drop1(x)
        x = self.dw_sep_conv(x)
        x = self.classifier(x)
        return x


def _select_num_heads(channels: int, requested_heads: int) -> int:
    candidates = [requested_heads, 4, 2, 1]
    for head_count in candidates:
        if head_count > 0 and channels % head_count == 0:
            return head_count
    return 1


def _to_sequence_features(x: Tensor) -> Tensor:
    if x.dim() == 4:
        if x.size(2) == 1:
            return x.squeeze(2)
        return x.flatten(1, 2)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected a 3D or 4D tensor, got shape={tuple(x.shape)}")


def _compute_eegnet_final_bins(
        input_window_samples: int,
        kernel_length: int,
        pool_time_length: int,
        pool_time_stride: int,
        kernel_length_dw_sep: int,
        pool_time_length_2: int,
        pool_time_stride_2: int,
) -> int:
    out = input_window_samples + 2 * (kernel_length // 2) - kernel_length + 1
    out = int((out - pool_time_length) / pool_time_stride + 1)
    out = out + 2 * (kernel_length_dw_sep // 2) - kernel_length_dw_sep + 1
    out = int((out - pool_time_length_2) / pool_time_stride_2 + 1)
    return max(out, 1)


class AttentionPooling1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.score = nn.Linear(channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        tokens = x.transpose(1, 2)
        tokens = self.norm(tokens)
        weights = torch.softmax(self.score(tokens), dim=1)
        return torch.sum(tokens * weights, dim=1)


class ConvBottleneck1D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.5):
        super().__init__()
        mid_channels = max(channels // 2, 8)
        self.block = nn.Sequential(
            nn.Conv1d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.Conv1d(mid_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MHSABottleneck1D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.5):
        super().__init__()
        mid_channels = max(channels // 2, 8)
        num_heads = _select_num_heads(mid_channels, num_heads)
        self.pre = nn.Sequential(
            nn.Conv1d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels, momentum=0.01, eps=1e-3),
            nn.ELU(),
        )
        self.norm = nn.LayerNorm(mid_channels)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=mid_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.post = nn.Sequential(
            nn.Conv1d(mid_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels, momentum=0.01, eps=1e-3),
            nn.ELU(),
        )
        self.shortcut = nn.Identity() if channels == channels else nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.pre(x)
        tokens = x.transpose(1, 2)
        tokens = self.norm(tokens)
        attn_out, _ = self.mhsa(tokens, tokens, tokens)
        x = x + self.dropout(attn_out.transpose(1, 2))
        x = self.post(x)
        return x + residual


class ExitRouter1D(nn.Module):
    def __init__(self, channels: int, shared_ratio: float = 0.5, private_ratio: float = 0.25, dropout: float = 0.5):
        super().__init__()
        shared_channels = max(int(channels * shared_ratio), 1)
        private_channels = max(int(channels * private_ratio), 1)
        selected_channels = min(shared_channels + private_channels, channels)
        self.selected_channels = selected_channels
        self.proj = nn.Sequential(
            nn.Conv1d(selected_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = _to_sequence_features(x)
        routed = x[:, : self.selected_channels, :]
        return self.proj(routed)


class MultiExitClassifier(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, bottleneck_type: str = "conv", num_heads: int = 4, dropout: float = 0.5):
        super().__init__()
        if bottleneck_type == "conv":
            self.bottleneck = ConvBottleneck1D(in_channels, dropout=dropout)
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif bottleneck_type == "mhsa":
            self.bottleneck = MHSABottleneck1D(in_channels, num_heads=num_heads, dropout=dropout)
            self.pool = AttentionPooling1D(in_channels)
        else:
            raise ValueError(f"Unsupported bottleneck_type: {bottleneck_type}")
        self.bottleneck_type = bottleneck_type
        self.fc = nn.Linear(in_channels, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bottleneck(x)
        if self.bottleneck_type == "mhsa":
            x = self.pool(x)
        else:
            x = self.pool(x).squeeze(-1)
        return self.fc(x)


class DistillationProjectionHead(nn.Module):
    def __init__(self, in_channels: int, proj_dim: int, dropout: float = 0.5):
        super().__init__()
        hidden_dim = max(proj_dim, in_channels // 2, 32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


class LiteMultiExitClassifier(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        mid_channels = max(in_channels, 8)
        hidden_channels = max(in_channels // 2, 8)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.Conv1d(mid_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_channels, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class EEGNetMultiExitModule(nn.Module):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            input_window_samples: int,
            pool_mode: str = "mean",
            F1: int = 8,
            D: int = 2,
            F2: int = 16,
            kernel_length: int = 32,
            drop_prob: float = 0.5,
            pool_time_length: int = 4,
            pool_time_stride: int = 4,
            kernel_length_dw_sep: int = 16,
            pool_time_length_2: int = 8,
            pool_time_stride_2: int = 8,
            exit_head_num_heads: int = 4,
            byot_proj_dim: int = 128,
    ):
        super().__init__()
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        self.rearrange_input = Rearrange("b c t -> b 1 c t")
        self.conv_temporal = nn.Conv2d(
            1,
            F1,
            (1, kernel_length),
            bias=False,
            padding=(0, kernel_length // 2),
        )
        self.bnorm_temporal = nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3)
        self.conv_spatial = Conv2dWithConstraint(
            F1,
            F1 * D,
            (n_channels, 1),
            bias=False,
            groups=F1,
            max_norm=1,
        )
        self.bnorm_1 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3)
        self.elu_1 = nn.ELU()
        self.pool_1 = pool_class((1, pool_time_length), stride=(1, pool_time_stride))
        self.drop_1 = nn.Dropout(drop_prob)

        self.dw_temporal = nn.Conv2d(
            F1 * D,
            F1 * D,
            (1, kernel_length_dw_sep),
            bias=False,
            groups=F1 * D,
            padding=(0, kernel_length_dw_sep // 2),
        )
        self.pw_projection = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bnorm_2 = nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3)
        self.elu_2 = nn.ELU()
        self.pool_2 = pool_class((1, pool_time_length_2), stride=(1, pool_time_stride_2))
        self.drop_2 = nn.Dropout(drop_prob)

        shallow_channels = F1 * D
        mid_channels = F2
        deep_channels = F2
        final_channels = F2

        self.router_shallow = ExitRouter1D(shallow_channels, shared_ratio=0.5, private_ratio=0.25, dropout=drop_prob)
        self.router_mid = ExitRouter1D(mid_channels, shared_ratio=0.5, private_ratio=0.25, dropout=drop_prob)
        self.router_deep = ExitRouter1D(deep_channels, shared_ratio=0.5, private_ratio=0.25, dropout=drop_prob)
        self.router_final = ExitRouter1D(final_channels, shared_ratio=0.6, private_ratio=0.3, dropout=drop_prob)

        self.classifier_shallow = MultiExitClassifier(
            shallow_channels,
            n_classes,
            bottleneck_type="conv",
            num_heads=exit_head_num_heads,
            dropout=drop_prob,
        )
        self.classifier_mid = MultiExitClassifier(
            mid_channels,
            n_classes,
            bottleneck_type="mhsa",
            num_heads=exit_head_num_heads,
            dropout=drop_prob,
        )
        self.classifier_deep = MultiExitClassifier(
            deep_channels,
            n_classes,
            bottleneck_type="mhsa",
            num_heads=exit_head_num_heads,
            dropout=drop_prob,
        )
        self.classifier_final = MultiExitClassifier(
            final_channels,
            n_classes,
            bottleneck_type="mhsa",
            num_heads=exit_head_num_heads,
            dropout=drop_prob,
        )

        self.proj_shallow = DistillationProjectionHead(shallow_channels, byot_proj_dim, drop_prob)
        self.proj_mid = DistillationProjectionHead(mid_channels, byot_proj_dim, drop_prob)
        self.proj_deep = DistillationProjectionHead(deep_channels, byot_proj_dim, drop_prob)
        self.proj_final = DistillationProjectionHead(final_channels, byot_proj_dim, drop_prob)

        self.input_window_samples = input_window_samples
        glorot_weight_zero_bias(self)

    def forward(self, x):
        x = self.rearrange_input(x)
        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)

        x = self.conv_spatial(x)
        x = self.bnorm_1(x)
        x = self.elu_1(x)
        block1 = self.pool_1(x)
        block1 = self.drop_1(block1)

        shallow_routed = self.router_shallow(block1)
        out_shallow = self.classifier_shallow(shallow_routed)
        feat_shallow = self.proj_shallow(shallow_routed)

        mid_map = self.dw_temporal(block1)
        mid_map = self.pw_projection(mid_map)
        mid_map = self.bnorm_2(mid_map)
        mid_map = self.elu_2(mid_map)

        mid_routed = self.router_mid(mid_map)
        out_mid = self.classifier_mid(mid_routed)
        feat_mid = self.proj_mid(mid_routed)

        deep_map = self.pool_2(mid_map)
        deep_map = self.drop_2(deep_map)

        deep_routed = self.router_deep(deep_map)
        out_deep = self.classifier_deep(deep_routed)
        feat_deep = self.proj_deep(deep_routed)

        final_routed = self.router_final(deep_map)
        out_final = self.classifier_final(final_routed)
        feat_final = self.proj_final(final_routed)

        return {
            "ensemble_logits": out_final,
            "logits": {
                "shallow": out_shallow,
                "mid": out_mid,
                "deep": out_deep,
                "final": out_final,
            },
            "distill_features": {
                "shallow": feat_shallow,
                "mid": feat_mid,
                "deep": feat_deep,
                "final": feat_final,
            },
        }


class EEGNetMultiExitLiteModule(nn.Module):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            input_window_samples: int,
            pool_mode: str = "mean",
            F1: int = 8,
            D: int = 2,
            F2: int = 16,
            kernel_length: int = 32,
            drop_prob: float = 0.5,
            pool_time_length: int = 4,
            pool_time_stride: int = 4,
            kernel_length_dw_sep: int = 16,
            pool_time_length_2: int = 8,
            pool_time_stride_2: int = 8,
            byot_proj_dim: int = 64,
    ):
        super().__init__()
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        self.rearrange_input = Rearrange("b c t -> b 1 c t")
        self.conv_temporal = nn.Conv2d(
            1,
            F1,
            (1, kernel_length),
            bias=False,
            padding=(0, kernel_length // 2),
        )
        self.bnorm_temporal = nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3)
        self.conv_spatial = Conv2dWithConstraint(
            F1,
            F1 * D,
            (n_channels, 1),
            bias=False,
            groups=F1,
            max_norm=1,
        )
        self.bnorm_1 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3)
        self.elu_1 = nn.ELU()
        self.pool_1 = pool_class((1, pool_time_length), stride=(1, pool_time_stride))
        self.drop_1 = nn.Dropout(drop_prob)

        self.dw_temporal = nn.Conv2d(
            F1 * D,
            F1 * D,
            (1, kernel_length_dw_sep),
            bias=False,
            groups=F1 * D,
            padding=(0, kernel_length_dw_sep // 2),
        )
        self.pw_projection = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bnorm_2 = nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3)
        self.elu_2 = nn.ELU()
        self.pool_2 = pool_class((1, pool_time_length_2), stride=(1, pool_time_stride_2))
        self.drop_2 = nn.Dropout(drop_prob)

        shallow_channels = F1 * D
        mid_channels = F2
        deep_channels = F2
        final_channels = F2
        proj_dim = min(byot_proj_dim, 64)

        self.router_shallow = ExitRouter1D(shallow_channels, shared_ratio=0.5, private_ratio=0.15, dropout=drop_prob)
        self.router_mid = ExitRouter1D(mid_channels, shared_ratio=0.5, private_ratio=0.15, dropout=drop_prob)
        self.router_deep = ExitRouter1D(deep_channels, shared_ratio=0.6, private_ratio=0.15, dropout=drop_prob)

        self.classifier_shallow = LiteMultiExitClassifier(shallow_channels, n_classes, dropout=drop_prob)
        self.classifier_mid = LiteMultiExitClassifier(mid_channels, n_classes, dropout=drop_prob)
        self.classifier_deep = LiteMultiExitClassifier(deep_channels, n_classes, dropout=drop_prob)

        # Keep a light convolutional final head so the teacher stays close to EEGNet.
        self.classifier_final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(final_channels, n_classes, (1, 1)),
            Rearrange("b n_classes 1 1 -> b n_classes"),
        )

        self.proj_shallow = DistillationProjectionHead(shallow_channels, proj_dim, drop_prob)
        self.proj_mid = DistillationProjectionHead(mid_channels, proj_dim, drop_prob)
        self.proj_deep = DistillationProjectionHead(deep_channels, proj_dim, drop_prob)
        self.proj_final = DistillationProjectionHead(final_channels, proj_dim, drop_prob)

        glorot_weight_zero_bias(self)

    def forward(self, x):
        x = self.rearrange_input(x)
        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)

        x = self.conv_spatial(x)
        x = self.bnorm_1(x)
        x = self.elu_1(x)
        block1 = self.pool_1(x)
        block1 = self.drop_1(block1)

        shallow_routed = self.router_shallow(block1)
        out_shallow = self.classifier_shallow(shallow_routed)
        feat_shallow = self.proj_shallow(shallow_routed)

        mid_map = self.dw_temporal(block1)
        mid_map = self.pw_projection(mid_map)
        mid_map = self.bnorm_2(mid_map)
        mid_map = self.elu_2(mid_map)

        mid_routed = self.router_mid(mid_map)
        out_mid = self.classifier_mid(mid_routed)
        feat_mid = self.proj_mid(mid_routed)

        deep_map = self.pool_2(mid_map)
        deep_map = self.drop_2(deep_map)

        deep_routed = self.router_deep(deep_map)
        out_deep = self.classifier_deep(deep_routed)
        feat_deep = self.proj_deep(deep_routed)

        out_final = self.classifier_final(deep_map)
        feat_final = self.proj_final(deep_map.squeeze(2))

        return {
            "ensemble_logits": out_final,
            "logits": {
                "shallow": out_shallow,
                "mid": out_mid,
                "deep": out_deep,
                "final": out_final,
            },
            "distill_features": {
                "shallow": feat_shallow,
                "mid": feat_mid,
                "deep": feat_deep,
                "final": feat_final,
            },
        }


class EEGNet(ClassificationModule):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            input_window_samples: int,
            pool_mode: str = "mean",
            F1: int = 8,
            D: int = 2,
            F2: int = 16,
            kernel_length: int = 32,
            drop_prob: float = 0.5,
            pool_time_length: int = 4,
            pool_time_stride: int = 4,
            kernel_length_dw_sep: int = 16,
            **kwargs
    ):
        model = EEGNetModule(
            n_channels=n_channels,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            pool_mode=pool_mode,
            F1=F1,
            D=D,
            F2=F2,
            kernel_length=kernel_length,
            drop_prob=drop_prob,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            kernel_length_dw_sep=kernel_length_dw_sep,
        )
        super(EEGNet, self).__init__(model, n_classes, **kwargs)


class EEGNetMultiExit(ClassificationModule):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            input_window_samples: int,
            pool_mode: str = "mean",
            F1: int = 8,
            D: int = 2,
            F2: int = 16,
            kernel_length: int = 32,
            drop_prob: float = 0.5,
            pool_time_length: int = 4,
            pool_time_stride: int = 4,
            kernel_length_dw_sep: int = 16,
            pool_time_length_2: int = 8,
            pool_time_stride_2: int = 8,
            exit_head_num_heads: int = 4,
            **kwargs
    ):
        model = EEGNetMultiExitModule(
            n_channels=n_channels,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            pool_mode=pool_mode,
            F1=F1,
            D=D,
            F2=F2,
            kernel_length=kernel_length,
            drop_prob=drop_prob,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            kernel_length_dw_sep=kernel_length_dw_sep,
            pool_time_length_2=pool_time_length_2,
            pool_time_stride_2=pool_time_stride_2,
            exit_head_num_heads=exit_head_num_heads,
            byot_proj_dim=kwargs.get("byot_proj_dim", 128),
        )
        super(EEGNetMultiExit, self).__init__(model, n_classes, **kwargs)


class EEGNetMultiExitLite(ClassificationModule):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            input_window_samples: int,
            pool_mode: str = "mean",
            F1: int = 8,
            D: int = 2,
            F2: int = 16,
            kernel_length: int = 32,
            drop_prob: float = 0.5,
            pool_time_length: int = 4,
            pool_time_stride: int = 4,
            kernel_length_dw_sep: int = 16,
            pool_time_length_2: int = 8,
            pool_time_stride_2: int = 8,
            **kwargs
    ):
        model = EEGNetMultiExitLiteModule(
            n_channels=n_channels,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            pool_mode=pool_mode,
            F1=F1,
            D=D,
            F2=F2,
            kernel_length=kernel_length,
            drop_prob=drop_prob,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            kernel_length_dw_sep=kernel_length_dw_sep,
            pool_time_length_2=pool_time_length_2,
            pool_time_stride_2=pool_time_stride_2,
            byot_proj_dim=kwargs.get("byot_proj_dim", 64),
        )
        super(EEGNetMultiExitLite, self).__init__(model, n_classes, **kwargs)


class EEGNetMultiExitLiteV2Module(nn.Module):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            input_window_samples: int,
            pool_mode: str = "mean",
            F1: int = 8,
            D: int = 2,
            F2: int = 16,
            kernel_length: int = 32,
            drop_prob: float = 0.5,
            pool_time_length: int = 4,
            pool_time_stride: int = 4,
            kernel_length_dw_sep: int = 16,
            pool_time_length_2: int = 8,
            pool_time_stride_2: int = 8,
            byot_proj_dim: int = 64,
    ):
        super().__init__()
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        self.rearrange_input = Rearrange("b c t -> b 1 c t")
        self.conv_temporal = nn.Conv2d(
            1,
            F1,
            (1, kernel_length),
            bias=False,
            padding=(0, kernel_length // 2),
        )
        self.bnorm_temporal = nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3)
        self.conv_spatial = Conv2dWithConstraint(
            F1,
            F1 * D,
            (n_channels, 1),
            bias=False,
            groups=F1,
            max_norm=1,
        )
        self.bnorm_1 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3)
        self.elu_1 = nn.ELU()
        self.pool_1 = pool_class((1, pool_time_length), stride=(1, pool_time_stride))
        self.drop_1 = nn.Dropout(drop_prob)

        self.dw_temporal = nn.Conv2d(
            F1 * D,
            F1 * D,
            (1, kernel_length_dw_sep),
            bias=False,
            groups=F1 * D,
            padding=(0, kernel_length_dw_sep // 2),
        )
        self.pw_projection = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bnorm_2 = nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3)
        self.elu_2 = nn.ELU()
        self.pool_2 = pool_class((1, pool_time_length_2), stride=(1, pool_time_stride_2))
        self.drop_2 = nn.Dropout(drop_prob)

        shallow_channels = F1 * D
        mid_channels = F2
        deep_channels = F2
        final_channels = F2
        proj_dim = min(byot_proj_dim, 64)
        final_bins = _compute_eegnet_final_bins(
            input_window_samples,
            kernel_length,
            pool_time_length,
            pool_time_stride,
            kernel_length_dw_sep,
            pool_time_length_2,
            pool_time_stride_2,
        )

        self.router_shallow = ExitRouter1D(shallow_channels, shared_ratio=0.5, private_ratio=0.15, dropout=drop_prob)
        self.router_mid = ExitRouter1D(mid_channels, shared_ratio=0.5, private_ratio=0.15, dropout=drop_prob)
        self.router_deep = ExitRouter1D(deep_channels, shared_ratio=0.6, private_ratio=0.15, dropout=drop_prob)

        self.classifier_shallow = LiteMultiExitClassifier(shallow_channels, n_classes, dropout=drop_prob)
        self.classifier_mid = LiteMultiExitClassifier(mid_channels, n_classes, dropout=drop_prob)
        self.classifier_deep = LiteMultiExitClassifier(deep_channels, n_classes, dropout=drop_prob)

        # Preserve the original EEGNet idea: keep multiple temporal bins before the final classification conv.
        self.classifier_final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, final_bins)),
            nn.Conv2d(final_channels, n_classes, (1, final_bins)),
            Rearrange("b n_classes 1 1 -> b n_classes"),
        )

        self.proj_shallow = DistillationProjectionHead(shallow_channels, proj_dim, drop_prob)
        self.proj_mid = DistillationProjectionHead(mid_channels, proj_dim, drop_prob)
        self.proj_deep = DistillationProjectionHead(deep_channels, proj_dim, drop_prob)
        self.proj_final = DistillationProjectionHead(final_channels, proj_dim, drop_prob)

        glorot_weight_zero_bias(self)

    def forward(self, x):
        x = self.rearrange_input(x)
        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)

        x = self.conv_spatial(x)
        x = self.bnorm_1(x)
        x = self.elu_1(x)
        block1 = self.pool_1(x)
        block1 = self.drop_1(block1)

        shallow_routed = self.router_shallow(block1)
        out_shallow = self.classifier_shallow(shallow_routed)
        feat_shallow = self.proj_shallow(shallow_routed)

        mid_map = self.dw_temporal(block1)
        mid_map = self.pw_projection(mid_map)
        mid_map = self.bnorm_2(mid_map)
        mid_map = self.elu_2(mid_map)

        mid_routed = self.router_mid(mid_map)
        out_mid = self.classifier_mid(mid_routed)
        feat_mid = self.proj_mid(mid_routed)

        deep_map = self.pool_2(mid_map)
        deep_map = self.drop_2(deep_map)

        deep_routed = self.router_deep(deep_map)
        out_deep = self.classifier_deep(deep_routed)
        feat_deep = self.proj_deep(deep_routed)

        out_final = self.classifier_final(deep_map)
        feat_final = self.proj_final(deep_map.squeeze(2))

        return {
            "ensemble_logits": out_final,
            "logits": {
                "shallow": out_shallow,
                "mid": out_mid,
                "deep": out_deep,
                "final": out_final,
            },
            "distill_features": {
                "shallow": feat_shallow,
                "mid": feat_mid,
                "deep": feat_deep,
                "final": feat_final,
            },
        }


class EEGNetMultiExitLiteV2(ClassificationModule):
    def __init__(
            self,
            n_channels: int,
            n_classes: int,
            input_window_samples: int,
            pool_mode: str = "mean",
            F1: int = 8,
            D: int = 2,
            F2: int = 16,
            kernel_length: int = 32,
            drop_prob: float = 0.5,
            pool_time_length: int = 4,
            pool_time_stride: int = 4,
            kernel_length_dw_sep: int = 16,
            pool_time_length_2: int = 8,
            pool_time_stride_2: int = 8,
            **kwargs
    ):
        model = EEGNetMultiExitLiteV2Module(
            n_channels=n_channels,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            pool_mode=pool_mode,
            F1=F1,
            D=D,
            F2=F2,
            kernel_length=kernel_length,
            drop_prob=drop_prob,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            kernel_length_dw_sep=kernel_length_dw_sep,
            pool_time_length_2=pool_time_length_2,
            pool_time_stride_2=pool_time_stride_2,
            byot_proj_dim=kwargs.get("byot_proj_dim", 64),
        )
        super(EEGNetMultiExitLiteV2, self).__init__(model, n_classes, **kwargs)
