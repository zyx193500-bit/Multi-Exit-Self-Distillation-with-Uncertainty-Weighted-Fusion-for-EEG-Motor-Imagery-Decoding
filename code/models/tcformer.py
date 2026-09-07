"""
Model Name: TCFormer - Temporal Convolutional Transformer for EEG-Based Motor Imagery Decoding

Citation:
H. Altaheri, F. Karray, and A.H. Karimi (2025). Temporal Convolutional 
Transformer for EEG Based Motor Imagery Decoding. *Scientific Reports*.

Repository:
Original implementation available at: https://github.com/altaheri/TCFormer

Note:
All default hyperparameters and configurations are consistent with those reported 
in the published paper.
"""

# Core Libraries
import torch
from torch import nn, Tensor

# Utility Libraries
from einops import rearrange
from einops.layers.torch import Rearrange

# Local application-specific imports
from .classification_module import ClassificationModule
from .modules import CausalConv1d, Conv1dWithConstraint
from .channel_group_attention import ChannelGroupAttention
from utils.weight_initialization import glorot_weight_zero_bias
from utils.latency  import measure_latency


# ------------------------------------------------------------------------------- #
class MultiKernelConvBlock(nn.Module):
    """
    Multi-Kernel Convolution Block for EEG Feature Extraction.

    This block applies multiple temporal convolutions with different kernel sizes,
    followed by channel-wise and temporal processing with optional group attention.
    """
    def __init__(
        self,
        n_channels: int,
        temp_kernel_lengths: tuple = (20, 32, 64),
        F1: int = 32,
        D: int = 2,
        pool_length_1: int = 8,
        pool_length_2: int = 7,
        dropout: float = 0.4,
        d_group: int = 16,
        use_group_attn: bool = True,
    ):
        super().__init__()

        # --- 1. one temporal conv per kernel -----------------
        self.rearrange = Rearrange("b c seq -> b 1 c seq")
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.ConstantPad2d((k//2-1, k//2, 0, 0) if k % 2 == 0 else (k//2, k//2, 0, 0), 0),
                nn.Conv2d(1, F1, (1, k), bias=False),
                nn.BatchNorm2d(F1),
                # nn.ELU() 
            )
            for k in temp_kernel_lengths
        ])
            
        # --- 2. shared processing after concatenation --------
        n_groups = len(temp_kernel_lengths)
        self.d_model = d_group * n_groups
        
        # Channel Reduction Stage 1: Grouped Pointwise (1×1) Conv (F1 * n_groups → d_model)
            # Channel Reduction Stage 1 takes more time and did not enhance performance,
            # so we set it as always False.
        self.use_channel_reduction_1  = False
        # self.use_channel_reduction_1  = (self.d_model != F1 * n_groups)
        if self.use_channel_reduction_1:
            self.channel_reduction_1 = nn.Sequential(
                    nn.Conv2d(F1 * n_groups, self.d_model, (1, 1), bias=False, groups=n_groups),
                    nn.BatchNorm2d(self.d_model),
                )
            
        # Depth-wise convolution across EEG channels
        F2 = self.d_model * D if self.use_channel_reduction_1 else F1 * n_groups * D 
        self.channel_DW_conv = nn.Sequential(
            nn.Conv2d(F1 * n_groups, F2, (n_channels, 1), bias=False, groups=F1 * n_groups),
            nn.BatchNorm2d(F2),
            nn.ELU(),
        )
        self.pool1 = nn.AvgPool2d((1, pool_length_1))
        self.drop1 = nn.Dropout(dropout)
        
        # Channel Reduction Stage 2: Grouped Pointwise (1×1) Conv (F2 → d_model)
        self.use_channel_reduction_2 = (self.d_model != F2)
        if self.use_channel_reduction_2:
            self.channel_reduction_2 = nn.Sequential(
                    nn.Conv2d(F2, self.d_model, (1, 1), bias=False, groups=n_groups),
                    nn.BatchNorm2d(self.d_model),
                )

        # Grouped temporal convolution (1 × 16) per group
        self.temporal_conv_2 = nn.Sequential(
            nn.ConstantPad2d((7, 8, 0, 0), 0),
            nn.Conv2d(self.d_model, self.d_model, (1, 16), padding='valid',
                       bias=False, groups=n_groups),
            nn.BatchNorm2d(self.d_model),
            nn.ELU(),
        )

        # Enable grouped attention only if multiple groups are used (two temp kernels or more)
        self.use_group_attn = False if n_groups == 1 else use_group_attn
        if self.use_group_attn:
            self.group_attn = ChannelGroupAttention(
                in_channels=self.d_model,
                num_groups=n_groups, 
            )
        
        self.pool2 = nn.AvgPool2d((1, pool_length_2))
        self.drop2 = nn.Dropout(dropout)

        # Initialize weights
        glorot_weight_zero_bias(self)

    def forward(self, x):
        # --- 1. one temporal conv per kernel -----------------
        x = self.rearrange(x)         # (B, 1, C, T)
        feats = [conv(x) for conv in self.temporal_convs] # list of (B, F1, C, T')
        x = torch.cat(feats, dim=1)   # concat on channel dim # [B, F1 x n_groups, C, T]

        # --- 2. shared processing after concatenation --------
        # Channel Reduction Stage 1: (F1 * n_groups → d_model)
        if self.use_channel_reduction_1:
            x = self.channel_reduction_1(x)
        
        # EEG channel depth-wise conv
        x = self.channel_DW_conv(x)                         
        x = self.pool1(x)                                # temporal pooling
        x = self.drop1(x)                                # dropout

        # Channel Reduction Stage 2: (F2 → d_model)
        if self.use_channel_reduction_2:
            x = self.channel_reduction_2(x)
       
        # Grouped Temporal Convolution (1 × 16) applied independently to each group
        x = self.temporal_conv_2(x)                      
        
        # Group attention (optional) 
        if self.use_group_attn:        
            x = x + self.group_attn(x)   # Residual connection 
        
        x = self.pool2(x)                                # temporal pooling
        x = self.drop2(x)                                # dropout
            
        return x.squeeze(2)
# ------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------- #
class TCNBlock(nn.Module):
    def __init__(self, kernel_length: int = 4, n_filters: int = 32, dilation: int = 1,
                 n_groups: int = 1, dropout: float = 0.3):
        super().__init__()
        self.conv1 = CausalConv1d(n_filters, n_filters, kernel_size=kernel_length,
                                  dilation=dilation, groups=n_groups)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.nonlinearity1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(n_filters, n_filters, kernel_size=kernel_length,
                                  dilation=dilation, groups=n_groups)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.nonlinearity2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout)

        self.nonlinearity3 = nn.ELU()

        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, input):
        x = self.drop1(self.nonlinearity1(self.bn1(self.conv1(input))))
        x = self.drop2(self.nonlinearity2(self.bn2(self.conv2(x))))
        x = self.nonlinearity3(input + x)
        return x

class TCN(nn.Module):
    def __init__(self, depth: int = 2, kernel_length: int = 4, n_filters: int = 32,
                 n_groups: int = 1, dropout: float = 0.3):
        super(TCN, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** i
            self.blocks.append(TCNBlock(kernel_length, n_filters, dilation, n_groups, dropout))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class ClassificationHead(nn.Module):
    """
    Maps TCN features to class logits and optionally averages across groups.
    Expected input shape: (batch, d_model, 1)   ← after time-step selection.
    Output shape:        (batch, n_classes)
    """
    def __init__(
        self,
        d_features: int,
        n_groups: int,
        n_classes: int,
        kernel_size: int = 1,
        max_norm: float = 0.25,
    ):
        super().__init__()
        self.n_groups   = n_groups
        self.n_classes  = n_classes

        # self.drop = nn.Dropout(0.3)

        # point-wise (1 × 1) grouped conv = class projection per group
        self.linear = Conv1dWithConstraint(
            in_channels=d_features,
            out_channels=n_classes * n_groups,
            kernel_size=kernel_size,
            groups=n_groups,
            max_norm=max_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d_model, Tc)  →  logits: (B, n_classes)
        """
        # (B, n_classes*n_groups, 1) → squeeze last dim
        # x = self.drop(x) 
        x = self.linear(x).squeeze(-1)

        # (B, n_groups, n_classes) → mean over groups
        x = x.view(x.size(0), self.n_groups, self.n_classes).mean(dim=1)
        return x

class TCNHead(nn.Module):
    def __init__(self, d_features: int = 64, n_groups: int = 1, tcn_depth: int = 2, 
                 kernel_length: int = 4,  dropout_tcn: float = 0.3, n_classes: int = 4):
        super().__init__()
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.tcn = TCN(tcn_depth, kernel_length, d_features, n_groups, dropout_tcn)

        # self.linear = Conv1dWithConstraint(d_model, n_classes*n_groups, kernel_size=1, 
        #                                         groups=n_groups, max_norm=0.25)   
        self.classifier = ClassificationHead(
            d_features=d_features,
            n_groups=n_groups,
            n_classes=n_classes,
        )     
    def forward(self, x):
        x = self.tcn(x)
        x = x[:, :, -1:]

        x = self.classifier(x)   # (B, n_classes)
        # tcn_out = self.linear(tcn_out).squeeze(-1)

        # tcn_out = tcn_out.view(x.shape[0], self.n_groups, self.n_classes)
        # tcn_out = tcn_out.mean(dim=1) 

        return x
# ------------------------------------------------------------------------------- #
    
# ------------------------------------------------------------------------------- #
#  Rotary positional embedding utilities
# Adapted from GPT‑NeoX & LLaMA implementations
def _build_rotary_cache(head_dim: int, seq_len: int, device: torch.device):
    """Return cos & sin tensors of shape (seq_len, head_dim)."""
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    seq_idx = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(seq_idx, theta)                 # (seq, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)             # duplicate for even/odd
    cos, sin = emb.cos(), emb.sin()
    return cos, sin                                     # each: (seq, head_dim)

def _rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):  # q/k: (B, h, T, d)
    def _rotate(x):                                        # half rotation
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)
    q_out = (q * cos) + (_rotate(q) * sin)
    k_out = (k * cos) + (_rotate(k) * sin)
    return q_out, k_out

# ---------------------------------------------------------
#  Grouped‑Query Self‑Attention (GQA) with RoPE
class _GQAttention(nn.Module):
    """Grouped‑Query Attention (num_q_heads >= num_kv_heads)."""
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int, dropout: float = 0.3):
        super().__init__()
        assert d_model % num_q_heads == 0, "d_model must divide num_q_heads"
        assert num_q_heads % num_kv_heads == 0, "q_heads must be multiple of kv_heads"
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.scale = self.head_dim ** -0.5
        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        _xavier_zero_bias(self)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:  # x (B,T,C)
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)  # (B, hq, T, d)
        kv = self.kv_proj(x)
        kv = kv.view(B, T, self.num_kv_heads, 2, self.head_dim)
        k, v = kv[..., 0, :].transpose(1, 2), kv[..., 1, :].transpose(1, 2)  # each (B, hk, T, d)
        # replicate k/v across groups
        repeat_factor = self.num_q_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
        # Rotary positional embedding
        q, k = _rope(q, k, cos[:T, :], sin[:T, :])
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B,h,T,T)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = attn @ v                                  # (B,h,T,d)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

# ---------------------------------------------------------
# from timm.models.layers import DropPath  # for stochastic depth
class DropPath(nn.Module):
    """Implements stochastic depth / DropPath
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    implementation from https://github.com/huggingface/pytorch-image-models/blob/a6e8598aaf90261402f3e9e9a3f12eac81356e9d/timm/models/layers/drop.py#L140
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # shape: (batch_size, 1, 1, ..., 1)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

# ---------------------------------------------------------
#  Transformer encoder block (Pre‑LN, GQA, MLP‑GEGLU)
class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, q_heads: int, kv_heads: int, mlp_ratio: int = 2, dropout=0.4, drop_path_rate=0.25):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _GQAttention(d_model, q_heads, kv_heads, dropout)
        #self.drop_path = nn.Dropout(dropout)
        self.drop_path   = DropPath(drop_path_rate)  # for stochastic depth
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),    # expands per-token features
            nn.GELU(),                                      # non-linearity
            nn.Linear(mlp_ratio * d_model, d_model),    # compresses back to d_model
            nn.Dropout(dropout),                            # dropout# regularisation
        )
    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), cos, sin))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
        
#   helper
def _xavier_zero_bias(module: nn.Module) -> None:
    """Apply Xavier‑uniform + zero bias to every conv/linear inside *module*."""
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
# ------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------- #

class Bottleneck1D(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        self.conv3 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += residual
        return self.relu(out)

def _select_num_heads(channels: int, preferred_heads: int) -> int:
    for heads in range(min(preferred_heads, channels), 0, -1):
        if channels % heads == 0:
            return heads
    return 1

def _build_sinusoidal_positional_encoding(seq_len: int, dim: int, device: torch.device):
    position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0, device=device)) / dim)
    )
    pe = torch.zeros(seq_len, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    if dim % 2 == 0:
        pe[:, 1::2] = torch.cos(position * div_term)
    else:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])
    return pe.unsqueeze(0)

class AttentionPooling1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.score = nn.Linear(channels, 1)

    def forward(self, x):
        tokens = x.transpose(1, 2)
        tokens = self.norm(tokens)
        attn = torch.softmax(self.score(tokens), dim=1)
        pooled = torch.sum(tokens * attn, dim=1)
        return pooled

class MHSABottleneck1D(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.norm1 = nn.LayerNorm(mid_channels)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=mid_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.conv3 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential()
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))

        tokens = out.permute(0, 2, 1)
        tokens = tokens + _build_sinusoidal_positional_encoding(
            tokens.size(1), tokens.size(2), tokens.device
        )
        attn_input = self.norm1(tokens)
        attn_out, _ = self.mhsa(attn_input, attn_input, attn_input)
        tokens = tokens + self.attn_dropout(attn_out)
        out = tokens.permute(0, 2, 1)

        out = self.bn3(self.conv3(out))
        out += residual
        return self.relu(out)

class FeatureClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=4, bottleneck_type="conv", num_heads=4, dropout=0.1):
        super().__init__()
        self.bottleneck_type = bottleneck_type

        if bottleneck_type == "conv":
            mid_channels = max(in_channels // 2, 1)
            self.bottleneck = Bottleneck1D(in_channels, mid_channels, in_channels)
        elif bottleneck_type == "mhsa":
            mid_channels = max(in_channels // 2, 1)
            num_heads = _select_num_heads(mid_channels, num_heads)
            self.bottleneck = MHSABottleneck1D(
                in_channels,
                mid_channels,
                in_channels,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.bottleneck = nn.Identity()

        if bottleneck_type == "mhsa":
            self.pool = AttentionPooling1D(in_channels)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        if self.bottleneck_type in ["conv", "mhsa"]:
            x = self.bottleneck(x)
        if self.bottleneck_type == "mhsa":
            x = self.pool(x)
        else:
            x = self.pool(x).squeeze(-1)
        out = self.fc(x)
        return out

class PartitionedExitRouter1D(nn.Module):
    def __init__(self, channels: int, shared_ratio: float = 0.5, private_ratio: float = 0.25, dropout: float = 0.1):
        super().__init__()
        shared_channels = max(int(channels * shared_ratio), 1)
        private_channels = max(int(channels * private_ratio), 1)
        selected_channels = min(shared_channels + private_channels, channels)
        self.selected_channels = selected_channels
        self.proj = nn.Sequential(
            nn.Conv1d(selected_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        routed = x[:, :self.selected_channels, :]
        return self.proj(routed)

class DistillationProjectionHead(nn.Module):
    def __init__(self, in_channels: int, proj_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(proj_dim, in_channels // 2, 32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x).squeeze(-1)
        return self.proj(x)

class TCFormerModule(nn.Module):
    def __init__(self,                 
            n_channels: int ,
            n_classes: int,
            F1: int = 16,
            temp_kernel_lengths=(16, 32, 64),
            pool_length_1: int = 8,
            pool_length_2: int = 7,
            D: int = 2,
            dropout_conv: float = 0.3,
            d_group: int = 16,
            tcn_depth: int = 2,
            kernel_length_tcn: int = 4,
            dropout_tcn: float = 0.3,
            use_group_attn: bool = True,
            kv_heads: int = 4, 
            q_heads: int = 8, 
            trans_dropout: float = 0.4,
            drop_path_max: float = 0.25, 
            trans_depth: int = 5,
            byot_proj_dim: int = 128,
        ):
        super().__init__()
        self.n_classes = n_classes
        self.n_groups = len(temp_kernel_lengths)
        self.d_model = d_group*self.n_groups

        self.rearrange = Rearrange("b c seq -> b seq c")

        self.conv_block = MultiKernelConvBlock(n_channels, temp_kernel_lengths, F1, D, 
                                               pool_length_1, pool_length_2, dropout_conv, 
                                               d_group, use_group_attn)
        self.mix = nn.Sequential(
            nn.Conv1d(
                in_channels=self.d_model,
                out_channels=self.d_model,
                kernel_size=1,              # across channels only
                groups=1, bias=False),
            nn.BatchNorm1d(self.d_model),
            nn.SiLU()
        )

        # linearly increasing drop path rates from 0 to drop_path_max for deeper layers
        # drop_rates = torch.linspace(0.0, drop_path_max, trans_depth)
        # Quadratically increasing drop path rates from 0 to drop_path_max for deeper layers
        drop_rates = torch.linspace(0, 1, trans_depth) ** 2 * drop_path_max

        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)
        self.transformer = nn.ModuleList([
            _TransformerBlock(self.d_model, q_heads, kv_heads, dropout=trans_dropout, 
                              drop_path_rate=drop_rates[i].item())
            for i in range(trans_depth)
        ])

        self.reduce = nn.Sequential(
            Rearrange("b t c -> b c t"),            # 1. rearrange for Conv1d over channels
            nn.Conv1d(in_channels=self.d_model,     # 2. 1x1 conv over channel dim
                out_channels=d_group,
                kernel_size=1,              
                groups=1, bias=False),
            nn.BatchNorm1d(d_group),
            nn.SiLU(),
        )

        self.tcn_head = TCNHead(d_group*(self.n_groups+1), (self.n_groups+1), tcn_depth, 
                                kernel_length_tcn, dropout_tcn, n_classes)

        # 4 Classifiers
        # 1. Shallow (After ConvBlock) - uses 1x1, 3x3, 1x1 Conv Bottleneck
        self.router_shallow = PartitionedExitRouter1D(self.d_model, shared_ratio=0.5, private_ratio=0.25, dropout=dropout_conv)
        self.classifier_shallow = FeatureClassifier(
            self.d_model,
            num_classes=n_classes,
            bottleneck_type="conv",
        )
        # 2. Mid (After half of bottlenecks) - uses 1x1, MHSA, 1x1 Bottleneck
        self.router_mid = PartitionedExitRouter1D(self.d_model, shared_ratio=0.5, private_ratio=0.25, dropout=trans_dropout)
        self.classifier_mid = FeatureClassifier(
            self.d_model,
            num_classes=n_classes,
            bottleneck_type="mhsa",
            num_heads=q_heads,
            dropout=trans_dropout,
        )
        # 3. Deep (After all bottlenecks) - uses 1x1, MHSA, 1x1 Bottleneck
        self.router_deep = PartitionedExitRouter1D(self.d_model, shared_ratio=0.5, private_ratio=0.25, dropout=trans_dropout)
        self.classifier_deep = FeatureClassifier(
            self.d_model,
            num_classes=n_classes,
            bottleneck_type="mhsa",
            num_heads=q_heads,
            dropout=trans_dropout,
        )
        # 4. Final (After TCN) - uses 1x1, MHSA, 1x1 Bottleneck on the full temporal map
        self.router_final = PartitionedExitRouter1D(d_group*(self.n_groups+1), shared_ratio=0.6, private_ratio=0.3, dropout=dropout_tcn)
        self.classifier_final = FeatureClassifier(
            d_group * (self.n_groups + 1),
            num_classes=n_classes,
            bottleneck_type="mhsa",
            num_heads=kv_heads,
            dropout=dropout_tcn,
        )
        self.proj_shallow = DistillationProjectionHead(self.d_model, byot_proj_dim, dropout_conv)
        self.proj_mid = DistillationProjectionHead(self.d_model, byot_proj_dim, trans_dropout)
        self.proj_deep = DistillationProjectionHead(self.d_model, byot_proj_dim, trans_dropout)
        self.proj_final = DistillationProjectionHead(d_group*(self.n_groups+1), byot_proj_dim, dropout_tcn)

        # # Kaiming (He) init is recommended for Conv layers that precede SiLU / ReLU
        # nn.init.kaiming_normal_(self.reduce[0].weight, nonlinearity="linear") 

    def forward(self, x):      # x: [B, C_electrodes, T]
        conv_features = self.conv_block(x)         
        B, C, T = conv_features.shape

        # Shallow output
        routed_shallow = self.router_shallow(conv_features)
        out_shallow = self.classifier_shallow(routed_shallow)
        feat_shallow = self.proj_shallow(routed_shallow)
        
        tokens = self.rearrange(self.mix(conv_features))  # tokens: [B, T, C]
        cos, sin = self._rotary_cache(T , tokens.device)
        
        out_mid = None
        feat_mid = None
        mid_idx = len(self.transformer) // 2
        for i, blk in enumerate(self.transformer):
            tokens = blk(tokens, cos, sin)
            if i == mid_idx - 1:
                # Transpose back to [B, C, T] for classifier
                routed_mid = self.router_mid(tokens.transpose(1, 2))
                out_mid = self.classifier_mid(routed_mid)
                feat_mid = self.proj_mid(routed_mid)
                
        # Deep output
        routed_deep = self.router_deep(tokens.transpose(1, 2))
        out_deep = self.classifier_deep(routed_deep)
        feat_deep = self.proj_deep(routed_deep)
        
        tran_features = self.reduce(tokens)
        features = torch.cat((conv_features, tran_features), dim=1)
        
        tcn_out = self.tcn_head.tcn(features)
        routed_final = self.router_final(tcn_out)
        out_final = self.classifier_final(routed_final)
        feat_final = self.proj_final(routed_final)
        
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
        
    def _rotary_cache(self, seq_len: int, device: torch.device):
        """Build (or reuse) RoPE caches for the current sequence length."""
        head_dim = self.transformer[0].attn.head_dim  # use per‑head dimension, **not** d_model
        if (self._cos is None) or (self._cos.shape[0] < seq_len):
            cos, sin = _build_rotary_cache(head_dim, seq_len, device)
            self._cos, self._sin = cos.to(device), sin.to(device)
        return self._cos, self._sin

class TCFormer(ClassificationModule):
    def __init__(self,
            n_channels: int,
            n_classes: int,
            F1: int = 16,
            temp_kernel_lengths: tuple = (16, 32, 64),
            pool_length_1: int = 8,
            pool_length_2: int = 7,
            D: int = 2,
            dropout_conv: float = 0.3,
            d_group: int = 16,
            tcn_depth: int = 2,
            kernel_length_tcn: int = 4,
            dropout_tcn: float = 0.3,
            use_group_attn: bool = True,
            q_heads: int = 8, 
            kv_heads: int = 4,
            trans_depth: int = 5,    
            trans_dropout: float = 0.4,
            **kwargs
        ):
        model = TCFormerModule(
            n_channels=n_channels,
            n_classes=n_classes,
            F1=F1,
            temp_kernel_lengths=temp_kernel_lengths,
            pool_length_1=pool_length_1,
            pool_length_2=pool_length_2,
            D=D,
            dropout_conv=dropout_conv,
            d_group=d_group,
            tcn_depth=tcn_depth,
            kernel_length_tcn=kernel_length_tcn,
            dropout_tcn=dropout_tcn,
            use_group_attn=use_group_attn,
            q_heads = q_heads, 
            kv_heads = kv_heads, 
            trans_depth = trans_depth,
            trans_dropout = trans_dropout,
            byot_proj_dim=kwargs.get("byot_proj_dim", 128),
        )
        super().__init__(model, n_classes, **kwargs)
    
    @staticmethod
    def benchmark(input_shape, device="cuda:0", warmup=100, runs=500):
        return measure_latency(TCFormer(22, 4), input_shape, device, warmup, runs)
    

if __name__ == "__main__":
    # Example usage: run benchmark with dummy input shape (batch, channels, time)
    C, T = 22, 1000  # adjust as needed
    print(TCFormer.benchmark((1, C, T)))
