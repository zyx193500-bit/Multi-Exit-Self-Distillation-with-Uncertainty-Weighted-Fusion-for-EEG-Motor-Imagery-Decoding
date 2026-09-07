"""
Figure 1: Multi-Exit Self-Distillation Architecture.
Nature-quality schematic — full detail with internal feature dimensions,
clear visual hierarchy, and publication-standard linework.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
import numpy as np

# ── Nature MANDATORY font setup ──────────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 7.5
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ── Nature color palette ─────────────────────────────────────────────────
BK  = '#4A6FA5'   # backbone blue
BK2 = '#7BA3D4'   # backbone light
SH  = '#E8A87C'   # shallow orange
SH2 = '#F5D0B5'   # shallow light
MD  = '#7BC89C'   # mid green
MD2 = '#B5E4C8'   # mid light
DP  = '#6B8EC2'   # deep blue
DP2 = '#B0CAE8'   # deep light
FN  = '#9B7EC4'   # final purple (teacher)
FN2 = '#D0C4E8'   # final light
KD  = '#D9544D'   # KD arrow red
HT  = '#5B7FCA'   # Hint arrow blue
EN  = '#F0D870'   # ensemble gold
PR  = '#E8C050'   # prediction
GR  = '#8F8F8F'   # gray
GR2 = '#E0E0E0'   # light gray
TX  = '#2D2D2D'   # text dark
WH  = '#FFFFFF'
BL  = '#383838'   # border

fig, ax = plt.subplots(1, 1, figsize=(13.5, 6.0))
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 6.0)
ax.set_aspect('equal')
ax.axis('off')
ax.set_facecolor(WH)

# ── Helpers ───────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, fc, ec=BL, lw=0.8, radius=0.08, z=3):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(b)
    return b

def txt(ax, x, y, s, fs=6.5, c=TX, ha='center', va='center', bold=False, z=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=c, fontweight='bold' if bold else 'normal', zorder=z)

def arrow(ax, x1, y1, x2, y2, c=GR, lw=0.7, ls='-', z=2, style='simple'):
    """Draw arrow. style='simple' or 'curved' with rad."""
    if style == 'simple':
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=lw, ls=ls,
                                   mutation_scale=10, shrinkA=0, shrinkB=0),
                    zorder=z)
    elif isinstance(style, (int, float)):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=lw, ls=ls,
                                   mutation_scale=10, shrinkA=0, shrinkB=0,
                                   connectionstyle=f'arc3,rad={style}'),
                    zorder=z)

# ═══════════════════════════════════════════════════════════════════════════
# BACKGROUND SECTION DIVIDERS (subtle gray bands)
# ═══════════════════════════════════════════════════════════════════════════
for (x, w, label) in [(0.55, 2.3, 'CNN Frontend'), (3.05, 5.5, 'Transformer Encoder'), (8.75, 2.2, 'TCN')]:
    rect = Rectangle((x, 0.7), w, 3.55, facecolor='#F7F7F7', edgecolor='#E8E8E8',
                      linewidth=0.3, linestyle='--', zorder=0)
    ax.add_patch(rect)
    txt(ax, x + w/2, 4.35, label, fs=5.5, c='#B0B0B0', z=1)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 1: BACKBONE (y ≈ 2.4)
# ═══════════════════════════════════════════════════════════════════════════
yb = 2.4  # backbone y center
bh = 0.78 # backbone box height

# EEG Input
box(ax, 0.5, yb - bh/2, 0.95, bh, WH, GR, lw=0.6, radius=0.06)
txt(ax, 0.975, yb, 'EEG Input\n(22 × T)', fs=5.5)

# MK-CNN internals — show sub-layers
box(ax, 1.85, yb - bh/2, 2.0, bh, BK2, BK, lw=0.9, radius=0.08)
txt(ax, 2.85, yb + 0.32, 'Multi-Kernel Conv Block', fs=5.8, bold=True)
# sub-layers inside MK-CNN
for j, (sx, sw, sl) in enumerate([(1.95, 0.55, 'Conv\n(k=20)'),
                                   (2.55, 0.55, 'Conv\n(k=32)'),
                                   (3.15, 0.55, 'Conv\n(k=64)')]):
    box(ax, sx, yb - 0.12, sw, 0.35, WH, BK, lw=0.35, radius=0.03, z=4)
    txt(ax, sx + sw/2, yb + 0.055, sl, fs=4.2, c=BK, z=5)
txt(ax, 2.85, yb - 0.35, 'Concat → DW Conv → Pool → 48 × T′', fs=4.8, c=BK, z=5)

# Transformer blocks — each with internal GQA + MLP
n_txf = 5
for i in range(n_txf):
    cx = 4.1 + i * 1.05
    box(ax, cx, yb - bh/2, 0.85, bh, BK2, BK, lw=0.7, radius=0.06)
    txt(ax, cx + 0.425, yb + 0.30, f'Block {i+1}', fs=5.2, bold=True)
    # internal sub-boxes
    box(ax, cx + 0.05, yb + 0.02, 0.35, 0.24, WH, BK, lw=0.25, radius=0.02, z=4)
    txt(ax, cx + 0.225, yb + 0.14, 'GQA', fs=4.0, c=BK, z=5)
    box(ax, cx + 0.45, yb + 0.02, 0.35, 0.24, WH, BK, lw=0.25, radius=0.02, z=4)
    txt(ax, cx + 0.625, yb + 0.14, 'MLP', fs=4.0, c=BK, z=5)
    txt(ax, cx + 0.425, yb - 0.24, 'RoPE', fs=4.2, c=GR, z=5)

# TCN Head
box(ax, 9.45, yb - bh/2, 1.3, bh, BK2, BK, lw=0.9, radius=0.08)
txt(ax, 10.1, yb + 0.30, 'TCN Head', fs=5.8, bold=True)
box(ax, 9.55, yb - 0.02, 0.52, 0.32, WH, BK, lw=0.25, radius=0.02, z=4)
txt(ax, 9.81, yb + 0.14, 'TCN₁', fs=4.2, c=BK, z=5)
box(ax, 10.13, yb - 0.02, 0.52, 0.32, WH, BK, lw=0.25, radius=0.02, z=4)
txt(ax, 10.39, yb + 0.14, 'TCN₂', fs=4.2, c=BK, z=5)
txt(ax, 10.1, yb - 0.35, 'Skip → 64 × T′', fs=4.8, c=BK, z=5)

# Forward arrows (backbone path)
prev_x, prev_w = 1.0, 0.95
cur_x, cur_w = 1.85, 2.0
arrow(ax, prev_x + prev_w/2 + 0.02, yb, cur_x - cur_w/2 - 0.02, yb, GR, lw=0.6, z=1)

prev_x, prev_w = 1.85 + 2.0, 2.0
cur_x, cur_w = 4.1, 0.85
arrow(ax, prev_x + 0.02, yb, cur_x - 0.02, yb, GR, lw=0.6, z=1)

# Between transformer blocks
for i in range(n_txf - 1):
    x1 = 4.1 + i * 1.05 + 0.85
    x2 = 4.1 + (i+1) * 1.05
    arrow(ax, x1 + 0.03, yb, x2 - 0.03, yb, GR, lw=0.5, z=1)

# T5 to TCN
arrow(ax, 4.1 + 4*1.05 + 0.85 + 0.03, yb, 9.45 - 0.03, yb, GR, lw=0.6, z=1)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 2: EXITS (y ≈ 3.55)
# ═══════════════════════════════════════════════════════════════════════════
ye = 3.55  # exit center y
eh = 0.58

# PartitionedExitRouter indicator
def exit_box(ax, x, y, w, h, fc, ec, title, feat_shape, z_out, lw=0.8):
    box(ax, x, y, w, h, fc, ec, lw=lw, radius=0.08)
    txt(ax, x + w/2, y + h - 0.12, title, fs=5.8, bold=True)
    txt(ax, x + w/2, y + h/2 - 0.02, feat_shape, fs=4.8, c=TX)
    # small router indicator
    box(ax, x + 0.06, y + 0.04, 0.46, 0.18, WH, ec, lw=0.25, radius=0.02, z=4)
    txt(ax, x + 0.29, y + 0.13, 'Router', fs=3.8, c=ec, z=5)
    txt(ax, x + w/2, y + 0.02, z_out, fs=5.0, c=TX, bold=True)

# Shallow Exit (above MK-CNN, x=1.85 to 3.85)
exit_box(ax, 2.05, ye - eh/2, 1.6, eh, SH2, SH, 'Shallow Exit', '48 × T′', 'z_shallow ∈ ℝᴮˣᴷ')
# Mid Exit (above T2, x=5.15)
exit_box(ax, 4.7, ye - eh/2, 1.6, eh, MD2, MD, 'Mid Exit', '48 × T′', 'z_mid ∈ ℝᴮˣᴷ')
# Deep Exit (above T5, x=8.35)
exit_box(ax, 7.85, ye - eh/2, 1.6, eh, DP2, DP, 'Deep Exit', '48 × T′', 'z_deep ∈ ℝᴮˣᴷ')
# Final Exit (right of TCN, teacher label)
exit_box(ax, 10.7, ye - eh/2, 1.7, eh, FN2, FN, 'Final Exit (Teacher)', '64 × T′', 'z_final ∈ ℝᴮˣᴷ', lw=1.1)

# ── Vertical connections: backbone → exits ────────────────────────────────
arrow(ax, 2.85, yb + bh/2 + 0.02, 2.85, ye - eh/2 - 0.02, GR, lw=0.55, z=1)
arrow(ax, 5.5, yb + bh/2 + 0.02, 5.5, ye - eh/2 - 0.02, GR, lw=0.55, z=1)
arrow(ax, 8.65, yb + bh/2 + 0.02, 8.65, ye - eh/2 - 0.02, GR, lw=0.55, z=1)
# TCN → Final
arrow(ax, 10.1, yb + bh/2 + 0.02, 10.1, ye - eh/2 - 0.15, GR, lw=0.55, z=1)
arrow(ax, 10.1, ye - eh/2 - 0.15, 11.55 - 1.7/2 - 0.02, ye - eh/2 - 0.15, GR, lw=0.55, z=1)
arrow(ax, 11.55 - 1.7/2 - 0.02, ye - eh/2 - 0.15, 11.55 - 1.7/2 - 0.02, ye - eh/2 - 0.02, GR, lw=0.55, z=1)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 3: DISTILLATION ARROWS (KD & Hint) — above exits
# ═══════════════════════════════════════════════════════════════════════════
yd = 4.02  # distillation arrow y

# ── KD arrows (Final → students, red dashed) ─────────────────────────────
# Final → Deep (straight)
arrow(ax, 11.55 - 1.7/2 + 0.05, yd, 8.65, yd, KD, lw=0.9, ls='--', z=2)
# Final → Mid (curved up)
arrow(ax, 11.55 - 1.7/2 + 0.05, yd + 0.05, 6.3, yd + 0.15, KD, lw=0.9, ls='--', z=2, style=0.25)
# Final → Shallow (curved more)
arrow(ax, 11.55 - 1.7/2 + 0.05, yd + 0.1, 3.65, yd + 0.25, KD, lw=0.9, ls='--', z=2, style=0.45)

# ── Hint arrows (cascade, blue dotted) ────────────────────────────────────
yh = 4.22  # hint arrow y
# Deep → Mid
arrow(ax, 8.65 - 0.3, yh, 6.3 + 0.3, yh, HT, lw=0.7, ls=':', z=2)
# Mid → Shallow
arrow(ax, 5.5 - 0.3, yh, 3.65 + 0.3, yh, HT, lw=0.7, ls=':', z=2)
# Final → Deep
arrow(ax, 11.55 - 1.7/2 + 0.3, yh, 8.65 + 0.3, yh, HT, lw=0.7, ls=':', z=2)

# ── Projection heads indicator ────────────────────────────────────────────
txt(ax, 6.0, 4.42, 'Proj → L2 normalize → MSE', fs=5.0, c=HT, z=2)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 4: DYNAMIC ENSEMBLE + PREDICTION (y ≈ 1.2)
# ═══════════════════════════════════════════════════════════════════════════
ye2 = 1.3  # ensemble center y

# Ensemble box (receives all 4 z's)
box(ax, 9.0, ye2 - 0.35, 3.2, 0.7, EN, '#C0A030', lw=0.9, radius=0.08)
txt(ax, 10.6, ye2 + 0.18, 'Dynamic Weighted Ensemble', fs=6.0, bold=True)
txt(ax, 10.6, ye2 - 0.08, 'z_ensemble = Σ wₖ · zₖ', fs=5.5, c='#6B5B00')
txt(ax, 10.6, ye2 - 0.22, 'wₖ ∝ Softmax(exp(−sₖ) · priorₖ)', fs=5.0, c='#6B5B00')

# Prediction box
box(ax, 9.8, 0.55, 2.0, 0.5, '#F5A623', '#C07810', lw=0.8, radius=0.06)
txt(ax, 10.8, 0.80, 'Final Prediction', fs=5.8, bold=True)

# Ensemble → prediction arrow
arrow(ax, 10.6, ye2 - 0.35 - 0.02, 10.6, 0.80 + 0.02, GR, lw=0.7, z=2)

# ── Vertical connections: exits → ensemble ────────────────────────────────
# Collect arrows from all four exits down to ensemble
# Each exit z feeds into ensemble

# z_final → ensemble (direct drop)
arrow(ax, 11.55, ye - eh/2 - 0.02, 11.55, ye2 + 0.35 + 0.02, GR, lw=0.5, z=1)
# z_shallow → ensemble (far left, longer path)
arrow(ax, 2.85, ye - eh/2 - 0.02, 2.85, 0.9, GR, lw=0.4, z=1)
arrow(ax, 2.85, 0.9, 10.6 - 1.6, 0.9, GR, lw=0.4, z=1)
arrow(ax, 10.6 - 1.6, 0.9, 10.6 - 1.6, ye2 + 0.35 + 0.02, GR, lw=0.4, z=1)
# z_mid → ensemble
arrow(ax, 5.5, ye - eh/2 - 0.02, 5.5, 1.05, GR, lw=0.4, z=1)
arrow(ax, 5.5, 1.05, 10.6 - 0.6, 1.05, GR, lw=0.4, z=1)
arrow(ax, 10.6 - 0.6, 1.05, 10.6 - 0.6, ye2 + 0.35 + 0.02, GR, lw=0.4, z=1)
# z_deep → ensemble
arrow(ax, 8.65, ye - eh/2 - 0.02, 8.65, 1.15, GR, lw=0.4, z=1)
arrow(ax, 8.65, 1.15, 10.6 + 0.4, 1.15, GR, lw=0.4, z=1)
arrow(ax, 10.6 + 0.4, 1.15, 10.6 + 0.4, ye2 + 0.35 + 0.02, GR, lw=0.4, z=1)

# ═══════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════
ly = 0.22
ax.plot([0.6, 1.1], [ly, ly], '--', color=KD, lw=1.0); txt(ax, 1.2, ly, 'Logit KD', fs=5.8, c=KD, ha='left')
ax.plot([2.5, 3.0], [ly, ly], ':', color=HT, lw=1.0); txt(ax, 3.1, ly, 'Hint (Feature MSE)', fs=5.8, c=HT, ha='left')
ax.plot([4.8, 5.3], [ly, ly], '-', color=GR, lw=0.6); txt(ax, 5.4, ly, 'Data flow', fs=5.8, c=GR, ha='left')
# Teacher indicator
box(ax, 6.8, ly - 0.06, 0.12, 0.12, FN, FN, lw=0.6, radius=0.01, z=3)
txt(ax, 7.0, ly, 'Teacher', fs=5.8, c=FN, ha='left', bold=True)
# Student
box(ax, 8.1, ly - 0.06, 0.12, 0.12, SH2, SH, lw=0.4, radius=0.01, z=3)
txt(ax, 8.3, ly, 'Student', fs=5.8, c=SH, ha='left')

# ═══════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════
txt(ax, 6.75, 5.75, 'Figure 1 | Architecture of the proposed multi-exit self-distillation framework.',
    fs=8.5, bold=True)
txt(ax, 6.75, 5.55, 'Four exits (Shallow, Mid, Deep, Final) are attached to a TCFormer backbone. '
    'The Final exit (teacher) supervises student exits via logit-based KD (red dashed) and '
    'hint-based feature distillation (blue dotted). All four exit logits are fused through '
    'a dynamic uncertainty-weighted ensemble to produce the final prediction.',
    fs=5.5, c=GR)

# ── Save ──────────────────────────────────────────────────────────────────
out_dir = r'C:\Users\zyx19\Desktop\zhengliu\蒸馏\TCFormer\analysis\paper_drafts'
out = f'{out_dir}\\fig1_architecture'

fig.tight_layout(pad=0.4)
fig.savefig(f'{out}.svg', bbox_inches='tight', dpi=300, facecolor=WH, edgecolor='none')
fig.savefig(f'{out}.pdf', bbox_inches='tight', dpi=300, facecolor=WH, edgecolor='none')
fig.savefig(f'{out}.png', bbox_inches='tight', dpi=300, facecolor=WH, edgecolor='none')
plt.close(fig)
print("Done: fig1_architecture.svg, .pdf, .png")
