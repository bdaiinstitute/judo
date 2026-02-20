"""Generate architecture diagram comparing old CPU vs new GPU MPC pipeline."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Colors — light theme for white-background slides
C_BG = "#FFFFFF"
C_ACCENT_OLD = "#C0392B"
C_ACCENT_NEW = "#2471A3"
C_GPU = "#1E8449"
C_GPU_BG = "#EBF5FB"
C_CPU = "#922B21"
C_CPU_BG = "#FDEDEC"
C_SIM = "#D35400"
C_CTRL = "#7D3C98"
C_TEXT = "#1C2833"
C_SUBTEXT = "#5D6D7E"
C_ARROW = "#2E86C1"
C_ARROW_OLD = "#C0392B"
C_BOX_STROKE = "#ABB2B9"

# Use non-equal aspect so we get full use of the figure area
fig = plt.figure(figsize=(30, 16), facecolor=C_BG)
ax_old = fig.add_axes([0.01, 0.05, 0.25, 0.90], facecolor=C_BG)
ax_new = fig.add_axes([0.28, 0.05, 0.71, 0.90], facecolor=C_BG)


def draw_box(ax, xy, w, h, color, alpha=0.9, lw=1.5, ec=C_BOX_STROKE, zorder=1, radius=0.008):
    box = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad={radius}",
                         facecolor=color, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    return box


def draw_arrow(ax, start, end, color=C_ARROW, lw=3, style="-|>", zorder=5):
    arrow = FancyArrowPatch(start, end, arrowstyle=style, color=color,
                            linewidth=lw, mutation_scale=20, zorder=zorder,
                            connectionstyle="arc3,rad=0.0")
    ax.add_patch(arrow)


# ============================================
# LEFT PANEL: Old Architecture (CPU) — compact
# ============================================
ax_old.set_xlim(0, 1)
ax_old.set_ylim(0, 1)
ax_old.axis('off')

# Title
ax_old.text(0.5, 0.97, "CPU Threaded Rollout", ha='center', va='top',
            fontsize=24, fontweight='bold', color=C_ACCENT_OLD, zorder=10)
ax_old.text(0.5, 0.92, "mujoco_extensions + ONNX Runtime", ha='center', va='top',
            fontsize=15, color=C_SUBTEXT, zorder=10)

# CPU badge
draw_box(ax_old, (0.05, 0.83), 0.90, 0.05, C_CPU_BG, ec=C_CPU, lw=2)
ax_old.text(0.5, 0.855, "CPU (x86_64)", ha='center', va='center',
            fontsize=17, fontweight='bold', color=C_CPU, zorder=10)

# MJSimulation
draw_box(ax_old, (0.08, 0.72), 0.84, 0.08, "#FFF3E0", ec=C_SIM, lw=2)
ax_old.text(0.5, 0.76, "MJSimulation", ha='center', va='center',
            fontsize=19, fontweight='bold', color=C_SIM, zorder=10)

draw_arrow(ax_old, (0.5, 0.72), (0.5, 0.67), color=C_ARROW_OLD)

# Controller
draw_box(ax_old, (0.08, 0.57), 0.84, 0.10, "#F4ECF7", ec=C_CTRL, lw=2)
ax_old.text(0.5, 0.635, "Controller", ha='center', va='center',
            fontsize=19, fontweight='bold', color=C_CTRL, zorder=10)
ax_old.text(0.5, 0.59, "CEM + Spline", ha='center', va='center',
            fontsize=14, color=C_SUBTEXT, zorder=10)

draw_arrow(ax_old, (0.5, 0.57), (0.5, 0.52), color=C_ARROW_OLD)

# RolloutBackend
draw_box(ax_old, (0.05, 0.17), 0.90, 0.35, "#FDEDEC", ec=C_ACCENT_OLD, lw=2)
ax_old.text(0.5, 0.49, "RolloutBackend", ha='center', va='center',
            fontsize=18, fontweight='bold', color=C_ACCENT_OLD, zorder=10)
ax_old.text(0.5, 0.45, "C++ threaded_rollout", ha='center', va='center',
            fontsize=14, color=C_SUBTEXT, zorder=10)

# Thread grid: 32 threads in 4x8
n_threads = 32
cols, rows = 8, 4
bw, bh = 0.088, 0.045
gap_x, gap_y = 0.015, 0.012
total_w = cols * bw + (cols - 1) * gap_x
x_start = (1 - total_w) / 2
y_top = 0.40

for i in range(n_threads):
    r, c = divmod(i, cols)
    bx = x_start + c * (bw + gap_x)
    by = y_top - r * (bh + gap_y)
    draw_box(ax_old, (bx, by), bw, bh, "#F9EBEA", ec=C_ACCENT_OLD, lw=1, alpha=0.8, radius=0.004)
    ax_old.text(bx + bw / 2, by + bh / 2, f"T{i}", ha='center', va='center',
                fontsize=9, color=C_ACCENT_OLD, zorder=10, fontweight='bold')

ax_old.text(0.5, 0.19, "32 CPU threads  |  ONNX + mj_step", ha='center', va='center',
            fontsize=13, color=C_SUBTEXT, zorder=10)

# Bottom label
draw_box(ax_old, (0.10, 0.04), 0.80, 0.08, "#FFF3E0", ec=C_SIM, lw=2)
ax_old.text(0.5, 0.08, "32 Sequential Rollouts", ha='center', va='center',
            fontsize=18, fontweight='bold', color=C_SIM, zorder=10)


# ============================================
# RIGHT PANEL: New Architecture (GPU) — large
# ============================================
ax_new.set_xlim(0, 1)
ax_new.set_ylim(0, 1)
ax_new.axis('off')

n_ctrl = 10

# Title
ax_new.text(0.5, 0.97, "GPU Batched MPC Pipeline", ha='center', va='top',
            fontsize=30, fontweight='bold', color=C_ACCENT_NEW, zorder=10)
ax_new.text(0.5, 0.925, "mujoco_warp  +  PyTorch  +  BatchedControllers", ha='center', va='top',
            fontsize=17, color=C_SUBTEXT, zorder=10)

# CPU section (sims)
draw_box(ax_new, (0.02, 0.82), 0.96, 0.07, C_CPU_BG, ec=C_CPU, lw=1.5, alpha=0.5)
ax_new.text(0.04, 0.885, "CPU", ha='left', va='center',
            fontsize=16, fontweight='bold', color=C_CPU, zorder=10)

# Simulations
sim_w = 0.07
sim_gap = 0.015
total_sims_w = n_ctrl * sim_w + (n_ctrl - 1) * sim_gap
sim_x_start = (1 - total_sims_w) / 2

for i in range(n_ctrl):
    sx = sim_x_start + i * (sim_w + sim_gap)
    draw_box(ax_new, (sx, 0.83), sim_w, 0.05, "#FFF3E0", ec=C_SIM, lw=1.5, radius=0.004)
    ax_new.text(sx + sim_w / 2, 0.855, f"Sim {i}", ha='center', va='center',
                fontsize=12, fontweight='bold', color=C_SIM, zorder=10)

ax_new.text(sim_x_start + n_ctrl * (sim_w + sim_gap) + 0.01, 0.855, "\u2026",
            ha='left', va='center', fontsize=20, fontweight='bold', color=C_SIM, zorder=10)

draw_arrow(ax_new, (0.5, 0.82), (0.5, 0.78), color=C_ARROW)

# GPU section
draw_box(ax_new, (0.02, 0.02), 0.96, 0.80, C_GPU_BG, ec=C_GPU, lw=2.5, alpha=0.4)
ax_new.text(0.04, 0.81, "GPU (NVIDIA)", ha='left', va='center',
            fontsize=16, fontweight='bold', color=C_GPU, zorder=10)

# BatchedControllers
draw_box(ax_new, (0.03, 0.66), 0.94, 0.11, "#F4ECF7", ec=C_CTRL, lw=2, alpha=0.85)
ax_new.text(0.5, 0.755, "BatchedControllers", ha='center', va='center',
            fontsize=22, fontweight='bold', color=C_CTRL, zorder=10)

ctrl_x_start = sim_x_start
for i in range(n_ctrl):
    cx = ctrl_x_start + i * (sim_w + sim_gap)
    draw_box(ax_new, (cx, 0.675), sim_w, 0.045, "#E8DAEF", ec=C_CTRL, lw=1.2, radius=0.003)
    ax_new.text(cx + sim_w / 2, 0.697, f"Ctrl {i}", ha='center', va='center',
                fontsize=11, fontweight='bold', color=C_CTRL, zorder=10)

ax_new.text(ctrl_x_start + n_ctrl * (sim_w + sim_gap) + 0.01, 0.697, "\u2026",
            ha='left', va='center', fontsize=20, fontweight='bold', color=C_CTRL, zorder=10)

draw_arrow(ax_new, (0.5, 0.66), (0.5, 0.62), color=C_ARROW)

# RolloutBackend (mujoco_warp)
draw_box(ax_new, (0.03, 0.03), 0.94, 0.59, "#E8F6F3", ec=C_ACCENT_NEW, lw=2.5, alpha=0.8)
ax_new.text(0.5, 0.605, "RolloutBackend (mujoco_warp)", ha='center', va='center',
            fontsize=22, fontweight='bold', color=C_ACCENT_NEW, zorder=10)

# CUDA Graph box with 2D grid
draw_box(ax_new, (0.05, 0.04), 0.90, 0.54, "#D6EAF8", ec=C_ACCENT_NEW, lw=2, alpha=0.7)
ax_new.text(0.5, 0.565, "mujoco_warp.step()  \u2014  CUDA Graph", ha='center', va='center',
            fontsize=20, fontweight='bold', color=C_ACCENT_NEW, zorder=10)

# 2D grid: 10 columns (controllers shown) x rows (representing 64 rollouts)
grid_cols = n_ctrl
grid_rows_shown = 18  # rows drawn (representing 64 with ellipsis)
cell_w = 0.07
cell_h = 0.022
cell_gap_x = 0.015
cell_gap_y = 0.003
total_grid_w = grid_cols * cell_w + (grid_cols - 1) * cell_gap_x
total_grid_h = grid_rows_shown * cell_h + (grid_rows_shown - 1) * cell_gap_y
grid_x0 = (1 - total_grid_w) / 2
grid_y_top = 0.52

# Pastel colors per column
col_colors = [
    "#AED6F1", "#A9DFBF", "#D7BDE2", "#F9E79F", "#A3E4D7",
    "#F5CBA7", "#AED6F1", "#D5F5E3", "#FADBD8", "#D6DBDF",
]

np.random.seed(42)
mid_row = grid_rows_shown // 2  # row where ellipsis goes

for c in range(grid_cols):
    base = col_colors[c]
    br, bg_, bb = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
    for r in range(grid_rows_shown):
        if r == mid_row:
            continue  # leave gap for ellipsis
        cx = grid_x0 + c * (cell_w + cell_gap_x)
        cy = grid_y_top - r * (cell_h + cell_gap_y)
        v = np.random.randint(-10, 10)
        fc = f"#{max(0,min(255,br+v)):02x}{max(0,min(255,bg_+v)):02x}{max(0,min(255,bb+v)):02x}"
        rect = plt.Rectangle((cx, cy), cell_w, cell_h, facecolor=fc,
                              edgecolor="#B0BEC5", linewidth=0.5, alpha=0.9, zorder=6)
        ax_new.add_patch(rect)

# Column headers
for c in range(grid_cols):
    cx = grid_x0 + c * (cell_w + cell_gap_x) + cell_w / 2
    ax_new.text(cx, grid_y_top + cell_h + 0.006, f"C{c}", ha='center', va='center',
                fontsize=11, fontweight='bold', color=C_SUBTEXT, zorder=10)

ax_new.text(grid_x0 + grid_cols * (cell_w + cell_gap_x) + 0.005,
            grid_y_top + cell_h + 0.006, "\u2026", ha='left', va='center',
            fontsize=16, fontweight='bold', color=C_SUBTEXT, zorder=10)

# Vertical ellipsis in each column
for c in range(grid_cols):
    cx = grid_x0 + c * (cell_w + cell_gap_x) + cell_w / 2
    cy = grid_y_top - mid_row * (cell_h + cell_gap_y) + cell_h / 2
    ax_new.text(cx, cy, "\u22ee", ha='center', va='center',
                fontsize=14, fontweight='bold', color="#5D6D7E", zorder=11)

# Row label on left
row_center_y = grid_y_top - total_grid_h / 2 + cell_h / 2
ax_new.text(grid_x0 - 0.035, row_center_y, "64\nrollouts\neach", ha='center', va='center',
            fontsize=13, color=C_SUBTEXT, zorder=10, linespacing=1.3, fontstyle='italic')

# Summary label at bottom
ax_new.text(0.5, 0.06, "100 controllers  \u00d7  64 rollouts  =  6,400 parallel GPU worlds",
            ha='center', va='center', fontsize=20, fontweight='bold', color=C_ACCENT_NEW, zorder=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_ACCENT_NEW, alpha=0.95, lw=2))

# Bottom banners
fig.text(0.135, 0.02, "32 CPU threads  \u2022  Sequential  \u2022  ONNX",
         ha='center', va='center', fontsize=16, color=C_ACCENT_OLD,
         fontweight='bold', fontstyle='italic')
fig.text(0.635, 0.02, "6,400 GPU worlds  \u2022  100 parallel trajectories  \u2022  PyTorch JIT + CUDA Graphs",
         ha='center', va='center', fontsize=16, color=C_ACCENT_NEW,
         fontweight='bold', fontstyle='italic')

plt.savefig("/home/dta/judo/docs/run_mpc_architecture.png", dpi=200, facecolor=C_BG,
            bbox_inches='tight', pad_inches=0.3)
plt.savefig("/home/dta/judo/docs/run_mpc_architecture.svg", facecolor=C_BG,
            bbox_inches='tight', pad_inches=0.3)
plt.savefig("/home/dta/judo/docs/run_mpc_architecture_transparent.png", dpi=200, facecolor='none',
            bbox_inches='tight', pad_inches=0.3, transparent=True)
print("Saved to docs/run_mpc_architecture.png, .svg, and _transparent.png")
