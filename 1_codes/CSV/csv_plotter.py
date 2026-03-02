#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

# ============================
# CONFIG
# ============================

R_sphere = 0.0254
r_jar = 0.3032

t_window_ns = 15.0          # first few nanoseconds (change as desired)
t_window = t_window_ns * 1e-9


# ============================
# LOAD CSV
# ============================

data = np.loadtxt("traj.csv", delimiter=",", skiprows=1)

t = data[:, 0]
x = data[:, 1]
y = data[:, 2]
z = data[:, 3]


# ============================
# SPHERE (precompute once)
# ============================

u = np.linspace(0, 2*np.pi, 80)
v = np.linspace(0, np.pi, 80)

xs = R_sphere * np.outer(np.cos(u), np.sin(v))
ys = R_sphere * np.outer(np.sin(u), np.sin(v))
zs = R_sphere * np.outer(np.ones_like(u), np.cos(v))


# ============================
# PLOTTING FUNCTION
# ============================

def make_plot(mask, title):

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Clean background
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))

    # Axis lines
    ax.xaxis.line.set_linewidth(1.5)
    ax.yaxis.line.set_linewidth(1.5)
    ax.zaxis.line.set_linewidth(1.5)

    # Limits
    lim = r_jar * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])

    # Ticks
    tick_vals = np.linspace(-0.03, 0.03, 5)
    ax.set_xticks(tick_vals)
    ax.set_yticks(tick_vals)
    ax.set_zticks(tick_vals)

    ax.tick_params(axis='x', labelsize=16, pad=10)
    ax.tick_params(axis='y', labelsize=16, pad=10)
    ax.tick_params(axis='z', labelsize=16, pad=10)

    # Labels
    ax.set_xlabel("X (m)", fontsize=22, labelpad=25)
    ax.set_ylabel("Y (m)", fontsize=22, labelpad=25)
    ax.set_zlabel("Z (m)", fontsize=22, labelpad=30)

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)

    ax.xaxis.label.set_rotation(0)
    ax.yaxis.label.set_rotation(0)
    ax.zaxis.label.set_rotation(0)

    # Trajectory slice
    ax.plot(x[mask], y[mask], z[mask], color="purple")

    # Sphere
    ax.plot_surface(xs, ys, zs, color="gray", alpha=0.9, edgecolor="none")

    ax.set_title(title, fontsize=20)

    plt.show()


# ============================
# MASKS
# ============================

mask_short  = t <= t_window
mask_double = t <= 2.0 * t_window
mask_full   = np.ones_like(t, dtype=bool)


# ============================
# GENERATE PLOTS
# ============================

make_plot(mask_short,  f"First {t_window_ns} ns")
make_plot(mask_double, f"First {2*t_window_ns} ns")
make_plot(mask_full,   "Full Trajectory")