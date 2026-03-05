#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

# ============================
# CONFIG
# ============================

R_sphere = 0.0254
r_jar = 0.3032

t_window_ns = 15.0
t_window = t_window_ns * 1e-9

plot_stride = 300      # << much fewer plotted points
z_thickness = 0.0025   # ring vertical thickness

# ============================
# LOAD CSV
# ============================

data = np.loadtxt("traj.csv", delimiter=",", skiprows=1)

t = data[:,0]
x = data[:,1]
y = data[:,2]
z = data[:,3]

# Downsample trajectory
t = t[::plot_stride]
x = x[::plot_stride]
y = y[::plot_stride]
z = z[::plot_stride]

# Add slight vertical spread for ring visibility
z = z + np.random.normal(0, z_thickness, size=z.shape)

# ============================
# SPHERE
# ============================

u = np.linspace(0, 2*np.pi, 40)
v = np.linspace(0, np.pi, 40)

xs = R_sphere * np.outer(np.cos(u), np.sin(v))
ys = R_sphere * np.outer(np.sin(u), np.sin(v))
zs = R_sphere * np.outer(np.ones_like(u), np.cos(v))

# ============================
# PLOT FUNCTION
# ============================

def make_plot(mask, title):

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")

    ax.grid(False)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1,1,1,0))

    ax.xaxis.line.set_linewidth(1.5)
    ax.yaxis.line.set_linewidth(1.5)
    ax.zaxis.line.set_linewidth(1.5)

    # limits (zoomed for rings)
    lim = 0.13
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.set_box_aspect([1,1,1])

    # ticks
    ticks = np.linspace(-0.1,0.1,5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    # push tick labels away from axes
    ax.tick_params(axis='x', pad=18, labelsize=14)
    ax.tick_params(axis='y', pad=18, labelsize=14)
    ax.tick_params(axis='z', pad=18, labelsize=14)

    # axis labels
    ax.set_xlabel("X [m]", fontsize=20, labelpad=50)
    ax.set_ylabel("Y [m]", fontsize=20, labelpad=50)
    ax.set_zlabel("Z [m]", fontsize=20, labelpad=50)

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)

    # trajectory points
    ax.scatter(
        x[mask],
        y[mask],
        z[mask],
        s=2,
        c="purple",
        alpha=0.8
    )

    # sphere
    ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.4)

    ax.set_title(title, fontsize=18)

    plt.tight_layout()
    plt.show()

# ============================
# MASKS
# ============================

mask_short  = t <= t_window
mask_double = t <= 2*t_window
mask_full   = np.ones_like(t, dtype=bool)

# ============================
# PLOTS
# ============================

make_plot(mask_short,  f"First {t_window_ns} ns")
make_plot(mask_double, f"First {2*t_window_ns} ns")
make_plot(mask_full,   "Full Trajectory")