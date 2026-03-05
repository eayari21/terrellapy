#!/usr/bin/env python3
"""
Two symmetric equatorial emission rings.
Clean publication-style 3D rendering.
More stochastic spatial structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ==========================================================
# PARAMETERS
# ==========================================================

np.random.seed(7)

R_sphere = 0.0254
N_points = 8_000

r_red_mean = 0.085
r_pur_mean = 0.040

sigma_red = 0.007
sigma_pur = 0.003

sigma_z_red = 0.002
sigma_z_pur = 0.006


# ==========================================================
# GENERATE RINGS
# ==========================================================

phi_red = np.random.uniform(0, 2*np.pi, N_points)
phi_pur = np.random.uniform(0, 2*np.pi, N_points)

phi_red += np.random.normal(0, 0.05, N_points)
phi_pur += np.random.normal(0, 0.05, N_points)

r_red = (
    r_red_mean
    + np.random.normal(0, sigma_red, N_points)
    + np.random.lognormal(mean=-6, sigma=0.8, size=N_points)
)

r_pur = (
    r_pur_mean
    + np.random.normal(0, sigma_pur, N_points)
    + np.random.lognormal(mean=-6.5, sigma=0.7, size=N_points)
)

x_red = r_red * np.cos(phi_red)
y_red = r_red * np.sin(phi_red)

x_pur = r_pur * np.cos(phi_pur)
y_pur = r_pur * np.sin(phi_pur)

z_red = np.random.normal(0, sigma_z_red * (1 + 0.5*np.random.rand(N_points)), N_points)
z_pur = np.random.normal(0, sigma_z_pur * (1 + 0.5*np.random.rand(N_points)), N_points)

x_red += np.random.normal(0, 0.0015, N_points)
y_red += np.random.normal(0, 0.0015, N_points)

x_pur += np.random.normal(0, 0.0010, N_points)
y_pur += np.random.normal(0, 0.0010, N_points)

mask_red = np.random.rand(N_points) > 0.35
mask_pur = np.random.rand(N_points) > 0.25

x_red, y_red, z_red = x_red[mask_red], y_red[mask_red], z_red[mask_red]
x_pur, y_pur, z_pur = x_pur[mask_pur], y_pur[mask_pur], z_pur[mask_pur]


# ==========================================================
# 3D RING PLOT
# ==========================================================

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection="3d")

ax.grid(False)

for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = False
    axis.pane.set_edgecolor((1,1,1,0))

ax.scatter(x_red, y_red, z_red, s=4, c="red")
ax.scatter(x_pur, y_pur, z_pur, s=4, c="purple")


# sphere

u = np.linspace(0,2*np.pi,40)
v = np.linspace(0,np.pi,40)

xs = R_sphere * np.outer(np.cos(u), np.sin(v))
ys = R_sphere * np.outer(np.sin(u), np.sin(v))
zs = R_sphere * np.outer(np.ones_like(u), np.cos(v))

ax.plot_wireframe(xs,ys,zs,color="gray",alpha=0.4)


lim = 0.12

ax.set_xlim(-lim,lim)
ax.set_ylim(-lim,lim)
ax.set_zlim(-lim,lim)

ax.set_box_aspect([1,1,1])

ax.set_xlabel("X [m]", fontsize=22, labelpad=35)
ax.set_ylabel("Y [m]", fontsize=22, labelpad=35)
ax.set_zlabel("Z [m]", fontsize=22, labelpad=35)

ax.tick_params(axis='x', labelsize=16, pad=12)
ax.tick_params(axis='y', labelsize=16, pad=12)
ax.tick_params(axis='z', labelsize=16, pad=12)

plt.tight_layout()
plt.show()


# ==========================================================
# CONTINUOUS BELL-JAR PLASMA FIELD
# ==========================================================

R_sphere_cm = 2.6
R_jar_cm = 30.0

Nr = 160
Ntheta = 320

r = np.linspace(R_sphere_cm, R_jar_cm, Nr)
theta = np.linspace(0, 2*np.pi, Ntheta)

R, T = np.meshgrid(r, theta)

X = R*np.cos(T)
Y = R*np.sin(T)


# ==========================================================
# PHYSICS MODELS
# ==========================================================

v0 = 1.4e6
velocity = v0*(R_sphere_cm/R)**0.35
velocity += 2e5*np.exp(-(R-8)**2/30)

velocity = np.clip(velocity,3e5,1.4e6)

# dipole field

B0 = 0.05
B = B0*(R_sphere_cm/R)**3

e = 1.602e-19
m = 9.11e-31

omega_g = e*B/m

n_gas = 5e20
sigma = 3e-19

nu_collision = n_gas*sigma*velocity

ratio = omega_g/nu_collision

collision_prob = nu_collision/np.max(nu_collision)


# ==========================================================
# AURORAL RING ENHANCEMENTS
# ==========================================================

ring1 = np.exp(-(R-5.0)**2/1.2)
ring2 = np.exp(-(R-10.5)**2/1.8)

# enhance collisions in ring regions
nu_collision *= (1 + 0.9*ring1 + 0.6*ring2)

# recompute ratio after modification
ratio = omega_g / nu_collision

# velocity enhancement (visual emission shell)
velocity *= (1 + 0.35*ring1 + 0.20*ring2)

collision_prob *= (1 + 0.7*ring1 + 0.4*ring2)
collision_prob = np.clip(collision_prob,0,1)


plt.rcParams.update({
    "font.size":18,
    "axes.labelsize":22,
    "xtick.labelsize":16,
    "ytick.labelsize":16
})


# ==========================================================
# VELOCITY POLAR MAP
# ==========================================================

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111, projection='polar')

pc = ax.pcolormesh(T,R,velocity/1e4,cmap="nipy_spectral",shading="auto")

theta_line = np.linspace(0,2*np.pi,500)
ax.plot(theta_line,np.full_like(theta_line,5.0),color="white",lw=1.5)
ax.plot(theta_line,np.full_like(theta_line,10.5),color="white",lw=1.5)

ax.set_rmin(R_sphere_cm)
ax.set_rmax(R_jar_cm)
ax.set_rticks([5,10,15,20,25,30])
ax.set_rlabel_position(22)

cbar = plt.colorbar(pc,pad=0.12)
cbar.set_label("Average velocity (10⁴ m/s)")

plt.tight_layout()
plt.show()


# ==========================================================
# GYROFREQUENCY / COLLISION FREQUENCY
# ==========================================================

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111, projection='polar')

pc = ax.pcolormesh(T,R,ratio,cmap="turbo",shading="auto")

ax.plot(theta_line,np.full_like(theta_line,5.0),color="white",lw=1.5)
ax.plot(theta_line,np.full_like(theta_line,10.5),color="white",lw=1.5)

ax.set_rmin(R_sphere_cm)
ax.set_rmax(R_jar_cm)
ax.set_rticks([5,10,15,20,25,30])
ax.set_rlabel_position(22)

cbar = plt.colorbar(pc,pad=0.12)
cbar.set_label("Gyrofrequency / Collision Frequency")

plt.tight_layout()
plt.show()


# ==========================================================
# COLLISION PROBABILITY MAP
# ==========================================================

fig,ax = plt.subplots(figsize=(8,8))

sc = ax.scatter(
    X.flatten(),
    Y.flatten(),
    c=collision_prob.flatten(),
    s=8,
    cmap="RdPu"
)

cbar = plt.colorbar(sc)
cbar.set_label("Collision probability")

ax.set_xlabel("X [cm]")
ax.set_ylabel("Y [cm]")

ax.set_xlim(-R_jar_cm,R_jar_cm)
ax.set_ylim(-R_jar_cm,R_jar_cm)

ax.set_aspect("equal")

plt.tight_layout()
plt.show()


# ==========================================================
# COLLISION DENSITY
# ==========================================================

points = np.vstack([X.flatten(),Y.flatten()])

kde = gaussian_kde(points)

grid = 250

xi = np.linspace(-R_jar_cm,R_jar_cm,grid)
yi = np.linspace(-R_jar_cm,R_jar_cm,grid)

XI,YI = np.meshgrid(xi,yi)

ZI = kde(np.vstack([XI.flatten(),YI.flatten()])).reshape(grid,grid)

fig,ax = plt.subplots(figsize=(8,8))

im = ax.contourf(XI,YI,ZI,levels=60,cmap="RdPu")

cbar = plt.colorbar(im)
cbar.set_label(r"Collision density [cm$^{-2}$]")

ax.set_xlabel("X [cm]")
ax.set_ylabel("Y [cm]")

ax.set_aspect("equal")

plt.tight_layout()
plt.show()