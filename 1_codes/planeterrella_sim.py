#!/usr/bin/env python3
"""
Planeterrella simulation (SI units)
Dipole + optional sphere E field
Boris pusher integration
Python + Numba parallel optimize path
Equatorial (green-like) launches enforced
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np

# For 3D sphere rotations:
import matplotlib
matplotlib.use("QtAgg")   # best option
# matplotlib.use("TkAgg") # fallback if Qt not installed

# ==========================================================
# OPTIONAL NUMBA
# ==========================================================

try:
    import numba as nb
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

# ==========================================================
# CONSTANTS
# ==========================================================

QE = -1.602176634e-19
ME = 9.1093837015e-31
MU0 = 4e-7 * math.pi
EV = 1.602176634e-19

# ==========================================================
# CONFIG
# ==========================================================

@dataclass(frozen=True)
class SimConfig:
    V0: float
    R_sphere: float
    r_jar: float
    dt: float
    tmax: float
    M_vec: Tuple[float, float, float]
    seed: int
    store_stride: int

# ==========================================================
# FIELDS
# ==========================================================

def E_field(r, V0, R):
    x,y,z = r
    rr = math.sqrt(x*x+y*y+z*z) + 1e-30
    coef = (V0*R)/(rr*rr)
    return np.array([coef*x/rr, coef*y/rr, coef*z/rr])

def B_dipole(r, M_vec):
    x,y,z = r
    rr2 = x*x+y*y+z*z + 1e-30
    rr = math.sqrt(rr2)
    rhat = np.array([x/rr,y/rr,z/rr])
    M = np.array(M_vec)
    mdotr = M @ rhat
    coef = (MU0/(4*math.pi))/(rr**3)
    return coef*(3*mdotr*rhat - M)

# ==========================================================
# BORIS (PYTHON)
# ==========================================================

def boris_step(r, v, dt, cfg):
    E = E_field(r, cfg.V0, cfg.R_sphere)
    B = B_dipole(r, cfg.M_vec)
    qmdt2 = (QE/ME)*dt*0.5

    v_minus = v + qmdt2*E
    t = qmdt2*B
    t2 = np.dot(t,t)
    s = 2*t/(1+t2)

    v_prime = v_minus + np.cross(v_minus,t)
    v_plus  = v_minus + np.cross(v_prime,s)

    v_new = v_plus + qmdt2*E
    r_new = r + v_new*dt
    return r_new, v_new

# ==========================================================
# PYTHON SIMULATION
# ==========================================================

def simulate_python(cfg,nparticles):

    rng = np.random.default_rng(cfg.seed)
    nsteps = int(cfg.tmax/cfg.dt)
    v0 = math.sqrt(2*12.0*EV/ME)

    trajectories=[]

    for i in range(nparticles):

        theta = 2*math.pi*rng.random()

        # Equatorial launches (green-like)
        while True:
            z0 = 2*rng.random()-1
            if abs(z0) < 0.4:
                break

        rxy = math.sqrt(1-z0*z0)

        r=np.array([
            cfg.R_sphere*1.05*rxy*math.cos(theta),
            cfg.R_sphere*1.05*rxy*math.sin(theta),
            cfg.R_sphere*1.05*z0
        ])

        while True:
            dirv=rng.normal(size=3)
            dirv/=np.linalg.norm(dirv)
            if np.dot(dirv,r)>0:
                break

        v=v0*dirv
        pts=[]

        for step in range(nsteps):

            if step%cfg.store_stride==0:
                pts.append(r.copy())

            if np.linalg.norm(r)<cfg.R_sphere: break
            if math.hypot(r[0],r[1])>cfg.r_jar: break

            r,v=boris_step(r,v,cfg.dt,cfg)

        trajectories.append(np.array(pts))

    return trajectories

# ==========================================================
# NUMBA OPTIMIZED PATH
# ==========================================================

if NUMBA_OK:

    @nb.njit(cache=True)
    def _boris_nb(x,y,z,vx,vy,vz,dt,V0,R,Mx,My,Mz):

        rr = math.sqrt(x*x+y*y+z*z) + 1e-30
        coefE = (V0*R)/(rr*rr)
        ex,ey,ez = coefE*x/rr, coefE*y/rr, coefE*z/rr

        rr2 = x*x+y*y+z*z + 1e-30
        rr = math.sqrt(rr2)
        rx,ry,rz = x/rr,y/rr,z/rr
        mdotr = Mx*rx+My*ry+Mz*rz
        coefB = (MU0/(4*math.pi))/(rr**3)
        bx = coefB*(3*mdotr*rx - Mx)
        by = coefB*(3*mdotr*ry - My)
        bz = coefB*(3*mdotr*rz - Mz)

        qmdt2 = (QE/ME)*dt*0.5

        vxm = vx + qmdt2*ex
        vym = vy + qmdt2*ey
        vzm = vz + qmdt2*ez

        tx,ty,tz = qmdt2*bx,qmdt2*by,qmdt2*bz
        t2 = tx*tx+ty*ty+tz*tz
        sx,sy,sz = 2*tx/(1+t2),2*ty/(1+t2),2*tz/(1+t2)

        vpx = vxm + (vym*tz - vzm*ty)
        vpy = vym + (vzm*tx - vxm*tz)
        vpz = vzm + (vxm*ty - vym*tx)

        vxp = vxm + (vpy*sz - vpz*sy)
        vyp = vym + (vpz*sx - vpx*sz)
        vzp = vzm + (vpx*sy - vpy*sx)

        vx_new = vxp + qmdt2*ex
        vy_new = vyp + qmdt2*ey
        vz_new = vzp + qmdt2*ez

        return x+vx_new*dt,y+vy_new*dt,z+vz_new*dt,vx_new,vy_new,vz_new

    @nb.njit(parallel=True)
    def simulate_many_nb(
        nparticles,nsteps,store_stride,dt,V0,R_sphere,r_jar,
        Mx,My,Mz
    ):

        nstore = (nsteps+store_stride-1)//store_stride
        out = np.empty((nparticles,nstore,3))
        v0 = math.sqrt(2*12.0*EV/ME)

        for i in nb.prange(nparticles):

            theta = 2*math.pi*np.random.random()

            while True:
                z0 = 2*np.random.random()-1
                if abs(z0) < 0.4:
                    break

            rxy = math.sqrt(1-z0*z0)

            x = R_sphere*1.05*rxy*math.cos(theta)
            y = R_sphere*1.05*rxy*math.sin(theta)
            z = R_sphere*1.05*z0

            while True:
                vx = np.random.normal()
                vy = np.random.normal()
                vz = np.random.normal()
                norm = math.sqrt(vx*vx+vy*vy+vz*vz)
                vx/=norm; vy/=norm; vz/=norm
                if vx*x + vy*y + vz*z > 0:
                    break

            vx*=v0; vy*=v0; vz*=v0

            j=0
            for step in range(nsteps):

                if step%store_stride==0:
                    out[i,j,0]=x
                    out[i,j,1]=y
                    out[i,j,2]=z
                    j+=1

                if math.sqrt(x*x+y*y+z*z)<R_sphere: break
                if math.sqrt(x*x+y*y)>r_jar: break

                x,y,z,vx,vy,vz = _boris_nb(
                    x,y,z,vx,vy,vz,dt,V0,R_sphere,Mx,My,Mz
                )

        return out

# ==========================================================
# PLOTTING
# ==========================================================

def plot_system_interactive(cfg, trajs):
    import numpy as np
    import matplotlib.pyplot as plt

    plt.ion()  # interactive mode ON

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # ----------------------------
    # Trajectories
    # ----------------------------
    for tr in trajs:
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2],
                color="green", linewidth=1.5)

    # ----------------------------
    # Sphere
    # ----------------------------
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, np.pi, 80)

    xs = cfg.R_sphere * np.outer(np.cos(u), np.sin(v))
    ys = cfg.R_sphere * np.outer(np.sin(u), np.sin(v))
    zs = cfg.R_sphere * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(xs, ys, zs,
                    color="gray",
                    alpha=0.9,
                    edgecolor="none")

    lim = cfg.r_jar * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])

    # Clean styling
    ax.grid(False)
    ax.tick_params(labelsize=16)

    ax.set_xlabel("X [cm]", fontsize=22, labelpad=20)
    ax.set_ylabel("Y [cm]", fontsize=22, labelpad=20)
    ax.set_zlabel("Z [cm]", fontsize=22, labelpad=20)

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)

    plt.show()

# ==========================================================
# MAIN
# ==========================================================

def main():

    p=argparse.ArgumentParser()
    p.add_argument("--nparticles",type=int,default=10)
    p.add_argument("--V0",type=float,default=0)
    p.add_argument("--M",type=float,default=1)
    p.add_argument("--dt",type=float,default=5e-12)
    p.add_argument("--tmax",type=float,default=2e-6)
    p.add_argument("--store_stride",type=int,default=10)
    p.add_argument("--optimize",action="store_true")
    p.add_argument(
        "--plottraj",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable trajectory plotting (default: on). Use --no-plottraj to disable."
    )
    args=p.parse_args()

    cfg=SimConfig(
        V0=args.V0,
        R_sphere=0.0254,
        r_jar=0.3032,
        dt=args.dt,
        tmax=args.tmax,
        M_vec=(0,0,args.M),
        seed=7,
        store_stride=args.store_stride
    )

    if args.optimize and NUMBA_OK:
        nsteps=int(cfg.tmax/cfg.dt)
        pos=simulate_many_nb(
            args.nparticles,nsteps,cfg.store_stride,cfg.dt,
            cfg.V0,cfg.R_sphere,cfg.r_jar,
            cfg.M_vec[0],cfg.M_vec[1],cfg.M_vec[2]
        )
        trajs=[pos[i,:,:] for i in range(args.nparticles)]
    else:
        trajs=simulate_python(cfg,args.nparticles)

    if args.plottraj:
        plot_system_interactive(cfg,trajs)

if __name__=="__main__":
    main()
