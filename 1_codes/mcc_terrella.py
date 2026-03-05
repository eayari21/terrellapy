#!/usr/bin/env python3
"""
High-performance Planeterrella
Boris + Monte Carlo Collisions (MCC)
- Species-resolved (N2 vs O2) collisions
- Inelastic energy loss + isotropic scattering for excitation
- Safe per-particle collision logging (no race conditions in prange)
- Progress + ETA
- Publication-style 3D plot (rings)
"""

import math
import time
import argparse
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

# ==========================================================
# CONSTANTS
# ==========================================================

QE = -1.602176634e-19
ME = 9.1093837015e-31
MU0 = 4e-7 * math.pi
EV = 1.602176634e-19
KB = 1.380649e-23

# Cross sections (m^2) — simplified constants (per your paper’s simplified MCC description)
SIG_ELASTIC = 35e-20
SIG_EXC     = 2e-20
SIG_ION     = 3e-20

# Excitation / ionization thresholds (eV)
O2_THRESH = 12.07   # red outer ring
N2_THRESH = 15.58   # purple inner ring

# “Collision-type selection” (used for sigma(E))
E_ELASTIC_MAX = 9.0
E_EXC_MAX     = 25.0

# ==========================================================
# HELPER: sigma(E) (Numba-safe)
# ==========================================================

@nb.njit(inline="always", fastmath=True)
def sigma_total_from_ke(ke_ev: float) -> float:
    # simplified piecewise model
    if ke_ev < E_ELASTIC_MAX:
        return SIG_ELASTIC
    elif ke_ev < E_EXC_MAX:
        return SIG_EXC
    else:
        return SIG_ION

@nb.njit(inline="always", fastmath=True)
def isotropic_velocity(vmag: float) -> tuple[float, float, float]:
    # sample isotropic direction on sphere
    theta = 2.0 * math.pi * np.random.random()
    u = 2.0 * np.random.random() - 1.0
    s = math.sqrt(max(0.0, 1.0 - u*u))
    vx = vmag * s * math.cos(theta)
    vy = vmag * s * math.sin(theta)
    vz = vmag * u
    return vx, vy, vz

# ==========================================================
# BORIS + MCC STEP
# ==========================================================

@nb.njit(inline="always", fastmath=True)
def boris_mcc_step(
    x, y, z, vx, vy, vz,
    dt, V0, R_sphere, Mx, My, Mz,
    n_total, o2_frac,
    z_record_max
):
    """
    Returns:
      (x,y,z,vx,vy,vz, ctype)
    where ctype:
      0 = no recorded emission
      1 = O2 emission (red)
      2 = N2 emission (purple)
    """

    # ------------------------------------------------------
    # Fields
    # ------------------------------------------------------
    rr = math.sqrt(x*x + y*y + z*z) + 1e-30

    # Electric field magnitude ~ V0*R / r^2
    # IMPORTANT: choose sign so electrons are repelled outward from the sphere.
    # With QE < 0, we want qE outward => E inward => minus sign on E.
    coefE = -(V0 * R_sphere) / (rr * rr)
    ex, ey, ez = coefE * x/rr, coefE * y/rr, coefE * z/rr

    # Dipole B field
    rx, ry, rz = x/rr, y/rr, z/rr
    mdotr = Mx*rx + My*ry + Mz*rz
    coefB = (MU0 / (4.0*math.pi)) / (rr**3)
    bx = coefB * (3.0*mdotr*rx - Mx)
    by = coefB * (3.0*mdotr*ry - My)
    bz = coefB * (3.0*mdotr*rz - Mz)

    # ------------------------------------------------------
    # Boris push
    # ------------------------------------------------------
    qmdt2 = (QE/ME) * dt * 0.5

    vxm = vx + qmdt2*ex
    vym = vy + qmdt2*ey
    vzm = vz + qmdt2*ez

    tx, ty, tz = qmdt2*bx, qmdt2*by, qmdt2*bz
    t2 = tx*tx + ty*ty + tz*tz
    sx, sy, sz = 2.0*tx/(1.0+t2), 2.0*ty/(1.0+t2), 2.0*tz/(1.0+t2)

    vpx = vxm + (vym*tz - vzm*ty)
    vpy = vym + (vzm*tx - vxm*tz)
    vpz = vzm + (vxm*ty - vym*tx)

    vxp = vxm + (vpy*sz - vpz*sy)
    vyp = vym + (vpz*sx - vpx*sz)
    vzp = vzm + (vpx*sy - vpy*sx)

    vx = vxp + qmdt2*ex
    vy = vyp + qmdt2*ey
    vz = vzp + qmdt2*ez

    x += vx*dt
    y += vy*dt
    z += vz*dt

    # ------------------------------------------------------
    # MCC collision
    # ------------------------------------------------------
    speed = math.sqrt(vx*vx + vy*vy + vz*vz) + 1e-30
    ke_ev = 0.5 * ME * speed*speed / EV

    sigma = sigma_total_from_ke(ke_ev)

    # P = 1 - exp(-(Σ n_s σ_s) |v| dt) ; here σ(E) same form for both species.
    lam = n_total * sigma * speed * dt
    P = 1.0 - math.exp(-lam)

    ctype = 0

    if np.random.random() < P:
        # pick neutral species by density fraction (air-like)
        # (weighted by n_s; sigma same here, so density fraction is enough)
        is_o2 = (np.random.random() < o2_frac)

        if is_o2:
            # O2 excitation if above threshold
            if ke_ev > O2_THRESH:
                ke_rem = ke_ev - O2_THRESH

                # energy-sharing approximation (captures “cooling” into O2-only band)
                f = np.random.random()
                ke_new = f * ke_rem
                vmag = math.sqrt(max(0.0, 2.0*ke_new*EV/ME))
                vx, vy, vz = isotropic_velocity(vmag)

                # record only near equator to reveal rings
                if abs(z) <= z_record_max:
                    ctype = 1
            else:
                # elastic-like reflection
                vx, vy, vz = -vx, -vy, -vz

        else:
            # N2 excitation if above threshold
            if ke_ev > N2_THRESH:
                ke_rem = ke_ev - N2_THRESH
                f = np.random.random()
                ke_new = f * ke_rem
                vmag = math.sqrt(max(0.0, 2.0*ke_new*EV/ME))
                vx, vy, vz = isotropic_velocity(vmag)

                if abs(z) <= z_record_max:
                    ctype = 2
            else:
                vx, vy, vz = -vx, -vy, -vz

    return x, y, z, vx, vy, vz, ctype

# ==========================================================
# SIMULATION CHUNK (PARALLEL)
# ==========================================================

@nb.njit(parallel=True, fastmath=True, cache=True)
def simulate_chunk(
    x, y, z, vx, vy, vz,
    active,
    coll_x, coll_y, coll_z, coll_type, coll_count,
    nsteps,
    dt, V0, R_sphere, r_jar,
    Mx, My, Mz,
    n_total, o2_frac,
    z_record_max
):
    """
    Safe collision recording:
      each particle has its own collision ring-buffer of length K
      => no shared counters in prange
    """
    n_particles = x.shape[0]
    K = coll_x.shape[1]

    for _ in range(nsteps):
        for i in nb.prange(n_particles):

            if active[i] == 0:
                continue

            xi = x[i]; yi = y[i]; zi = z[i]
            vxi = vx[i]; vyi = vy[i]; vzi = vz[i]

            xi, yi, zi, vxi, vyi, vzi, ctype = boris_mcc_step(
                xi, yi, zi, vxi, vyi, vzi,
                dt, V0, R_sphere, Mx, My, Mz,
                n_total, o2_frac,
                z_record_max
            )

            # boundaries: sphere hit / jar wall
            rr = math.sqrt(xi*xi + yi*yi + zi*zi)
            rho = math.sqrt(xi*xi + yi*yi)

            if rr <= R_sphere:
                active[i] = 0
            elif rho >= r_jar:
                active[i] = 0

            x[i] = xi; y[i] = yi; z[i] = zi
            vx[i] = vxi; vy[i] = vyi; vz[i] = vzi

            # record emission collisions
            if ctype != 0:
                c = coll_count[i]
                if c < K:
                    coll_x[i, c] = xi
                    coll_y[i, c] = yi
                    coll_z[i, c] = zi
                    coll_type[i, c] = ctype
                    coll_count[i] = c + 1

# ==========================================================
# MAIN
# ==========================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pressure", type=float, default=400.0, help="mTorr")
    p.add_argument("--temperature", type=float, default=300.0, help="K")
    p.add_argument("--o2_frac", type=float, default=0.20, help="O2 fraction (air ~0.2). Set 0 for pure N2.")
    p.add_argument("--nparticles", type=int, default=50000)
    p.add_argument("--V0", type=float, default=1200.0, help="Volts (magnitude)")
    p.add_argument("--M", type=float, default=250.0, help="Dipole moment scalar used in your existing scaling")
    p.add_argument("--dt", type=float, default=5e-12)
    p.add_argument("--tmax", type=float, default=3e-6)
    p.add_argument("--chunk_steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max_coll_per_particle", type=int, default=30)
    p.add_argument("--z_record_max", type=float, default=0.02, help="Only record emissions with |z| <= this [m]")
    args = p.parse_args()

    np.random.seed(args.seed)

    # geometry
    R_sphere = 0.0254
    r_jar = 0.3032

    # neutral density from pressure
    pressure_Pa = args.pressure * 0.133322
    n_total = pressure_Pa / (KB * args.temperature)

    dt = args.dt
    total_steps = int(args.tmax / dt)
    nparticles = args.nparticles

    # ------------------------------------------------------
    # Injection (more faithful to your paper):
    # Random around equator just outside the sphere, ~zero initial speed.
    # Let E-field accelerate, B traps, collisions cool into two emission shells.
    # ------------------------------------------------------
    theta = np.random.uniform(0.0, 2.0*np.pi, nparticles)

    r0 = 1.05 * R_sphere
    x = r0 * np.cos(theta)
    y = r0 * np.sin(theta)
    z = np.zeros(nparticles)

    # small “thermal” seed velocities (so we get bounce/mirror spread)
    # keep them small so E-field + ExB dominates early
    vth = 2.0e4  # m/s
    vx = vth * (np.random.random(nparticles)*2.0 - 1.0)
    vy = vth * (np.random.random(nparticles)*2.0 - 1.0)
    vz = vth * (np.random.random(nparticles)*2.0 - 1.0)

    active = np.ones(nparticles, dtype=np.int32)

    # collision ring-buffer per particle (safe under prange)
    K = args.max_coll_per_particle
    coll_x = np.zeros((nparticles, K), dtype=np.float64)
    coll_y = np.zeros((nparticles, K), dtype=np.float64)
    coll_z = np.zeros((nparticles, K), dtype=np.float64)
    coll_type = np.zeros((nparticles, K), dtype=np.int32)
    coll_count = np.zeros(nparticles, dtype=np.int32)

    # ------------------------------------------------------
    # Run with progress
    # ------------------------------------------------------
    completed = 0
    start = time.time()
    print("\nStarting simulation...")
    print(f"  particles:   {nparticles:,}")
    print(f"  steps:       {total_steps:,}  (dt={dt:g}s, tmax={args.tmax:g}s)")
    print(f"  pressure:    {args.pressure:g} mTorr  -> n={n_total:.3e} m^-3")
    print(f"  mixture:     O2 frac={args.o2_frac:.2f}, N2 frac={1.0-args.o2_frac:.2f}")
    print("--------------------------------------------------")

    # warm-up call to compile (small chunk)
    warm = min(args.chunk_steps, max(10, total_steps))
    simulate_chunk(
        x, y, z, vx, vy, vz,
        active,
        coll_x, coll_y, coll_z, coll_type, coll_count,
        warm,
        dt, args.V0, R_sphere, r_jar,
        0.0, 0.0, args.M,
        n_total, args.o2_frac,
        args.z_record_max
    )
    completed += warm

    # main loop
    while completed < total_steps:
        steps_now = min(args.chunk_steps, total_steps - completed)

        simulate_chunk(
            x, y, z, vx, vy, vz,
            active,
            coll_x, coll_y, coll_z, coll_type, coll_count,
            steps_now,
            dt, args.V0, R_sphere, r_jar,
            0.0, 0.0, args.M,
            n_total, args.o2_frac,
            args.z_record_max
        )

        completed += steps_now

        elapsed = time.time() - start
        rate = completed / max(elapsed, 1e-12)
        eta = (total_steps - completed) / max(rate, 1e-12)

        print(f"Progress {100*completed/total_steps:6.2f}% | "
              f"Elapsed {elapsed:7.1f}s | ETA {eta:7.1f}s | "
              f"Active {int(active.sum()):,}")

    print("--------------------------------------------------")
    print("Simulation complete.\n")

    # ------------------------------------------------------
    # Flatten collisions for plotting
    # ------------------------------------------------------
    # Build mask of stored entries
    counts = coll_count.copy()
    total_rec = int(counts.sum())
    if total_rec == 0:
        print("No emission collisions recorded (try larger tmax, higher pressure, or larger z_record_max).")
        return

    flat_x = np.empty(total_rec, dtype=np.float64)
    flat_y = np.empty(total_rec, dtype=np.float64)
    flat_z = np.empty(total_rec, dtype=np.float64)
    flat_t = np.empty(total_rec, dtype=np.int32)

    j = 0
    for i in range(nparticles):
        c = counts[i]
        if c > 0:
            flat_x[j:j+c] = coll_x[i, :c]
            flat_y[j:j+c] = coll_y[i, :c]
            flat_z[j:j+c] = coll_z[i, :c]
            flat_t[j:j+c] = coll_type[i, :c]
            j += c

    # diagnostics: radial means (helps verify ring separation)
    rho = np.sqrt(flat_x*flat_x + flat_y*flat_y)
    red_rho = rho[flat_t == 1]
    pur_rho = rho[flat_t == 2]
    if red_rho.size:
        print(f"O2 (red)   rho mean = {red_rho.mean():.4f} m, std = {red_rho.std():.4f} m, N={red_rho.size}")
    if pur_rho.size:
        print(f"N2 (purple)rho mean = {pur_rho.mean():.4f} m, std = {pur_rho.std():.4f} m, N={pur_rho.size}")

    # ------------------------------------------------------
    # PLOT (same styling you liked)
    # ------------------------------------------------------
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    mask_red = (flat_t == 1)
    mask_pur = (flat_t == 2)

    ax.scatter(flat_x[mask_red], flat_y[mask_red], flat_z[mask_red], s=4, c="red")
    ax.scatter(flat_x[mask_pur], flat_y[mask_pur], flat_z[mask_pur], s=4, c="purple")

    # sphere wireframe
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    xs = R_sphere * np.outer(np.cos(u), np.sin(v))
    ys = R_sphere * np.outer(np.sin(u), np.sin(v))
    zs = R_sphere * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.4)

    lim = 0.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])

    # large horizontal labels + large ticks
    ax.set_xlabel("X [m]", fontsize=22, labelpad=20)
    ax.set_ylabel("Y [m]", fontsize=22, labelpad=20)
    ax.set_zlabel("Z [m]", fontsize=22, labelpad=20)

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel("X [m]", rotation=0)
    ax.set_ylabel("Y [m]", rotation=0)
    ax.set_zlabel("Z [m]", rotation=0)

    ax.tick_params(axis="both", labelsize=16)
    ax.tick_params(axis="z", labelsize=16)

    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()