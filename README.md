<div align="center">

# **TerrellaPy**

<p>
  <img alt="theme" src="https://img.shields.io/badge/theme-midnight%20%2B%20gold-0B0F1A?style=for-the-badge&labelColor=0B0F1A&color=C7A54A">
  <img alt="physics" src="https://img.shields.io/badge/focus-plasma%20ring%20currents-111111?style=for-the-badge&labelColor=111111&color=F2F2F2">
</p>

*Numerical simulation tools for visualizing ring-current-like charged-particle dynamics in a Planeterrella-inspired dipole field and low-pressure collisional environment.*

</div>

---

## Overview

This repository contains Python and Fortran code for simulating charged electron trajectories in a Planeterrella-style apparatus, where particles are launched from a biased magnetized sphere and evolved under electromagnetic fields and collisional effects.

The project’s scientific framing matches the IMPACT/LASP Planeterrella context shown in your slides:

- Dipole magnetic geometry and trapped/near-trapped particle behavior.
- Guiding-center drifts that help explain ring-like structures.
- Electron-neutral collisions modeled with Monte Carlo collision probability.
- Visualization/output workflows for trajectories and ring-current interpretation.

---

## Minimal Quick Start (n particles, n steps)

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib numba
```

### 2) Run a minimal simulation

> Example: run **`n=200` particles** for approximately **`n=200` integration steps** by choosing `tmax = n*dt`.

```bash
python 1_codes/planeterrella_sim.py \
  --nparticles 200 \
  --dt 5e-12 \
  --tmax 1e-9 \
  --store_stride 5 \
  --energy 20 \
  --V0 150 \
  --M 250
```

This run will:

- Launch particles from the source sphere surface.
- Integrate trajectories under electric + dipole magnetic fields.
- Stop trajectories on wall-hit / sphere-reentry conditions.
- Write the longest sampled trajectory to `traj.csv`.
- Display a 3D trajectory plot.

---

## User Manual Guide (Structured, High-Readability Summary)

The repository includes a full user manual document at:

- `0_info/terella_user_manual.docx`

Below is a practical, section-by-section guide to what a new user should expect from that manual and how to use it effectively.

### 1) Scientific Motivation & Context

This section explains *why* Planeterrella ring currents matter:

- Historical Terra/Terrella lineage (Birkeland-style laboratory space physics).
- Connection to auroral and magnetospheric plasma phenomena.
- Why ring-current-like structures are observed under specific voltage/pressure windows.
- How a controlled lab setup can mimic essential single-particle plasma physics.

**How to use it:** Read first for conceptual grounding before touching parameters.

### 2) Device / Experiment Configuration

This section describes the physical setup that the simulation approximates:

- Magnetized source sphere acting as star/planet analog.
- Bias potential accelerating electrons outward.
- Chamber pressure regime where mean free path permits visible collisional emission.
- Why low-pressure neutral background enables discrete glow structures.

**How to use it:** Map each hardware parameter to simulation constants (`V0`, dipole moment `M`, geometry radii, launch energy).

### 3) Single-Particle Motion Theory

This section introduces the motion decomposition used to interpret trajectories:

- Gyromotion about local magnetic field lines.
- Bounce/mirror behavior at strong-field regions.
- Guiding-center drift motion (especially \\(\mathbf{E}\times\mathbf{B}\\), curvature drift, and gradient drift).
- Pitch angle and mirror-point interpretation.

**How to use it:** Use this section when reading trajectory plots—identify which motion scale dominates each segment.

### 4) Collision Physics & Kinetic Parameters

This section explains low-pressure plasma collision handling:

- Cross section \\(\sigma\\), number density \\(n\\), mean free path \\(\lambda_{mfp} = 1/(\sigma n)\\).
- Collision frequency scaling \\(\nu_{coll} = \sigma n v\\).
- Why higher collision rates reduce acceleration time between collisions.
- Ionization-threshold relevance for photon-producing events.

**How to use it:** Tune pressure/collision terms to move between weakly collisional and strongly collisional ring behavior.

### 5) Numerical Simulation Development

This section covers algorithm design and assumptions:

- Time integration of Lorentz dynamics.
- Monte Carlo collision check per timestep.
- Particle termination conditions and trajectory filtering.
- Why electrons are tracked explicitly while neutral background is statistical.

**How to use it:** Consult while modifying timestep, particle count, or collision model.

### 6) Results, Interpretation, and Diagnostics

This section presents interpretation of output patterns:

- Primary/secondary ring appearance and spacing.
- Dependence on potential, pressure, and launch conditions.
- How to compare simulated rings to experiment imagery.

**How to use it:** Treat as the benchmark for “reasonable” model behavior.

### 7) Future Work / Extensions

Typical extension themes include:

- More realistic cross-section energy dependence.
- Adaptive integration and error control strategies.
- Extended field models and multi-species plasma effects.

**How to use it:** Start here when planning major model improvements.

---

## Basic Physics Implemented

## 1) Equation of Motion (Lorentz Force)

Particle velocity evolves by:

\\[
\frac{d\mathbf{v}}{dt} = \frac{q}{m}\left(\mathbf{E} + \mathbf{v}\times\mathbf{B}\right)
\\]

and position by \\(d\mathbf{r}/dt = \mathbf{v}\\).

In this repository, the fields are modeled as:

- **Dipole-like magnetic field** from magnetic moment \\(\mathbf{M}\\).
- **Radial sphere electric field** derived from sphere bias potential.

## 2) Guiding-Center Drift Quantities (Interpretive)

For weak field nonuniformity over a gyro-orbit, useful drifts include:

- \\(\mathbf{E}\times\mathbf{B}\\) drift:
  \\[
  \mathbf{v}_{E\times B}=\frac{\mathbf{E}\times\mathbf{B}}{|\mathbf{B}|^2}
  \\]
- Curvature and gradient drifts (sign/magnitude depend on charge and parallel/perpendicular energy partitions).

These drifts help explain azimuthal ring-current organization in dipolar geometry.

## 3) Monte Carlo Collision (MCC) Method

Per timestep \\(\Delta t\\), a collision probability is tested using:

\\[
P_{coll} = 1 - \exp\left(-n\sigma v\,\Delta t\right)
\\]

where:

- \\(n\\): neutral number density
- \\(\sigma\\): effective collision cross section
- \\(v\\): particle speed

Algorithmically:

1. Draw uniform random \\(u\sim U(0,1)\\).
2. If \\(u < P_{coll}\\), register a collision event.
3. Optionally test impact energy against ionization threshold to mark emissive events.

This is the core probability logic shown in your MCC slide.

## 4) Runge–Kutta Adaptive 4/5 (RK45) — Brief Integration Note

Your slides discuss a Runge–Kutta formulation; in general, **adaptive RK45** for Lorentz dynamics works as follows:

- Two embedded estimates (4th and 5th order) are computed each step.
- Their difference estimates local truncation error.
- The next timestep is enlarged/reduced to satisfy tolerance.

Why it helps:

- Resolves fast gyromotion without globally tiny fixed steps.
- Controls error near strong-field regions (e.g., close to mirror points).

In this codebase, trajectory pushing is currently done with a **Boris pusher** implementation (a standard plasma integrator), while RK45 remains a useful conceptual reference and alternative integrator option.

---

## Repository Layout

```text
0_info/      reference docs/manuals/slides
1_codes/     simulation scripts and utilities
2_figures/   generated analysis figures and PDFs
```

Key files:

- `1_codes/planeterrella_sim.py` — core particle simulation and plotting.
- `1_codes/mcc_terrella.py` — collisional trajectory simulation utilities.
- `1_codes/collisionalgo.py` — collision algorithm helpers.
- `1_codes/integrator.f95` — Fortran integration-related implementation.

---

## Notes

- Units are SI unless otherwise stated.
- For high particle counts, use Numba-enabled paths where available.
- If you run on Linux/headless systems, you may need to switch Matplotlib backend from `MacOSX` to `Agg` or `Qt5Agg`.

