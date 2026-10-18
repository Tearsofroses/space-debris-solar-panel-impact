# 14.2 – Space Debris Impact on Solar Panels

General Physics 1 (S261, class CC01), group project.
General topic **14 – Conservation of Linear Momentum in Collisions**.

A small debris fragment strikes a satellite's solar panel at orbital speed. This code works out
how much momentum and angular momentum the hit gives the satellite, how much kinetic energy
turns into heat and crater damage, and how the resulting Δv moves the satellite out of its
orbital slot. Every result is checked against conservation laws and closed-form solutions.

## Quick start

```bash
# one-time: install uv (https://docs.astral.sh/uv/), then inside this folder
uv sync                      # creates .venv with numpy, scipy, pandas, matplotlib

uv run main.py               # all scenarios + parametric study  (~40 s)
uv run main.py inputs/scenarios/baseline.json --no-parametric   # one scenario only
```

To try a new case, copy a file in `inputs/scenarios/`, change the numbers and run it. Nothing
physical is hard-coded in the Python files.

## Project structure (as in the course guidelines)

```
main.py            orchestration: load inputs → solve → write CSV → plot
physics_model.py   equations: impulse-momentum, contact force, CW orbit equations, energy/momentum
solvers.py         scipy solve_ivp integrations, parametric sweeps, Monte Carlo, validation
input_parser.py    reads JSON/CSV, rejects non-physical inputs (m ≤ 0, e ∉ [0,1], ...)
plotting.py        all figures (matplotlib, 300 dpi)
data/              constants.json (μ_Earth, R_Earth, thresholds), materials.csv (densities)
inputs/            scenarios/*.json (one impact each), parametric_study.json
outputs/           CSV tables (generated)
visualizations/    PNG figures (generated)
```

## Physics model

**Frame:** LVLH frame in the orbital plane. *x* is radial, *y* is along-track (direction of
flight) and *z* is the orbit normal (spin axis). The satellite is a rigid body with mass *M* and
moment of inertia *I*. The fragment hits the panel at lever arm **r** from the centre of mass.
The panel's outward normal is **n**. The fragment is a point mass *m* arriving at speed *v₀*,
at angle θ from the normal.

**1. Collision: impulse-momentum (analytic).** The contact is frictionless with coefficient of
restitution *e*.

| quantity | formula |
|---|---|
| effective mass | 1/m_eff = 1/m + 1/M + (r × n)²/I |
| normal impulse | J = (1 + e) m_eff v₀ cos θ |
| satellite | ΔV = −J n / M,  Δω = (r × (−J n)) / I |
| debris | Δv = +J n / m |
| energy lost | E_diss = ½ m_eff (v₀ cos θ)² (1 − e²) |
| momentum transfer efficiency | β = J / (m v₀ cos θ) = (1 + e) m_eff / m |

The total linear momentum m**v** + M**V** is the same before and after the hit, for every *e*.
The hit loses kinetic energy unless *e* = 1.

**2. Collision: time-resolved (numerical).** During contact the panel pushes back with a
linear spring-dashpot force f = kδ + c dδ/dt, where δ is the penetration depth. This lets
`solve_ivp` (method **DOP853**, rtol = 1e‑10) follow the hit microsecond by microsecond.
- k = m_eff (π/t_c)² sets the contact time t_c (1 µs by default).
- The damping ratio ζ = −ln e / √(π² + ln² e) makes the rebound reproduce the target *e*.
- e = 0 gives critical damping, so the fragment stays embedded.
- An event function detects separation (δ → 0).

At every time step the code tracks:
- total linear momentum **P**
- angular momentum L_z
- total energy K + U_spring + E_dissipated

All three stay constant to about 1e‑15 (see `05_momentum_conservation.png` and
`04_energy_error.png`).

**3. After the hit: orbital drift.** The Δv feeds the Clohessy–Wiltshire (Hill) equations
ẍ = 3n²x + 2nẏ, ÿ = −2nẋ, integrated with `solve_ivp` and checked against their exact solution.
Their conserved Jacobi integral ½(ẋ² + ẏ²) − (3/2)n²x² serves as the energy check. The code also
reports the change in semi-major axis (vis-viva) and the along-track drift per orbit.

**4. Limits of applicability.** Each run reports the impact energy-to-mass ratio
EMR = ½mv₀²/M. Above **40 J/g** (NASA Standard Breakup Model) the satellite breaks apart, so a
rigid-body momentum result has no physical meaning. The run flags this (see the
`catastrophic_steel_fragment` scenario and the orange line in `01_mass_speed_dv_map.png`).

## Scenarios (`inputs/scenarios/`)

| file | case |
|---|---|
| `baseline.json` | 1 g Al fragment, 10 km/s, 30°, e = 0.3, 500 kg satellite, hit 4 m from CoM |
| `head_on_embedding.json` | normal hit, perfectly inelastic (e = 0) |
| `grazing_elastic.json` | 80° grazing hit, perfectly elastic (e = 1) |
| `catastrophic_steel_fragment.json` | 2 kg steel at 12 km/s: beyond the model's validity |

`inputs/parametric_study.json` drives:
- a debris mass × speed map
- impact-angle, restitution and mass-ratio sweeps (the textbook 1-D collision; m/M from 10⁻⁶ to 100)
- a 20 000-impact Monte Carlo with fixed seed 42, so it is reproducible
- the extreme cases: equal masses, m ≫ M, grazing, hit at the CoM, e = 0 / 1

## Outputs

For each scenario, in `outputs/<scenario>/`:
- `simulation_results.csv`: impact time series. It uses the guideline schema (`time_t`,
  `pos_*`, `vel_*`, `kinetic_energy`, `potential_energy`, `total_energy`; primary body is the
  satellite) plus debris state, contact force, momenta and error columns.
- `post_impact_orbit.csv`: CW drift, numerical vs analytic
- `summary.csv`: key results (Δv, spin, energy lost, Δa, EMR, maximum conservation errors)
- `validation.csv`: numerical vs analytic

`outputs/` also holds the combined tables `scenario_summary.csv` and `validation.csv`, and the
parametric tables in `outputs/parametric/`.

Figures for each scenario are in `visualizations/<scenario>/`:
1. trajectory
2. time series
3. phase space
4. energy and error
5. momentum conservation

The parametric figures are in `visualizations/parametric/`.

## Assumptions and known limitations (useful for the Q&A)

- **2-D, frictionless contact.** Only the normal impulse is transferred. Any tangential
  friction impulse is neglected.
- **Rigid satellite, point-mass fragment.** In reality a hypervelocity fragment (> 3 km/s)
  melts or vaporises and throws out crater ejecta, which can push β above 2. Here *e* is an
  effective parameter that lumps that behaviour together. The fragment is also assumed to stop
  in the panel or bounce off it, not to perforate it.
- **Spring-dashpot artifact.** The linear spring-dashpot pulls slightly at the very end of
  contact (the negative force just before separation in `02_time_series.png`). This is the
  known artifact of that model. It is kept because it reproduces *e* exactly.
- **Numerical vs analytic spin rate.** The two differ by about 10⁻⁴ to 10⁻³. This is real
  physics, not numerical error: the fragment slides a few mm along the panel during the 1 µs
  contact, which changes the lever arm.
- **Orbit model.** CW equations assume a circular reference orbit and small offsets. No drag,
  J2 or attitude control is modelled.
