# 14.2 – Space Debris Impact on Solar Panels

General Physics 1 (S261, class CC01), group project.
General topic **14 – Conservation of Linear Momentum in Collisions**.

A small debris fragment strikes a satellite's solar panel at orbital speed. This code works out
how much momentum and angular momentum the hit gives the satellite, how much kinetic energy
turns into heat and crater damage, and how big the resulting Δv is compared with the orbital
speed. Every result is checked against the conservation laws.

The model uses only lecture-slide material (Ch2-Ch3): impulse-momentum, conservation of linear
and angular momentum, torque and moment of inertia, elastic / inelastic collisions, kinetic
energy, gravity as centripetal force.

## Quick start

```bash
uv sync                      # creates .venv with numpy, pandas, matplotlib
uv run main.py               # all scenarios + parametric study
uv run main.py inputs/scenarios/baseline.json --no-parametric   # one scenario only
```

To try a new case, copy a file in `inputs/scenarios/`, change the numbers and run it. Nothing
physical is hard-coded in the Python files.

## Project structure

```
main.py            orchestration: load inputs -> solve -> write CSV -> plot
physics_model.py   equations: impulse-momentum, effective mass, conservation checks
solvers.py         one-scenario solution and parametric sweeps
input_parser.py    reads JSON/CSV, rejects non-physical inputs (m <= 0, e not in [0,1], ...)
plotting.py        all figures (matplotlib, 300 dpi)
data/              constants.json (G*M_Earth, R_Earth), materials.csv (densities)
inputs/            scenarios/*.json (one impact each), parametric_study.json
outputs/           CSV tables (generated)
visualizations/    PNG figures (generated)
```

## Physics model

**Frame:** 2-D, orbital plane. *x* is radial, *y* is along-track, *z* is the orbit normal (spin
axis). The satellite is a rigid body of mass *M* and moment of inertia *I*, at rest before the
hit. The fragment is a point mass *m* arriving at speed *v₀*, at angle θ from the panel normal
**n**, at lever arm **r** from the centre of mass (CoM). Contact is frictionless, so only a
normal impulse *J* acts.

**1. Collision.** Two cases from the slides: *elastic* (kinetic energy conserved) and
*completely inelastic* (the fragment sticks to the panel).

| quantity | formula |
|---|---|
| impulse-momentum, debris | m Δv = +J n |
| impulse-momentum, satellite | M ΔV = −J n |
| torque on satellite about CoM | I Δω = r × (−J n) |
| change of relative normal speed per unit impulse | k = 1/m + 1/M + (r × n)²/I |
| elastic: relative normal speed reverses | J = 2 v₀ cos θ / k |
| completely inelastic: relative normal speed becomes 0 | J = v₀ cos θ / k |
| energy lost | 0 (elastic), (v₀ cos θ)² / (2k) (inelastic) |

**2. Conservation checks** (`physics_model.conservation_check`). For each impact the code
compares before and after:
- total linear momentum m**v** + M**V**
- angular momentum about the CoM, m (**r** × **v**) + Iω
- energy: K_debris + K_satellite (translation) + K_satellite (rotation) + E_diss = ½ m v₀²

The residuals are at round-off level (~1e-16).

**3. Orbit.** Gravity acts as the centripetal force, G M m / r² = m v²/r, so the circular
orbital speed is v = √(GM/r). The code reports ΔV / v_orbit to show how small the kick is.

## Assumptions and limitations

- **Instantaneous collision.** External forces (gravity) are negligible during the very short
  contact, so momentum is conserved. The time-resolved contact is not modelled.
- **2-D, frictionless contact.** Tangential friction impulse is neglected.
- **Rigid satellite, point-mass fragment, two ideal collision types.** A real hypervelocity
  fragment melts or vaporises and falls between the elastic and completely inelastic limits.
- **Orbit.** Only the size of ΔV relative to the orbital speed is reported. Orbit
  perturbation theory is outside the course scope and is not modelled.

## Scenarios (`inputs/scenarios/`)

| file | case |
|---|---|
| `baseline.json` | 1 g Al fragment, 10 km/s, 30°, completely inelastic, 500 kg satellite, hit 4 m from CoM |
| `head_on_embedding.json` | normal hit, completely inelastic |
| `grazing_elastic.json` | 80° grazing hit, elastic |

`inputs/parametric_study.json` drives:
- a debris mass × speed map
- impact-angle and mass-ratio sweeps, each for elastic and completely inelastic (the textbook 1-D collision; m/M from 10⁻⁶ to 100)
- the extreme cases: equal masses, m ≫ M, grazing, hit at the CoM, elastic vs completely inelastic

## Outputs

For each scenario, in `outputs/<scenario>/`:
- `summary.csv`: Δv, spin, energy lost, β, ΔV / v_orbit, conservation residuals
- `conservation.csv`: before / after values of P, L and the energy balance

`outputs/` also holds the combined `scenario_summary.csv` and `conservation.csv`, and the
parametric tables in `outputs/parametric/`.

Figures: `visualizations/<scenario>/01_momentum_energy_budget.png` for each scenario, and the
parametric figures in `visualizations/parametric/` (ΔV map, angle sweep,
mass-ratio sweep, conservation residuals).
