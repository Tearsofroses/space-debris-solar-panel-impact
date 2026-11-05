# 14.2 – Space Debris Impact on Solar Panels

General Physics 1 (S261, class CC01), group project.
General topic **14 – Conservation of Linear Momentum in Collisions**.

A small debris fragment strikes a satellite's solar panel at orbital speed. This code works out
how much momentum and angular momentum the hit gives the satellite, how much kinetic energy
turns into heat and crater damage, and how big the resulting Δv is compared with the orbital
speed. Every result is checked against the conservation laws.

The model uses only lecture-slide material (Ch2-Ch3): centre of mass, impulse-momentum,
conservation of linear and angular momentum, torque, moment of inertia and the parallel-axis
theorem, elastic / completely inelastic collisions, kinetic energy of translation and
rotation, gravity as centripetal force.

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
physics_model.py   equations: impulse-momentum, torque, conservation checks
solvers.py         one-scenario solution and parametric sweeps
input_parser.py    reads JSON/CSV, rejects non-physical inputs (m <= 0, unknown collision type,
                   impossible sweep ranges, ...)
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
**n**, at lever arm **r** from the centre of mass (CoM).

**1. Collision.** Two cases from the slides: *elastic* (kinetic energy conserved) and
*completely inelastic* (the fragment sticks to the panel; `collision_type` values `elastic`
and `completely_inelastic`).

*Elastic* (frictionless contact, so only a normal impulse *J* along **n**):

| quantity | formula |
|---|---|
| impulse-momentum, debris | m Δv = +J n |
| impulse-momentum, satellite | M ΔV = −J n |
| torque on satellite about CoM | I Δω = r × (−J n) |
| change of relative normal speed per unit impulse | k = 1/m + 1/M + (r × n)²/I |
| relative normal speed reverses (K conserved) | J = 2 v₀ cos θ / k |

Where k comes from: the first three rows say one unit of impulse changes the debris's normal
speed by 1/m, the satellite's by 1/M, and the struck point's spin speed by (r × n)²/I (the
lever arm enters once in the torque and once in v = ωr). For a hit through the CoM,
(r × n) = 0 and the result is the slides' 1-D elastic collision formula.

*Completely inelastic*: after the hit, satellite + fragment is one rigid body, solved like the
slides' "collision of extended bodies in 2-D":

| quantity | formula |
|---|---|
| new centre of mass (from satellite CoM) | r_c = m r / (m + M) |
| conservation of linear momentum | m v₀ = (m + M) v_CM |
| moment of inertia about the new CoM (parallel-axis theorem) | I_c = I + M r_c² + m (r − r_c)² |
| conservation of angular momentum about the new CoM | m (r − r_c) × v₀ = I_c ω |
| velocity of any point of the combined body | v = v_CM + ω × (position from new CoM) |
| energy lost (heat, crater, sound) | ½ m v₀² − ½ (m + M) v_CM² − ½ I_c ω² |

The fragment moves with the struck point afterwards, in both the normal and tangential
directions, so for m ≪ M almost all of its kinetic energy is lost, fraction M/(m + M) for a
hit through the CoM (compare the slides' "bullet embeds in a wooden block" problem).

**2. Conservation checks** (`physics_model.conservation_check`). For each impact the code
compares before and after:
- total linear momentum m**v** + M**V**
- angular momentum about the CoM, m (**r** × **v**) + Iω
- energy: K_debris + K_satellite (translation) + K_satellite (rotation) + E_diss = ½ m v₀²

The residuals are at round-off level (~1e-16). The head-on, through-CoM mass-ratio sweep is
also compared with the slides' 1-D formulas v₁' = (m₁ − m₂)v₁/(m₁ + m₂),
v₂' = 2m₁v₁/(m₁ + m₂) (elastic) and v = m₁v₁/(m₁ + m₂) (completely inelastic).

**3. Orbit.** Gravity acts as the centripetal force, G M m / r² = m v²/r, so the circular
orbital speed is v = √(GM/r). The code reports ΔV / v_orbit to show how small the kick is.

## Assumptions and limitations

- **Instantaneous collision.** External forces (gravity) are negligible during the very short
  contact, so momentum is conserved. The time-resolved contact is not modelled.
- **2-D.** Elastic case: frictionless contact (no tangential impulse). Completely inelastic
  case: the fragment sticks, so the contact also gives a tangential impulse.
- **Rigid satellite, point-mass fragment, two ideal collision types.** A real hypervelocity
  fragment melts or vaporises and falls between the elastic and completely inelastic limits.
- **Orbit.** Only the size of ΔV relative to the orbital speed is reported. Orbit
  perturbation theory is outside the course scope and is not modelled.
- **No time integration.** Because the collision is instantaneous, every result comes from
  closed-form conservation laws, so there is no equation of motion to integrate (no
  `solve_ivp`, no time series). Conservation is checked before vs after the hit instead of
  at every time step.
- **Deterministic.** There are no random numbers, so no random seed is needed and every run
  gives the same tables.

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
- `summary.csv`: one row with impulse, Δv, spin, energy lost, ΔV / v_orbit, conservation
  residuals
- `conservation.csv`: before / after values of P, L and the energy balance

`outputs/` also holds the combined `scenario_summary.csv` and `conservation.csv`, and the
parametric tables in `outputs/parametric/`.

Figures: `visualizations/<scenario>/01_momentum_energy_budget.png` for each scenario, and the
parametric figures in `visualizations/parametric/` (ΔV map, angle sweep,
mass-ratio sweep, conservation residuals).
