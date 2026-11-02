"""Project 14.2 - Space Debris Impact on Solar Panels
(General topic 14: Conservation of Linear Momentum in Collisions)

Usage
    uv run main.py                                   # all scenarios + parametric study
    uv run main.py inputs/scenarios/baseline.json    # one (or several) scenarios only
    uv run main.py --no-parametric                   # skip the parametric study
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import plotting
import solvers
from input_parser import (PROJECT_ROOT, load_constants, load_materials,
                          load_parametric_config, load_scenario)

INPUTS = PROJECT_ROOT / "inputs"
OUTPUTS = PROJECT_ROOT / "outputs"
VISUALS = PROJECT_ROOT / "visualizations"


def run_scenario(path, constants, materials):
    s = load_scenario(path, materials)
    print(f"\n=== Scenario: {s.name} ===\n    {s.description}")

    summary, conservation = solvers.solve_impact(s, constants)

    out = OUTPUTS / s.name
    out.mkdir(parents=True, exist_ok=True)
    pd.Series(summary, name="value").to_csv(out / "summary.csv", index_label="quantity")
    conservation.to_csv(out / "conservation.csv", index=False)
    plotting.plot_budget(summary, s, VISUALS / s.name / "01_momentum_energy_budget.png")

    print(f"    debris: {s.debris_mass:g} kg {s.debris_material} "
          f"(d = {summary['debris_diameter_mm']:.2f} mm), {s.speed / 1e3:g} km/s, "
          f"{s.impact_angle_deg:g} deg, {s.collision_type}")
    print(f"    impulse J = {summary['impulse_Ns']:.4g} N s, "
          f"beta = {summary['momentum_transfer_beta']:.4g}")
    print(f"    satellite dV = {summary['sat_dv_mag_m_s'] * 1e3:.4g} mm/s "
          f"({summary['sat_dv_over_orbital_speed']:.2e} of orbital speed), "
          f"spin = {np.degrees(summary['sat_omega_rad_s']):.4g} deg/s")
    print(f"    energy dissipated = {summary['dissipated_energy_J']:.4g} J "
          f"of {summary['impact_energy_J']:.4g} J")
    print(f"    conservation residuals: |dP|/P = {summary['momentum_rel_error']:.1e}, "
          f"|dL|/L = {summary['angular_momentum_rel_error']:.1e}, "
          f"|dE|/E = {summary['energy_rel_error']:.1e}")
    return summary, conservation


def run_parametric(path, materials):
    cfg = load_parametric_config(path)
    base = load_scenario(PROJECT_ROOT / cfg["reference_scenario"], materials)
    print(f"\n=== Parametric study (reference: {base.name}) ===")
    out, vis = OUTPUTS / "parametric", VISUALS / "parametric"
    out.mkdir(parents=True, exist_ok=True)

    grid = solvers.mass_speed_grid(base, cfg["mass_speed_grid"])
    grid.to_csv(out / "mass_speed_grid.csv", index=False)
    plotting.plot_mass_speed_map(grid, vis / "01_mass_speed_dv_map.png")

    for key, solve, plot, png in [
            ("impact_angle_sweep", solvers.impact_angle_sweep, plotting.plot_angle_sweep,
             "02_impact_angle_sweep.png"),
            ("mass_ratio_sweep", solvers.mass_ratio_sweep, plotting.plot_mass_ratio_sweep,
             "03_mass_ratio_sweep.png")]:
        df = solve(base, cfg[key])
        df.to_csv(out / f"{key}.csv", index=False)
        plot(df, vis / png)

    ext = solvers.extreme_cases(base, cfg["extreme_cases"])
    ext.to_csv(out / "extreme_cases.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.4g}".format):
        print("\n    Extreme cases:")
        print(ext[["case", "sat_dv_m_s", "debris_v_after_m_s", "dissipated_fraction",
                   "momentum_rel_error"]].to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Space debris impact on solar panels")
    parser.add_argument("scenarios", nargs="*", type=Path,
                        help="scenario JSON files (default: all in inputs/scenarios)")
    parser.add_argument("--parametric", type=Path, default=INPUTS / "parametric_study.json",
                        help="parametric-study JSON file")
    parser.add_argument("--no-parametric", action="store_true",
                        help="skip the parametric study")
    args = parser.parse_args(argv)

    constants = load_constants()
    materials = load_materials()
    paths = args.scenarios or sorted((INPUTS / "scenarios").glob("*.json"))
    if not paths:
        sys.exit("No scenario files found.")

    summaries, conservations, failed = [], [], []
    for path in paths:
        try:
            summary, conservation = run_scenario(path, constants, materials)
            summaries.append(summary)
            conservations.append(conservation)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            print(f"\n[ERROR] {path}: {exc}", file=sys.stderr)
            failed.append(path)

    OUTPUTS.mkdir(exist_ok=True)
    if summaries:
        pd.DataFrame(summaries).to_csv(OUTPUTS / "scenario_summary.csv", index=False)
        all_conservation = pd.concat(conservations, ignore_index=True)
        all_conservation.to_csv(OUTPUTS / "conservation.csv", index=False)
        if not args.no_parametric:
            plotting.plot_conservation(all_conservation,
                                       VISUALS / "parametric" / "04_conservation.png")

    if not args.no_parametric:
        run_parametric(args.parametric, materials)

    print(f"\nDone. Tables in {OUTPUTS.relative_to(PROJECT_ROOT)}/, "
          f"figures in {VISUALS.relative_to(PROJECT_ROOT)}/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
