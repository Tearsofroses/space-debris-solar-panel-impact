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

import pandas as pd

import physics_model as pm
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

    impact_df, summary = solvers.solve_impact(s)
    orbit_df, orbit_info = solvers.solve_post_impact_orbit(
        s, constants, summary["sat_dv_x_m_s"], summary["sat_dv_y_m_s"],
        summary["sat_omega_rad_s"])
    summary.update(orbit_info)
    summary.update(pm.validity_checks(s.debris_mass, s.sat_mass, s.speed, constants))
    validation = solvers.compare_with_analytic(s, summary)

    out = OUTPUTS / s.name
    out.mkdir(parents=True, exist_ok=True)
    impact_df.to_csv(out / "simulation_results.csv", index=False)
    orbit_df.to_csv(out / "post_impact_orbit.csv", index=False)
    pd.Series(summary, name="value").to_csv(out / "summary.csv", index_label="quantity")
    validation.to_csv(out / "validation.csv", index=False)
    plotting.scenario_figures(impact_df, orbit_df, s, VISUALS / s.name)

    print(f"    debris: {s.debris_mass:g} kg {s.debris_material} "
          f"(d = {summary['debris_diameter_mm']:.2f} mm), {s.speed / 1e3:g} km/s, "
          f"{s.impact_angle_deg:g} deg, e = {s.restitution:g}")
    print(f"    satellite dV = {summary['sat_dv_mag_m_s'] * 1e3:.4g} mm/s, "
          f"spin = {summary['sat_omega_rad_s'] * 57.29577951308232:.4g} deg/s")
    print(f"    energy dissipated = {summary['dissipated_energy_J']:.4g} J "
          f"of {summary['impact_energy_J']:.4g} J")
    print(f"    semi-major axis change = {summary['delta_a_m']:.4g} m, "
          f"along-track drift = {summary['along_track_drift_per_orbit_m']:.4g} m/orbit")
    print(f"    max drift: |dP|/P = {summary['max_momentum_rel_error']:.1e}, "
          f"|dL|/L = {summary['max_angular_momentum_rel_error']:.1e}, "
          f"|dE|/E = {summary['max_energy_rel_error']:.1e}")
    if summary["catastrophic_breakup"]:
        print(f"    WARNING: EMR = {summary['emr_J_kg'] / 1e3:.3g} J/g >= "
              f"{constants.emr_threshold / 1e3:g} J/g -> catastrophic breakup expected; "
              f"rigid-body momentum results are NOT physically valid here.")
    return summary, validation


def run_parametric(path, constants, materials, validation_frames):
    cfg = load_parametric_config(path)
    base = load_scenario(PROJECT_ROOT / cfg["reference_scenario"], materials)
    print(f"\n=== Parametric study (reference: {base.name}) ===")
    out, vis = OUTPUTS / "parametric", VISUALS / "parametric"
    out.mkdir(parents=True, exist_ok=True)

    grid = solvers.mass_speed_grid(base, cfg["mass_speed_grid"], constants)
    grid.to_csv(out / "mass_speed_grid.csv", index=False)
    plotting.plot_mass_speed_map(grid, constants.emr_threshold, vis / "01_mass_speed_dv_map.png")

    ang = solvers.impact_angle_sweep(base, cfg["impact_angle_sweep"])
    ang.to_csv(out / "impact_angle_sweep.csv", index=False)
    plotting.plot_angle_sweep(ang, vis / "02_impact_angle_sweep.png")

    res = solvers.restitution_sweep(base, cfg["restitution_sweep"])
    res.to_csv(out / "restitution_sweep.csv", index=False)
    plotting.plot_restitution_sweep(res, vis / "03_restitution_sweep.png")

    ratio = solvers.mass_ratio_sweep(base, cfg["mass_ratio_sweep"])
    ratio.to_csv(out / "mass_ratio_sweep.csv", index=False)
    plotting.plot_mass_ratio_sweep(ratio, vis / "04_mass_ratio_sweep.png")

    mc_cfg = cfg["monte_carlo"]
    mc = solvers.monte_carlo(base, mc_cfg, constants)
    mc.to_csv(out / "monte_carlo_impacts.csv", index=False)
    plotting.plot_monte_carlo(mc, constants.emr_threshold, mc_cfg["spin_rate_alarm_deg_s"],
                              vis / "05_monte_carlo.png")
    mc_stats = mc[["debris_mass_kg", "impact_speed_m_s", "sat_dv_mag_m_s",
                   "sat_spin_deg_s"]].describe(percentiles=[0.5, 0.9, 0.99])
    mc_stats.to_csv(out / "monte_carlo_statistics.csv")
    print(f"    Monte Carlo ({len(mc):,} impacts, seed {mc_cfg['seed']}): "
          f"median dV = {mc['sat_dv_mag_m_s'].median() * 1e3:.3g} mm/s, "
          f"{mc['spin_alarm'].mean() * 100:.2f}% exceed the spin alarm, "
          f"{mc['catastrophic_breakup'].mean() * 100:.2f}% catastrophic")

    ext = solvers.extreme_cases(base, cfg["extreme_cases"], constants)
    ext.to_csv(out / "extreme_cases.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.4g}".format):
        print("\n    Extreme cases:")
        print(ext[["case", "sat_dv_numerical_m_s", "sat_dv_analytic_m_s",
                   "debris_v_after_m_s", "dissipated_fraction",
                   "max_momentum_rel_error"]].to_string(index=False))

    if validation_frames:
        plotting.plot_validation(pd.concat(validation_frames, ignore_index=True),
                                 vis / "06_validation.png")


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

    summaries, validations, failed = [], [], []
    for path in paths:
        try:
            summary, validation = run_scenario(path, constants, materials)
            summaries.append(summary)
            validations.append(validation)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            print(f"\n[ERROR] {path}: {exc}", file=sys.stderr)
            failed.append(path)

    OUTPUTS.mkdir(exist_ok=True)
    if summaries:
        pd.DataFrame(summaries).to_csv(OUTPUTS / "scenario_summary.csv", index=False)
        pd.concat(validations, ignore_index=True).to_csv(OUTPUTS / "validation.csv", index=False)

    if not args.no_parametric:
        run_parametric(args.parametric, constants, materials, validations)

    print(f"\nDone. Tables in {OUTPUTS.relative_to(PROJECT_ROOT)}/, "
          f"figures in {VISUALS.relative_to(PROJECT_ROOT)}/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
