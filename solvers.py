"""Studies built on the impulse-momentum solution.

1. solve_impact  : one scenario -> results + conservation check
2. parametric sweeps (mass x speed, angle, mass ratio) and extreme cases
"""

import numpy as np
import pandas as pd

import physics_model as pm
from input_parser import override_scenario


def analytic_for(s, m=None, speed=None, angle=None, collision=None, arm=None):
    """Impulse-momentum solution for scenario `s`; keyword values override its fields."""
    return pm.analytic_collision(
        s.debris_mass if m is None else m, s.sat_mass, s.sat_inertia,
        s.panel_point if arm is None else arm, s.panel_normal,
        s.speed if speed is None else speed,
        s.impact_angle if angle is None else angle,
        s.collision_type if collision is None else collision)


def scalar_solution(s):
    """analytic_for(s) as plain floats (one scenario, no sweep)."""
    return {k: float(v) for k, v in analytic_for(s).items()}


def conservation_for(s, r):
    """Conservation residuals of the solution `r` of scenario `s`."""
    return pm.conservation_check(r, s.debris_mass, s.sat_mass, s.sat_inertia, s.panel_point,
                                 s.panel_normal, s.speed, s.impact_angle)


# ----------------------------------------------------------------------------
# 1. One scenario
# ----------------------------------------------------------------------------
def solve_impact(s, constants):
    """Returns (summary dict, conservation DataFrame row)."""
    r = scalar_solution(s)
    cons = conservation_for(s, r)
    v_orb = float(pm.orbital_speed(constants.mu_earth, constants.earth_radius + s.altitude))
    summary = {
        "scenario": s.name,
        "debris_mass_kg": s.debris_mass,
        "debris_material": s.debris_material,
        "debris_diameter_mm": s.debris_diameter * 1e3,
        "impact_speed_m_s": s.speed,
        "impact_angle_deg": s.impact_angle_deg,
        "collision_type": s.collision_type,
        **r,
        "orbital_speed_m_s": v_orb,
        "sat_dv_over_orbital_speed": r["sat_dv_mag_m_s"] / v_orb,
        **{k: cons[k] for k in ("momentum_rel_error", "angular_momentum_rel_error",
                                "energy_rel_error")},
    }
    conservation = pd.DataFrame([
        {"scenario": s.name, "quantity": "linear momentum |P| (kg m/s)",
         "before": cons["momentum_before_kg_m_s"], "after": cons["momentum_after_kg_m_s"],
         "rel_error": cons["momentum_rel_error"]},
        {"scenario": s.name, "quantity": "angular momentum L_z about CoM (kg m^2/s)",
         "before": cons["angular_momentum_before"], "after": cons["angular_momentum_after"],
         "rel_error": cons["angular_momentum_rel_error"]},
        {"scenario": s.name, "quantity": "kinetic energy + dissipated (J)",
         "before": cons["energy_before_J"], "after": cons["energy_after_J"],
         "rel_error": cons["energy_rel_error"]},
    ])
    return summary, conservation


# ----------------------------------------------------------------------------
# 2. Parametric studies (vectorised)
# ----------------------------------------------------------------------------
def mass_speed_grid(s, cfg):
    m = np.logspace(np.log10(cfg["mass_min_kg"]), np.log10(cfg["mass_max_kg"]), cfg["n_mass"])
    v = np.linspace(cfg["speed_min_m_s"], cfg["speed_max_m_s"], cfg["n_speed"])
    M_grid, V_grid = np.meshgrid(m, v, indexing="ij")
    r = analytic_for(s, m=M_grid, speed=V_grid)
    return pd.DataFrame({
        "debris_mass_kg": M_grid.ravel(), "impact_speed_m_s": V_grid.ravel(),
        "sat_dv_mag_m_s": r["sat_dv_mag_m_s"].ravel(),
        "sat_omega_rad_s": r["sat_omega_rad_s"].ravel(),
        "dissipated_energy_J": r["dissipated_energy_J"].ravel(),
    })


def impact_angle_sweep(s, cfg):
    ang = np.linspace(cfg["angle_min_deg"], cfg["angle_max_deg"], cfg["n_angle"])
    frames = []
    for kind in cfg["collision_types"]:
        r = analytic_for(s, angle=np.radians(ang), collision=kind)
        frames.append(pd.DataFrame({
            "impact_angle_deg": ang, "collision_type": kind,
            "impulse_Ns": r["impulse_Ns"], "sat_dv_mag_m_s": r["sat_dv_mag_m_s"],
            "sat_omega_rad_s": r["sat_omega_rad_s"],
            "dissipated_fraction": r["dissipated_fraction"],
        }))
    return pd.concat(frames, ignore_index=True)


def mass_ratio_sweep(s, cfg):
    """Head-on hit through the CoM: the classic 1-D collision, m/M from tiny to huge."""
    ratio = np.logspace(np.log10(cfg["ratio_min"]), np.log10(cfg["ratio_max"]), cfg["n_points"])
    # along -n is "forward" for the debris; express both results as fractions of v0
    n = s.panel_normal
    frames = []
    for kind in cfg["collision_types"]:
        r = analytic_for(s, m=ratio * s.sat_mass, angle=0.0, collision=kind, arm=np.zeros(2))
        debris_v = (r["debris_vx_after_m_s"], r["debris_vy_after_m_s"])
        frames.append(pd.DataFrame({
            "mass_ratio_m_over_M": ratio, "collision_type": kind,
            "sat_dv_over_v0": r["sat_dv_mag_m_s"] / s.speed,
            "debris_v_after_over_v0": -pm.dot(debris_v, n) / s.speed,
            "dissipated_fraction": r["dissipated_fraction"],
        }))
    return pd.concat(frames, ignore_index=True)


def extreme_cases(base, cases):
    rows = []
    for case in cases:
        s = override_scenario(base, case)
        ref = scalar_solution(s)
        cons = conservation_for(s, ref)
        rows.append({
            "case": case["name"], "debris_mass_kg": s.debris_mass,
            "impact_angle_deg": s.impact_angle_deg, "collision_type": s.collision_type,
            "panel_arm_m": float(np.linalg.norm(s.panel_point)),
            "sat_dv_m_s": ref["sat_dv_mag_m_s"],
            "sat_omega_rad_s": ref["sat_omega_rad_s"],
            "debris_v_after_m_s": float(np.hypot(ref["debris_vx_after_m_s"],
                                                 ref["debris_vy_after_m_s"])),
            "dissipated_fraction": ref["dissipated_fraction"],
            "momentum_transfer_beta": ref["momentum_transfer_beta"],
            "momentum_rel_error": cons["momentum_rel_error"],
            "angular_momentum_rel_error": cons["angular_momentum_rel_error"],
            "energy_rel_error": cons["energy_rel_error"],
        })
    return pd.DataFrame(rows)
