"""Studies built on the impulse-momentum solution.

1. solve_impact    : one scenario -> results + conservation check
2. simulate_orbit  : time integration (solve_ivp) of the struck satellite's orbit after the
                     impact, with E = K + U and L_z tracked at every time step
3. parametric sweeps (mass x speed, angle, mass ratio) and extreme cases
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

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
# 2. Post-impact orbit (time integration)
# ----------------------------------------------------------------------------
def post_impact_body(s, r):
    """Mass, moment of inertia, CoM velocity and spin of the body that stays in orbit.

    Elastic: the satellite alone (the fragment bounces away).
    Completely inelastic: satellite + fragment, I_c from the parallel-axis theorem.
    `r` is the scalar_solution of scenario `s`.
    """
    m, M, I = s.debris_mass, s.sat_mass, s.sat_inertia
    dV = np.array([r["sat_dv_x_m_s"], r["sat_dv_y_m_s"]])
    if s.collision_type == "elastic":
        return M, I, np.zeros(2), dV, r["sat_omega_rad_s"]
    r_c = m * s.panel_point / (m + M)                  # new CoM from the satellite CoM
    I_c = I + M * pm.dot(r_c, r_c) + m * pm.dot(s.panel_point - r_c, s.panel_point - r_c)
    vd1 = np.array([r["debris_vx_after_m_s"], r["debris_vy_after_m_s"]])
    v_cm = (m * vd1 + M * dV) / (m + M)                # total momentum / total mass
    return m + M, I_c, r_c, v_cm, r["sat_omega_rad_s"]


def _integrate(sim, mu, y0, t_eval):
    sol = solve_ivp(pm.gravity_rhs, (0.0, t_eval[-1]), y0, method=sim.method,
                    t_eval=t_eval, rtol=sim.rtol, atol=sim.atol, args=(mu,),
                    vectorized=True)
    if not sol.success:
        raise RuntimeError(f"solve_ivp ({sim.method}) failed: {sol.message}")
    return sol.y


def simulate_orbit(s, r, constants):
    """Integrate the orbit after the impact and the un-hit reference orbit.

    Before the impact the satellite is on a circular orbit of radius R_E + h, at
    (R_E + h, 0) moving along +y, so x is radial and y along-track, as in the
    collision frame. Returns the tidy time-series DataFrame (simulation_results.csv).
    """
    sim, mu = s.simulation, constants.mu_earth
    radius = constants.earth_radius + s.altitude
    v_orb = float(pm.orbital_speed(mu, radius))
    t = np.arange(0.0, sim.t_max + 0.5 * sim.dt, sim.dt)

    mass, inertia, r_c, v_cm, omega = post_impact_body(s, r)
    y0 = np.array([radius + r_c[0], r_c[1], v_cm[0], v_orb + v_cm[1], 0.0, omega])
    hit = _integrate(sim, mu, y0, t)
    ref = _integrate(sim, mu, np.array([radius, 0.0, 0.0, v_orb, 0.0, 0.0]), t)

    kinetic, potential, total = pm.mechanical_energy(hit, mass, inertia, mu)
    L = pm.angular_momentum_z(hit, mass, inertia)
    # displacement from the un-hit orbit, in the local radial / along-track directions
    d = hit[:2] - ref[:2]
    e_r = ref[:2] / np.hypot(*ref[:2])
    e_t = pm.z_cross(e_r)
    zeros = np.zeros_like(t)
    return pd.DataFrame({
        "time_t": t,
        "pos_x": hit[0], "pos_y": hit[1], "pos_z": zeros,
        "vel_x": hit[2], "vel_y": hit[3], "vel_z": zeros,
        "kinetic_energy": kinetic,
        "potential_energy": potential,
        "total_energy": total,
        "energy_rel_error": np.abs((total - total[0]) / total[0]),
        "angular_momentum_z": L,
        "angular_momentum_rel_error": np.abs((L - L[0]) / L[0]),
        "spin_angle_rad": hit[4],
        "spin_rate_rad_s": hit[5],
        "deviation_radial_m": pm.dot(d, e_r),
        "deviation_along_track_m": pm.dot(d, e_t),
        "scenario": s.name,
    })


# ----------------------------------------------------------------------------
# 3. Parametric studies (vectorised)
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
        sat_v = r["sat_dv_mag_m_s"] / s.speed
        debris_v = -pm.dot(debris_v, n) / s.speed
        # 1-D collision formulas of the slides (m1 = m moving at v0, m2 = M at rest)
        if kind == "elastic":
            slide_debris, slide_sat = (ratio - 1) / (ratio + 1), 2 * ratio / (ratio + 1)
        else:
            slide_debris = slide_sat = ratio / (ratio + 1)
        frames.append(pd.DataFrame({
            "mass_ratio_m_over_M": ratio, "collision_type": kind,
            "sat_dv_over_v0": sat_v,
            "debris_v_after_over_v0": debris_v,
            "dissipated_fraction": r["dissipated_fraction"],
            "slide_formula_error": np.maximum(abs(sat_v - slide_sat),
                                              abs(debris_v - slide_debris)),
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
            "momentum_rel_error": cons["momentum_rel_error"],
            "angular_momentum_rel_error": cons["angular_momentum_rel_error"],
            "energy_rel_error": cons["energy_rel_error"],
        })
    return pd.DataFrame(rows)
