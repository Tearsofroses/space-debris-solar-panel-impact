"""Numerical solvers.

1. solve_impact          : time-resolved collision (scipy solve_ivp, contact model)
2. solve_post_impact_orbit : satellite drift after the hit (Clohessy-Wiltshire)
3. parametric / Monte Carlo studies built on the analytic impulse solution
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import physics_model as pm
from input_parser import override_scenario


# ----------------------------------------------------------------------------
# 1. Impact phase
# ----------------------------------------------------------------------------
def build_impact_params(s):
    m_eff = pm.effective_mass(s.debris_mass, s.sat_mass, s.sat_inertia,
                              s.panel_point, s.panel_normal)
    k, c = pm.contact_parameters(m_eff, s.contact_duration, s.restitution)
    return {"m": s.debris_mass, "M": s.sat_mass, "I": s.sat_inertia,
            "arm_body": s.panel_point, "normal_body": s.panel_normal,
            "m_eff": m_eff, "k": k, "c": c}


def analytic_for(s):
    return pm.analytic_collision(s.debris_mass, s.sat_mass, s.sat_inertia, s.panel_point,
                                 s.panel_normal, s.speed, s.impact_angle, s.restitution)


def _tolerance_scales(s, ref, t_max):
    """Absolute tolerance per state component, from the expected size of each one."""
    length = s.speed * s.contact_duration
    dv = max(float(ref["sat_dv_mag_m_s"]), 1e-300)
    omega = max(abs(float(ref["sat_omega_rad_s"])), dv / max(np.linalg.norm(s.panel_point), length))
    energy = float(ref["impact_energy_J"])
    return np.array([length, length, s.speed, s.speed,
                     dv * t_max, dv * t_max, dv, dv,
                     omega * t_max, omega, energy])


def solve_impact(s):
    """Integrate the collision through contact and separation.

    Returns (DataFrame time series, dict summary).
    """
    p = build_impact_params(s)
    ref = analytic_for(s)
    t_max = s.t_max_factor * s.contact_duration

    y0 = np.zeros(11)
    y0[0:2] = s.panel_point                      # debris starts touching the panel
    y0[2:4] = pm.debris_velocity(s.speed, s.impact_angle, s.panel_normal)
    atol = s.rtol * 1e-3 * _tolerance_scales(s, ref, t_max)

    def separation(t, y, *_):                   # penetration returns to zero -> fragment leaves
        return pm.contact_geometry(y, p)[0]
    separation.terminal = True
    separation.direction = -1

    common = dict(method=s.method, rtol=s.rtol, atol=atol, dense_output=True)
    seg1 = solve_ivp(pm.impact_rhs, (0.0, t_max), y0, args=(p, True), events=separation,
                     max_step=s.contact_duration / 50.0, **common)
    if not seg1.success:
        raise RuntimeError(f"Impact integration failed: {seg1.message}")

    separated = seg1.status == 1
    t_sep = float(seg1.t[-1]) if separated else np.nan
    y_sep = seg1.y[:, -1]

    t_grid = np.linspace(0.0, t_max, s.n_output_impact)
    if separated:
        t_in, t_out = t_grid[t_grid <= t_sep], t_grid[t_grid > t_sep]
        seg2 = solve_ivp(pm.impact_rhs, (t_sep, t_max), y_sep, args=(p, False), **common)
        if not seg2.success:
            raise RuntimeError(f"Free-flight integration failed: {seg2.message}")
        Y = np.hstack([seg1.sol(t_in), seg2.sol(t_out)])
        contact_flag = np.r_[np.ones(t_in.size, bool), np.zeros(t_out.size, bool)]
    else:
        Y = seg1.sol(t_grid)
        contact_flag = np.ones(t_grid.size, bool)

    df = impact_dataframe(t_grid, Y, p, contact_flag)

    # --- summary -------------------------------------------------------------
    _, dd0, _, _ = pm.contact_geometry(y0, p)
    _, dd_sep, _, _ = pm.contact_geometry(y_sep, p)
    last = df.iloc[-1]
    summary = {
        "scenario": s.name,
        "debris_mass_kg": s.debris_mass,
        "debris_material": s.debris_material,
        "debris_diameter_mm": s.debris_diameter * 1e3,
        "impact_speed_m_s": s.speed,
        "impact_angle_deg": s.impact_angle_deg,
        "restitution_target": s.restitution,
        "restitution_effective": -dd_sep / dd0,
        "separated": separated,
        "contact_time_s": t_sep,
        "max_penetration_m": df["penetration_m"].max(),
        "peak_contact_force_N": df["contact_force_N"].max(),
        "sat_dv_x_m_s": last["vel_x"],
        "sat_dv_y_m_s": last["vel_y"],
        "sat_dv_mag_m_s": float(np.hypot(last["vel_x"], last["vel_y"])),
        "sat_omega_rad_s": last["angular_velocity_rad_s"],
        "debris_vx_after_m_s": last["debris_vel_x"],
        "debris_vy_after_m_s": last["debris_vel_y"],
        "impulse_Ns": s.sat_mass * float(np.hypot(last["vel_x"], last["vel_y"])),
        "impact_energy_J": df["total_energy"].iloc[0],
        "dissipated_energy_J": last["dissipated_energy"],
        "max_momentum_rel_error": df["momentum_rel_error"].max(),
        "max_angular_momentum_rel_error": df["angular_momentum_rel_error"].max(),
        "max_energy_rel_error": df["energy_rel_error"].max(),
        "contact_stiffness_N_m": p["k"],
        "contact_damping_Ns_m": p["c"],
        "damping_ratio": pm.damping_ratio(s.restitution),
    }
    return df, summary


def impact_dataframe(t, Y, p, contact_flag):
    delta, delta_dot, _, _ = pm.contact_geometry(Y, p)
    force = np.where(contact_flag, pm.contact_force(delta, delta_dot, p), 0.0)
    P, L = pm.system_momentum(Y, p)
    ke, pe, ed = pm.system_energy(Y, p, contact_flag)
    total = ke + pe + ed
    p_scale = np.hypot(P[0, 0], P[1, 0])
    l_scale = max(abs(L[0]), p_scale * np.linalg.norm(p["arm_body"]), p_scale * 1e-3)
    zeros = np.zeros_like(t)
    return pd.DataFrame({
        "time_t": t,
        # primary body = satellite centre of mass
        "pos_x": Y[4], "pos_y": Y[5], "pos_z": zeros,
        "vel_x": Y[6], "vel_y": Y[7], "vel_z": zeros,
        "attitude_rad": Y[8],
        "angular_velocity_rad_s": Y[9],
        # secondary body = debris fragment
        "debris_pos_x": Y[0], "debris_pos_y": Y[1],
        "debris_vel_x": Y[2], "debris_vel_y": Y[3],
        "in_contact": contact_flag,
        "penetration_m": np.where(contact_flag, delta, 0.0),
        "penetration_rate_m_s": delta_dot,        # > 0 while the debris digs into the panel
        "contact_force_N": force,
        "momentum_x": P[0], "momentum_y": P[1],
        "debris_momentum_x": p["m"] * Y[2], "debris_momentum_y": p["m"] * Y[3],
        "sat_momentum_x": p["M"] * Y[6], "sat_momentum_y": p["M"] * Y[7],
        "angular_momentum_z": L,
        "kinetic_energy": ke,
        "potential_energy": pe,
        "dissipated_energy": ed,
        "total_energy": total,
        "momentum_rel_error": np.hypot(P[0] - P[0, 0], P[1] - P[1, 0]) / p_scale,
        "angular_momentum_rel_error": np.abs(L - L[0]) / l_scale,
        "energy_rel_error": np.abs((total - total[0]) / total[0]),
    })


# ----------------------------------------------------------------------------
# 2. Post-impact orbital drift
# ----------------------------------------------------------------------------
def solve_post_impact_orbit(s, constants, dv_radial, dv_along, omega):
    a = constants.earth_radius + s.altitude
    n = pm.mean_motion(constants.mu_earth, a)
    t_end = s.n_orbits * 2.0 * np.pi / n
    t = np.linspace(0.0, t_end, s.n_output_orbit)
    y0 = np.array([0.0, 0.0, dv_radial, dv_along])

    v_scale = max(np.hypot(dv_radial, dv_along), 1e-300)
    atol = s.rtol * 1e-3 * np.array([v_scale / n, v_scale / n, v_scale, v_scale])
    sol = solve_ivp(pm.cw_rhs, (0.0, t_end), y0, args=(n,), method=s.method,
                    rtol=s.rtol, atol=atol, t_eval=t)
    if not sol.success:
        raise RuntimeError(f"CW integration failed: {sol.message}")

    x, y, vx, vy = sol.y
    xa, ya, _, _ = pm.cw_analytic(t, *y0, n)
    ke, pe = pm.cw_energy(s.sat_mass, n, x, vx, vy)
    total = ke + pe
    zeros = np.zeros_like(t)
    df = pd.DataFrame({
        "time_t": t,
        "pos_x": x, "pos_y": y, "pos_z": zeros,
        "vel_x": vx, "vel_y": vy, "vel_z": zeros,
        "attitude_rad": omega * t,                  # torque-free spin about z
        "analytic_pos_x": xa, "analytic_pos_y": ya,
        "position_error_m": np.hypot(x - xa, y - ya),
        "kinetic_energy": ke,
        "potential_energy": pe,
        "total_energy": total,
        "energy_rel_error": np.abs((total - total[0]) / total[0]),
    })
    info = pm.orbit_change(constants.mu_earth, a, dv_radial, dv_along)
    info.update({"semi_major_axis_m": a, "mean_motion_rad_s": n,
                 "max_cw_position_error_m": df["position_error_m"].max(),
                 "max_cw_energy_rel_error": df["energy_rel_error"].max()})
    return df, info


# ----------------------------------------------------------------------------
# 3. Parametric and Monte Carlo studies (vectorised analytic model)
# ----------------------------------------------------------------------------
def _analytic(s, m=None, speed=None, angle=None, e=None, arm=None):
    return pm.analytic_collision(
        s.debris_mass if m is None else m, s.sat_mass, s.sat_inertia,
        s.panel_point if arm is None else arm, s.panel_normal,
        s.speed if speed is None else speed,
        s.impact_angle if angle is None else angle,
        s.restitution if e is None else e)


def mass_speed_grid(s, cfg, constants):
    m = np.logspace(np.log10(cfg["mass_min_kg"]), np.log10(cfg["mass_max_kg"]), cfg["n_mass"])
    v = np.linspace(cfg["speed_min_m_s"], cfg["speed_max_m_s"], cfg["n_speed"])
    M_grid, V_grid = np.meshgrid(m, v, indexing="ij")
    r = _analytic(s, m=M_grid, speed=V_grid)
    return pd.DataFrame({
        "debris_mass_kg": M_grid.ravel(), "impact_speed_m_s": V_grid.ravel(),
        "sat_dv_mag_m_s": r["sat_dv_mag_m_s"].ravel(),
        "sat_omega_rad_s": r["sat_omega_rad_s"].ravel(),
        "dissipated_energy_J": r["dissipated_energy_J"].ravel(),
        "emr_J_kg": r["emr_J_kg"].ravel(),
        "catastrophic_breakup": (r["emr_J_kg"] >= constants.emr_threshold).ravel(),
    })


def impact_angle_sweep(s, cfg):
    ang = np.linspace(cfg["angle_min_deg"], cfg["angle_max_deg"], cfg["n_angle"])
    frames = []
    for e in cfg["restitution_values"]:
        r = _analytic(s, angle=np.radians(ang), e=e)
        frames.append(pd.DataFrame({
            "impact_angle_deg": ang, "restitution": e,
            "impulse_Ns": r["impulse_Ns"], "sat_dv_mag_m_s": r["sat_dv_mag_m_s"],
            "sat_omega_rad_s": r["sat_omega_rad_s"],
            "dissipated_fraction": r["dissipated_fraction"],
        }))
    return pd.concat(frames, ignore_index=True)


def restitution_sweep(s, cfg):
    e = np.linspace(0.0, 1.0, cfg["n_points"])
    r = _analytic(s, e=e)
    return pd.DataFrame({
        "restitution": e, "momentum_transfer_beta": r["momentum_transfer_beta"],
        "sat_dv_mag_m_s": r["sat_dv_mag_m_s"], "sat_omega_rad_s": r["sat_omega_rad_s"],
        "dissipated_energy_J": r["dissipated_energy_J"],
        "dissipated_fraction": r["dissipated_fraction"],
    })


def mass_ratio_sweep(s, cfg):
    """Head-on hit through the CoM: the classic 1-D collision, m/M from tiny to huge."""
    ratio = np.logspace(np.log10(cfg["ratio_min"]), np.log10(cfg["ratio_max"]), cfg["n_points"])
    frames = []
    for e in cfg["restitution_values"]:
        r = _analytic(s, m=ratio * s.sat_mass, angle=0.0, e=e, arm=np.zeros(2))
        # along -n is "forward" for the debris; express both results as fractions of v0
        n = s.panel_normal
        frames.append(pd.DataFrame({
            "mass_ratio_m_over_M": ratio, "restitution": e,
            "sat_dv_over_v0": r["sat_dv_mag_m_s"] / s.speed,
            "debris_v_after_over_v0": -(r["debris_vx_after_m_s"] * n[0]
                                        + r["debris_vy_after_m_s"] * n[1]) / s.speed,
            "dissipated_fraction": r["dissipated_fraction"],
        }))
    return pd.concat(frames, ignore_index=True)


def monte_carlo(s, cfg, constants):
    """Random debris population hitting the same panel point. Reproducible (fixed seed)."""
    np.random.seed(cfg["seed"])
    N = int(cfg["n_impacts"])
    # mass: truncated power law, cumulative N(>m) ~ m^-alpha  (inverse-CDF sampling)
    alpha, m_lo, m_hi = cfg["mass_cumulative_power_index"], cfg["mass_min_kg"], cfg["mass_max_kg"]
    u = np.random.uniform(size=N)
    mass = (m_lo ** -alpha - u * (m_lo ** -alpha - m_hi ** -alpha)) ** (-1.0 / alpha)
    # speed: normal, truncated by resampling outside [min, max]
    speed = np.random.normal(cfg["speed_mean_m_s"], cfg["speed_std_m_s"], N)
    bad = (speed < cfg["speed_min_m_s"]) | (speed > cfg["speed_max_m_s"])
    while bad.any():
        speed[bad] = np.random.normal(cfg["speed_mean_m_s"], cfg["speed_std_m_s"], bad.sum())
        bad = (speed < cfg["speed_min_m_s"]) | (speed > cfg["speed_max_m_s"])
    # angle: isotropic flux on a flat plate, p(theta) ~ sin(theta) cos(theta)
    angle = np.arcsin(np.sqrt(np.random.uniform(size=N)))
    e = np.random.uniform(cfg["restitution_min"], cfg["restitution_max"], N)

    r = _analytic(s, m=mass, speed=speed, angle=angle, e=e)
    spin_deg_s = np.degrees(np.abs(r["sat_omega_rad_s"]))
    return pd.DataFrame({
        "debris_mass_kg": mass, "impact_speed_m_s": speed,
        "impact_angle_deg": np.degrees(angle), "restitution": e,
        "sat_dv_mag_m_s": r["sat_dv_mag_m_s"], "sat_spin_deg_s": spin_deg_s,
        "dissipated_energy_J": r["dissipated_energy_J"], "emr_J_kg": r["emr_J_kg"],
        "catastrophic_breakup": r["emr_J_kg"] >= constants.emr_threshold,
        "spin_alarm": spin_deg_s >= cfg["spin_rate_alarm_deg_s"],
    })


def compare_with_analytic(s, summary):
    """Numerical (contact simulation) vs analytic (impulse-momentum) results."""
    ref = analytic_for(s)
    dv = float(ref["sat_dv_mag_m_s"])
    # error is measured relative to the natural size of each quantity (a component
    # that is exactly zero would otherwise give a meaningless relative error)
    scales = {"sat_dv_x_m_s": dv, "sat_dv_y_m_s": dv,
              "sat_omega_rad_s": max(abs(float(ref["sat_omega_rad_s"])),
                                     dv / max(np.linalg.norm(s.panel_point), 1.0)),
              "debris_vx_after_m_s": s.speed, "debris_vy_after_m_s": s.speed,
              "dissipated_energy_J": float(ref["impact_energy_J"])}
    rows = []
    for key, scale in scales.items():
        num, ana = float(summary[key]), float(ref[key])
        rows.append({"scenario": s.name, "quantity": key, "numerical": num,
                     "analytic": ana, "abs_error": abs(num - ana),
                     "rel_error": abs(num - ana) / scale})
    return pd.DataFrame(rows)


def extreme_cases(base, cases, constants):
    rows = []
    for case in cases:
        s = override_scenario(base, case)
        _, summ = solve_impact(s)
        ref = analytic_for(s)
        checks = pm.validity_checks(s.debris_mass, s.sat_mass, s.speed, constants)
        rows.append({
            "case": case["name"], "debris_mass_kg": s.debris_mass,
            "impact_angle_deg": s.impact_angle_deg, "restitution": s.restitution,
            "panel_arm_m": float(np.linalg.norm(s.panel_point)),
            "sat_dv_numerical_m_s": summ["sat_dv_mag_m_s"],
            "sat_dv_analytic_m_s": float(ref["sat_dv_mag_m_s"]),
            "sat_omega_numerical_rad_s": summ["sat_omega_rad_s"],
            "sat_omega_analytic_rad_s": float(ref["sat_omega_rad_s"]),
            "debris_v_after_m_s": float(np.hypot(summ["debris_vx_after_m_s"],
                                                 summ["debris_vy_after_m_s"])),
            "dissipated_fraction": float(ref["dissipated_fraction"]),
            "momentum_transfer_beta": float(ref["momentum_transfer_beta"]),
            "max_momentum_rel_error": summ["max_momentum_rel_error"],
            "max_energy_rel_error": summ["max_energy_rel_error"],
            "emr_J_kg": checks["emr_J_kg"],
            "catastrophic_breakup": checks["catastrophic_breakup"],
        })
    return pd.DataFrame(rows)
