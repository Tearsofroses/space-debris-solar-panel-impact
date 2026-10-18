"""Output generation: publication-ready figures (titles, labelled axes with units,
legends, grids) for every scenario and for the parametric studies."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

# Categorical colours (fixed order, colour-blind checked) and neutral inks
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
US, MM, KM = 1e6, 1e3, 1e-3   # unit conversions for plotting only

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelcolor": INK,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linestyle": "--", "grid.linewidth": 0.6,
    "lines.linewidth": 1.8, "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _floor(err):
    """Log axes cannot show exact zeros; clip to machine precision."""
    return np.maximum(np.asarray(err, dtype=float), np.finfo(float).eps * 1e-2)


# ----------------------------------------------------------------------------
# Per-scenario figures
# ----------------------------------------------------------------------------
def plot_trajectory(imp, orb, s, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    # (a) impact geometry, relative to the struck point
    px, py = s.panel_point
    n = s.panel_normal
    t = np.array([-n[1], n[0]])
    dx, dy = (imp["debris_pos_x"] - px) * MM, (imp["debris_pos_y"] - py) * MM
    span = 1.15 * max(np.abs(dx).max(), np.abs(dy).max(), 1e-3)
    a1.plot([-span * t[0], span * t[0]], [-span * t[1], span * t[1]], color=INK2,
            lw=4, solid_capstyle="round", label="Solar panel surface (t = 0)")
    a1.annotate("", xy=(0.35 * span * n[0], 0.35 * span * n[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=INK2))
    a1.text(0.4 * span * n[0], 0.4 * span * n[1], "  n", color=INK2)
    contact = imp["in_contact"].to_numpy()
    a1.plot(dx[contact], dy[contact], color=C2, label="Debris (in contact)")
    a1.plot(dx[~contact], dy[~contact], color=C1, label="Debris (after separation)")
    a1.set_aspect("equal", adjustable="datalim")
    a1.set_title("Impact phase: debris path near the panel")
    a1.set_xlabel("x - x_impact, radial (mm)")
    a1.set_ylabel("y - y_impact, along-track (mm)")
    a1.legend(loc="best", fontsize=8)
    # (b) post-impact drift of the satellite from its original slot
    a2.plot(orb["pos_y"] * KM, orb["pos_x"] * KM, color=C1, label="Satellite (numerical CW)")
    a2.plot(orb["analytic_pos_y"] * KM, orb["analytic_pos_x"] * KM, color=C2, ls=":",
            lw=2.4, label="Analytic CW solution")
    a2.plot(0, 0, "o", ms=8, color=INK, label="Original orbital slot")
    a2.set_title(f"Post-impact drift over {s.n_orbits:g} orbits (LVLH frame)")
    a2.set_xlabel("Along-track offset y (km)")
    a2.set_ylabel("Radial offset x (km)")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)
    fig.suptitle(f"Trajectories - {s.name}", fontweight="bold")
    return _save(fig, out)


def plot_time_series(imp, s, out):
    t = imp["time_t"] * US
    fig, ax = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    ax[0].plot(t, imp["penetration_m"] * 1e6, color=C1, label="Penetration depth δ")
    ax[0].set_ylabel("δ (µm)")
    ax[1].plot(t, imp["contact_force_N"] * 1e-3, color=C2, label="Normal contact force f")
    ax[1].set_ylabel("Force (kN)")
    ax[2].plot(t, imp["debris_vel_x"] * KM, color=C1, label="Debris v_x (radial)")
    ax[2].plot(t, imp["debris_vel_y"] * KM, color=C2, label="Debris v_y (along-track)")
    ax[2].set_ylabel("Velocity (km/s)")
    ax[3].plot(t, imp["vel_x"] * MM, color=C1, label="Satellite V_x (radial)")
    ax[3].plot(t, imp["vel_y"] * MM, color=C2, label="Satellite V_y (along-track)")
    ax[3].set_ylabel("Velocity (mm/s)")
    ax[4].plot(t, np.degrees(imp["angular_velocity_rad_s"]), color=C3,
               label="Satellite spin rate ω")
    ax[4].set_ylabel("ω (deg/s)")
    ax[4].set_xlabel("Time (µs)")
    for a in ax:
        a.legend(loc="best", fontsize=8)
    ax[0].set_title(f"Time series of state variables during the impact - {s.name}")
    return _save(fig, out)


def plot_phase_space(imp, orb, s, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    c = imp["in_contact"].to_numpy()
    a1.plot(imp["penetration_m"][c] * 1e6, imp["penetration_rate_m_s"][c] * KM, color=C1,
            label="Debris relative to panel (contact)")
    a1.plot(imp["penetration_m"].iloc[0] * 1e6, imp["penetration_rate_m_s"].iloc[0] * KM,
            "o", ms=8, color=C2, label="First contact")
    a1.axhline(0, color=INK2, lw=0.8)
    a1.set_title("Impact phase portrait (normal direction)")
    a1.set_xlabel("Penetration depth δ (µm)")
    a1.set_ylabel("Penetration rate dδ/dt (km/s)")
    a1.legend(loc="best", fontsize=8)
    a2.plot(orb["pos_x"], orb["vel_x"] * MM, color=C1, label="Satellite, radial motion")
    a2.plot(orb["pos_x"].iloc[0], orb["vel_x"].iloc[0] * MM, "o", ms=8, color=C2,
            label="Just after impact")
    a2.set_title("Post-impact phase space (radial direction)")
    a2.set_xlabel("Radial offset x (m)")
    a2.set_ylabel("Radial velocity v_x (mm/s)")
    a2.legend(loc="best", fontsize=8)
    fig.suptitle(f"Phase space diagrams - {s.name}", fontweight="bold")
    return _save(fig, out)


def plot_energy(imp, orb, s, out):
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    t = imp["time_t"] * US
    ax[0, 0].plot(t, imp["kinetic_energy"] * KM, color=C1, label="Kinetic energy K")
    ax[0, 0].plot(t, imp["potential_energy"] * KM, color=C2, label="Contact spring energy U")
    ax[0, 0].plot(t, imp["dissipated_energy"] * KM, color=C3, label="Dissipated (heat, crater)")
    ax[0, 0].plot(t, imp["total_energy"] * KM, color=INK, ls="--", label="Total K + U + E_diss")
    ax[0, 0].set_title("Impact phase: energy budget")
    ax[0, 0].set_xlabel("Time (µs)")
    ax[0, 0].set_ylabel("Energy (kJ)")
    ax[1, 0].plot(t, _floor(imp["energy_rel_error"]), color=C2,
                  label="|E(t) - E(0)| / E(0)")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_title("Impact phase: fractional energy error")
    ax[1, 0].set_xlabel("Time (µs)")
    ax[1, 0].set_ylabel("Fractional error (ΔE/E₀)")

    to = orb["time_t"] / 60.0
    ax[0, 1].plot(to, orb["kinetic_energy"], color=C1, label="Relative kinetic energy")
    ax[0, 1].plot(to, orb["potential_energy"], color=C2, label="Tidal potential -3/2 M n² x²")
    ax[0, 1].plot(to, orb["total_energy"], color=INK, ls="--", label="Jacobi integral (total)")
    ax[0, 1].set_title("Post-impact drift: CW energy")
    ax[0, 1].set_xlabel("Time after impact (min)")
    ax[0, 1].set_ylabel("Energy (J)")
    ax[1, 1].plot(to, _floor(orb["energy_rel_error"]), color=C2,
                  label="|E(t) - E(0)| / E(0)")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_title("Post-impact drift: fractional energy error")
    ax[1, 1].set_xlabel("Time after impact (min)")
    ax[1, 1].set_ylabel("Fractional error (ΔE/E₀)")
    for a in ax.ravel():
        a.grid(True, which="both", ls="--")
        a.legend(loc="best", fontsize=8)
    fig.suptitle(f"Energy conservation and integration error - {s.name}", fontweight="bold")
    return _save(fig, out)


def plot_momentum(imp, s, out):
    t = imp["time_t"] * US
    n = s.panel_normal
    proj = lambda px, py: px * n[0] + py * n[1]   # component along the panel normal
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    a1.plot(t, proj(imp["debris_momentum_x"], imp["debris_momentum_y"]), color=C1,
            label="Debris p·n")
    a1.plot(t, proj(imp["sat_momentum_x"], imp["sat_momentum_y"]), color=C2,
            label="Satellite p·n")
    a1.plot(t, proj(imp["momentum_x"], imp["momentum_y"]), color=INK, ls="--",
            label="Total p·n (conserved)")
    a1.set_title(f"Linear momentum along the panel normal - {s.name}")
    a1.set_ylabel("Momentum (kg m/s)")
    a2.plot(t, _floor(imp["momentum_rel_error"]), color=C1, label="Linear momentum |ΔP|/|P₀|")
    a2.plot(t, _floor(imp["angular_momentum_rel_error"]), color=C2,
            label="Angular momentum |ΔL|/L_ref")
    a2.set_yscale("log")
    a2.set_title("Conservation error of momentum")
    a2.set_xlabel("Time (µs)")
    a2.set_ylabel("Relative error")
    for a in (a1, a2):
        a.grid(True, which="both", ls="--")
        a.legend(loc="best", fontsize=8)
    return _save(fig, out)


def scenario_figures(imp, orb, s, folder):
    folder = Path(folder)
    return [
        plot_trajectory(imp, orb, s, folder / "01_trajectory.png"),
        plot_time_series(imp, s, folder / "02_time_series.png"),
        plot_phase_space(imp, orb, s, folder / "03_phase_space.png"),
        plot_energy(imp, orb, s, folder / "04_energy_error.png"),
        plot_momentum(imp, s, folder / "05_momentum_conservation.png"),
    ]


# ----------------------------------------------------------------------------
# Parametric-study figures
# ----------------------------------------------------------------------------
def plot_mass_speed_map(df, emr_threshold, out):
    masses = np.sort(df["debris_mass_kg"].unique())
    speeds = np.sort(df["impact_speed_m_s"].unique())
    dv = df.pivot(index="impact_speed_m_s", columns="debris_mass_kg", values="sat_dv_mag_m_s")
    emr = df.pivot(index="impact_speed_m_s", columns="debris_mass_kg", values="emr_J_kg")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    mesh = ax.pcolormesh(masses, speeds * KM, dv.values, cmap="Blues", shading="auto",
                         norm=LogNorm(vmin=dv.values.min(), vmax=dv.values.max()))
    fig.colorbar(mesh, ax=ax, label="Satellite velocity change |ΔV| (m/s)")
    cs = ax.contour(masses, speeds * KM, emr.values, levels=[emr_threshold], colors=C2,
                    linewidths=2)
    ax.clabel(cs, fmt={emr_threshold: "EMR = 40 J/g"}, fontsize=8)
    ax.plot([], [], color=C2, lw=2, label="Catastrophic breakup limit (model invalid beyond)")
    ax.set_xscale("log")
    ax.set_title("Satellite ΔV vs debris mass and impact speed")
    ax.set_xlabel("Debris mass (kg)")
    ax.set_ylabel("Impact speed (km/s)")
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    return _save(fig, out)


def plot_angle_sweep(df, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for colour, (e, g) in zip((C1, C2, C3), df.groupby("restitution")):
        a1.plot(g["impact_angle_deg"], g["sat_dv_mag_m_s"] * MM, color=colour, label=f"e = {e:g}")
        a2.plot(g["impact_angle_deg"], np.degrees(np.abs(g["sat_omega_rad_s"])), color=colour,
                label=f"e = {e:g}")
    a1.set_title("Satellite ΔV vs impact angle")
    a1.set_xlabel("Impact angle from panel normal (deg)")
    a1.set_ylabel("|ΔV| (mm/s)")
    a2.set_title("Satellite spin rate vs impact angle")
    a2.set_xlabel("Impact angle from panel normal (deg)")
    a2.set_ylabel("|ω| (deg/s)")
    for a in (a1, a2):
        a.legend(loc="best", fontsize=8)
    return _save(fig, out)


def plot_restitution_sweep(df, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.plot(df["restitution"], df["momentum_transfer_beta"], color=C1,
            label="β = J / (m v_n)")
    a1.set_title("Momentum transfer efficiency")
    a1.set_xlabel("Coefficient of restitution e")
    a1.set_ylabel("β (dimensionless)")
    a2.plot(df["restitution"], df["dissipated_fraction"] * 100, color=C2,
            label="Kinetic energy dissipated")
    a2.set_title("Energy lost in the collision")
    a2.set_xlabel("Coefficient of restitution e")
    a2.set_ylabel("Fraction of impact energy (%)")
    for a in (a1, a2):
        a.legend(loc="best", fontsize=8)
    return _save(fig, out)


def plot_mass_ratio_sweep(df, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for colour, (e, g) in zip((C1, C2, C3), df.groupby("restitution")):
        a1.plot(g["mass_ratio_m_over_M"], g["sat_dv_over_v0"], color=colour, label=f"e = {e:g}")
        a2.plot(g["mass_ratio_m_over_M"], g["debris_v_after_over_v0"], color=colour,
                label=f"e = {e:g}")
    a1.axvline(1.0, color=INK2, lw=0.8, ls=":")
    a2.axvline(1.0, color=INK2, lw=0.8, ls=":")
    a2.axhline(0.0, color=INK2, lw=0.8)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_title("Head-on 1-D collision: target velocity")
    a1.set_xlabel("Mass ratio m / M")
    a1.set_ylabel("ΔV_satellite / v₀")
    a2.set_xscale("log")
    a2.set_title("Head-on 1-D collision: debris velocity after")
    a2.set_xlabel("Mass ratio m / M")
    a2.set_ylabel("v_debris,after / v₀ (forward > 0)")
    for a in (a1, a2):
        a.legend(loc="best", fontsize=8)
    return _save(fig, out)


def plot_monte_carlo(df, emr_threshold, spin_alarm, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    bins = np.logspace(np.log10(df["sat_dv_mag_m_s"].min()),
                       np.log10(df["sat_dv_mag_m_s"].max()), 60)
    a1.hist(df["sat_dv_mag_m_s"], bins=bins, color=C1, edgecolor="white", linewidth=0.5,
            label=f"{len(df):,} random impacts")
    a1.set_xscale("log")
    a1.set_title("Monte Carlo: distribution of satellite ΔV")
    a1.set_xlabel("|ΔV| (m/s)")
    a1.set_ylabel("Number of impacts")
    a1.legend(loc="best", fontsize=8)
    safe = ~df["catastrophic_breakup"]
    a2.scatter(df.loc[safe, "debris_mass_kg"], df.loc[safe, "sat_spin_deg_s"], s=4,
               color=C1, alpha=0.4, linewidths=0, label="Non-catastrophic")
    if (~safe).any():
        a2.scatter(df.loc[~safe, "debris_mass_kg"], df.loc[~safe, "sat_spin_deg_s"], s=8,
                   color=C2, linewidths=0, label=f"EMR ≥ {emr_threshold / 1e3:g} J/g")
    a2.axhline(spin_alarm, color=INK2, ls="--", lw=1, label=f"Spin alarm {spin_alarm:g} deg/s")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_title("Monte Carlo: spin rate given to the satellite")
    a2.set_xlabel("Debris mass (kg)")
    a2.set_ylabel("|ω| (deg/s)")
    a2.legend(loc="best", fontsize=8)
    return _save(fig, out)


def plot_validation(df, out):
    labels = [f"{r.scenario}\n{r.quantity}" for r in df.itertuples()]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(df))))
    ax.barh(range(len(df)), _floor(df["rel_error"]), color=C1, height=0.7,
            label="|numerical - analytic| / scale")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.grid(False, axis="y")
    ax.set_title("Validation: time-resolved contact simulation vs impulse-momentum formula")
    ax.set_xlabel("Relative difference")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, out)
