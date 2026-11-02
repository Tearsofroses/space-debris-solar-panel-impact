"""Output generation: figures (titles, labelled axes with units, legends, grids)
for every scenario and for the parametric studies."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

import physics_model as pm  # noqa: E402

# Categorical colours (fixed order, colour-blind checked) and neutral inks
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
MILLI, KILO = 1e3, 1e-3   # SI-prefix scale factors, for plotting only

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelcolor": INK,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linestyle": "--", "grid.linewidth": 0.6,
    "lines.linewidth": 1.8, "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False, "legend.fontsize": 8,
    "axes.prop_cycle": matplotlib.cycler(color=[C1, C2, C3, C4]),
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
# Per-scenario figure: momentum and energy budget before / after the impact
# ----------------------------------------------------------------------------
def plot_budget(summary, s, out):
    n = s.panel_normal
    vd0 = pm.debris_velocity(s.speed, s.impact_angle, n)
    vd1 = np.array([summary["debris_vx_after_m_s"], summary["debris_vy_after_m_s"]])
    dV = np.array([summary["sat_dv_x_m_s"], summary["sat_dv_y_m_s"]])
    m, M = s.debris_mass, s.sat_mass

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    # (a) linear momentum, projected on the panel normal n and the panel tangent t
    labels, before, after = [], [], []
    for axis, e in (("n", n), ("t", pm.z_cross(n))):
        p_debris0, p_debris1, p_sat1 = m * np.dot(vd0, e), m * np.dot(vd1, e), M * np.dot(dV, e)
        labels += [f"Debris ({axis})", f"Satellite ({axis})", f"Total ({axis})"]
        before += [p_debris0, 0.0, p_debris0]
        after += [p_debris1, p_sat1, p_debris1 + p_sat1]
    x = np.arange(len(labels))
    a1.bar(x - 0.2, before, 0.4, color=C1, label="Before impact")
    a1.bar(x + 0.2, after, 0.4, color=C2, label="After impact")
    a1.axhline(0, color=INK2, lw=0.8)
    a1.set_xticks(x)
    a1.set_xticklabels(labels, fontsize=7)
    a1.set_ylabel("Momentum component p·n, p·t (kg m/s)")
    a1.set_title("Linear momentum: each total component is conserved")
    a1.legend()
    # (b) energy budget
    parts = [("Debris K after", summary["ke_debris_after_J"]),
             ("Satellite K (translation)", summary["ke_sat_translation_J"]),
             ("Satellite K (rotation)", summary["ke_sat_rotation_J"]),
             ("Dissipated (heat, crater)", summary["dissipated_energy_J"])]
    a2.bar(["Before: debris K"], [summary["impact_energy_J"]], color=C1)
    bottom = 0.0
    for (name, v), c in zip(parts, [C2, C3, C4, INK2]):
        a2.bar(["After"], [v], bottom=bottom, color=c, label=name)
        bottom += v
    a2.set_ylabel("Energy (J)")
    a2.set_title("Energy: K lost to heat / damage unless elastic")
    a2.set_ylim(top=1.3 * summary["impact_energy_J"])
    a2.legend(fontsize=7, loc="upper center", ncol=2)
    fig.suptitle(f"Momentum and energy budget - {s.name}", fontweight="bold")
    return _save(fig, out)


# ----------------------------------------------------------------------------
# Parametric-study figures
# ----------------------------------------------------------------------------
def plot_mass_speed_map(df, out):
    dv = df.pivot(index="impact_speed_m_s", columns="debris_mass_kg", values="sat_dv_mag_m_s")
    masses, speeds = dv.columns.to_numpy(), dv.index.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    mesh = ax.pcolormesh(masses, speeds * KILO, dv.values, cmap="Blues", shading="auto",
                         norm=LogNorm(vmin=dv.values.min(), vmax=dv.values.max()))
    fig.colorbar(mesh, ax=ax, label="Satellite velocity change |ΔV| (m/s)")
    ax.set_xscale("log")
    ax.set_title("Satellite ΔV vs debris mass and impact speed")
    ax.set_xlabel("Debris mass (kg)")
    ax.set_ylabel("Impact speed (km/s)")
    return _save(fig, out)


def plot_angle_sweep(df, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for kind, g in df.groupby("collision_type"):          # colours from axes.prop_cycle
        a1.plot(g["impact_angle_deg"], g["sat_dv_mag_m_s"] * MILLI, label=kind)
        a2.plot(g["impact_angle_deg"], np.degrees(np.abs(g["sat_omega_rad_s"])),
                label=kind)
    a1.set_title("Satellite ΔV vs impact angle")
    a1.set_xlabel("Impact angle from panel normal (deg)")
    a1.set_ylabel("|ΔV| (mm/s)")
    a2.set_title("Satellite spin rate vs impact angle")
    a2.set_xlabel("Impact angle from panel normal (deg)")
    a2.set_ylabel("|ω| (deg/s)")
    for a in (a1, a2):
        a.legend()
    return _save(fig, out)


def plot_mass_ratio_sweep(df, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for kind, g in df.groupby("collision_type"):          # colours from axes.prop_cycle
        a1.plot(g["mass_ratio_m_over_M"], g["sat_dv_over_v0"], label=kind)
        a2.plot(g["mass_ratio_m_over_M"], g["debris_v_after_over_v0"], label=kind)
    for a in (a1, a2):
        a.axvline(1.0, color=INK2, lw=0.8, ls=":")
        a.set_xscale("log")
    a2.axhline(0.0, color=INK2, lw=0.8)
    a1.set_yscale("log")
    a1.set_title("Head-on 1-D collision: target velocity")
    a1.set_xlabel("Mass ratio m / M")
    a1.set_ylabel("ΔV_satellite / v₀")
    a2.set_title("Head-on 1-D collision: debris velocity after")
    a2.set_xlabel("Mass ratio m / M")
    a2.set_ylabel("v_debris,after / v₀ (forward > 0)")
    for a in (a1, a2):
        a.legend()
    return _save(fig, out)


def plot_conservation(df, out):
    labels = [f"{r.scenario}\n{r.quantity}" for r in df.itertuples()]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(df))))
    ax.barh(range(len(df)), _floor(df["rel_error"]), color=C1, height=0.7,
            label="|after - before| / before")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.grid(False, axis="y")
    ax.set_title("Conservation check: momentum, angular momentum, energy balance")
    ax.set_xlabel("Relative residual (round-off level)")
    ax.legend(loc="lower right")
    return _save(fig, out)
